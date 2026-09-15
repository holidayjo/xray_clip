"""Build an h5 of LUNG-CROPPED images, mirroring run_preprocess.py otherwise.

Why: the ChestX-ray14 benchmark we compare against (Mezina & Burget 2024) segments the lungs
with a pre-trained U-Net and crops to that region before classifying, at 384x384. Our pipeline
feeds the whole padded image at 224x224. That is a confound on top of supervised-vs-zero-shot,
and it plausibly explains our weakest labels: a nodule occupies a handful of pixels in a whole
224px frame. Cropping to the lungs raises effective resolution ~1.3-2x linear at no extra cost.

Cropping must be applied to TRAINING data as well as test, or the encoder sees a different
image distribution at inference than it was trained on.

Pipeline (single pass over the source images, since they live on a slow HDD):
    worker  : decode JPEG/PNG -> grayscale, downsample to long side <= 1024   (CPU, pooled)
    main    : batch -> PSPNet lung segmentation -> bbox -> crop -> preprocess() -> h5   (GPU)

Falls back to the uncropped image when segmentation finds too little lung, and reports how
often that happened.
"""

import argparse
import os
from pathlib import Path
from multiprocessing import Pool

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_process import preprocess, get_cxr_paths_list

LUNG_CHANNELS = [4, 5]      # 'Left Lung', 'Right Lung' in PSPNet.targets
WORK_SIZE = 1024            # long side kept for cropping; final output is 320 anyway
SEG_SIZE = 512              # PSPNet input


def _load_one(path):
    """Decode and downsample. Runs in a pool worker; returns a modest array to keep IPC cheap."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, f"unreadable: {path}"
        h, w = img.shape
        s = WORK_SIZE / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return img, None
    except Exception as e:
        return None, repr(e)


def lung_boxes(batch_imgs, seg, device, thr=0.5, margin=0.03):
    """Bounding box of the lung mask for each image, as fractional (x0,y0,x1,y1) or None."""
    import torchxrayvision as xrv
    stack = np.stack([
        xrv.datasets.normalize(
            cv2.resize(im, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32), 255)
        for im in batch_imgs])
    t = torch.from_numpy(stack)[:, None].to(device)
    with torch.no_grad():
        m = torch.sigmoid(seg(t))[:, LUNG_CHANNELS].amax(1).cpu().numpy()

    out = []
    for mask in m:
        ys, xs = np.where(mask > thr)
        if len(xs) < 100:                       # segmentation effectively failed
            out.append(None)
            continue
        x0, x1 = xs.min() / SEG_SIZE, xs.max() / SEG_SIZE
        y0, y1 = ys.min() / SEG_SIZE, ys.max() / SEG_SIZE
        mx, my = margin * (x1 - x0), margin * (y1 - y0)
        out.append((max(0., x0 - mx), max(0., y0 - my), min(1., x1 + mx), min(1., y1 + my)))
    return out


def build(cxr_paths, out_filepath, resolution=320, num_workers=None, batch=32, box_csv=None):
    from torchxrayvision.baseline_models.chestx_det import PSPNet
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seg = PSPNet().to(device).eval()
    if num_workers is None:
        num_workers = max(1, min(12, (os.cpu_count() or 2) - 6))

    n = len(cxr_paths)
    failed, no_crop, boxes = [], 0, []

    with h5py.File(out_filepath, "w") as h5f:
        dset = h5f.create_dataset("cxr", shape=(n, resolution, resolution), dtype="uint8")
        idx = 0
        buf_img, buf_i = [], []

        def flush():
            nonlocal idx, no_crop
            if not buf_img:
                return
            for i, im, box in zip(buf_i, buf_img, lung_boxes(buf_img, seg, device)):
                if box is None:
                    no_crop += 1
                    crop = im
                else:
                    h, w = im.shape
                    x0, y0, x1, y1 = box
                    crop = im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
                    if crop.size == 0:
                        no_crop += 1
                        crop = im
                boxes.append((str(cxr_paths[i]),) + (box if box else (np.nan,) * 4))
                from PIL import Image
                dset[i] = np.asarray(preprocess(Image.fromarray(crop), desired_size=resolution),
                                     dtype=np.uint8)
            buf_img.clear(); buf_i.clear()

        with Pool(num_workers) as pool:
            for i, (img, err) in enumerate(tqdm(pool.imap(_load_one, cxr_paths, chunksize=8), total=n)):
                if err is not None:
                    failed.append((cxr_paths[i], err))
                    continue
                buf_img.append(img); buf_i.append(i)
                if len(buf_img) == batch:
                    flush()
            flush()

    print(f"{len(failed)} / {n} images failed to load.", failed[:3])
    print(f"{no_crop} / {n} ({100*no_crop/max(n,1):.2f}%) had no usable lung mask -> kept uncropped")
    if box_csv:
        pd.DataFrame(boxes, columns=["Path", "x0", "y0", "x1", "y1"]).to_csv(box_csv, index=False)
        print(f"wrote boxes to {box_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paths_csv", required=True,
                   help="CSV with a 'Path' column, e.g. data/cxr_paths.csv or an eval paths csv. "
                        "Reusing the existing file preserves row order and hence alignment.")
    p.add_argument("--cxr_out_path", required=True)
    p.add_argument("--box_csv", default=None, help="Optional: record the crop box per image.")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    paths = get_cxr_paths_list(a.paths_csv).tolist()
    print(f"{len(paths):,} images from {a.paths_csv}")
    build(paths, a.cxr_out_path, num_workers=a.num_workers, batch=a.batch, box_csv=a.box_csv)
