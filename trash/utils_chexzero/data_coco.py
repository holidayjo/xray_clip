"""STEP 1 -- the data.

The whole idea of CheXzero rests on a distinction that lives entirely in this file, so it is
worth stating before any code:

    TRAINING sees   (image, free text)          <- no labels, ever
    EVALUATION sees (image, multi-hot labels)   <- labels exist ONLY to score with

That is what "zero-shot" means in the paper. The model is never told "this X-ray has
cardiomegaly". It is only ever told "this X-ray goes with this report". The pathology names do
appear during training -- but as WORDS INSIDE THE REPORTS, not as a label column.

Mapping the paper onto COCO, which is the same structure in a different domain:

    paper                             COCO
    ---------------------------       -------------------------------------
    chest X-ray                       natural photograph
    radiology report (impressions)    caption
    377,110 image-report pairs        118,287 image-caption pairs
    14 pathologies, multi-label       80 object categories, multi-label
    "{pathology}" / "no {pathology}"  "{category}" / "no {category}"

So this file provides exactly two datasets:

    CocoCaptionPairs -> (image_tensor, caption_str)   for contrastive TRAINING
    CocoMultiLabel   -> (image_tensor, multihot[80])  for zero-shot EVALUATION
"""
import json
import pathlib

import numpy as np
import torch
import torch.utils.data
import PIL.Image
import torchvision.transforms as T

COCO_ROOT = pathlib.Path("/mnt/Documents/Dad/dataset/coco")


# ---------------------------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------------------------
# The paper (Methods, "Pre-processing"): "Each of the 377,110 chest X-rays in the MIMIC-CXR
# dataset were re-sized to 224 x 224 and zero padded before training."
#
# Note this is NOT what CLIP does by default. CLIP resizes the SHORT side to 224 then
# centre-crops, which throws away the edges of the image. The paper instead scales the LONG
# side to 224 and pads the short side with zeros, so the whole image survives -- sensible for a
# radiograph, where a finding at the edge of the lung field matters.
class ResizeAndZeroPad:
    """Scale the long side to `size`, keep aspect ratio, pad the rest with zeros (black)."""
    def __init__(self, size=224):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), PIL.Image.BICUBIC)

        # paste onto a black square so the result is exactly size x size
        canvas = PIL.Image.new(img.mode, (self.size, self.size), 0)
        canvas.paste(img, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return canvas


def build_transform(size=224, mean=None, std=None):
    """Returns the image pipeline: zero-padded resize -> tensor -> normalise.

    The paper normalises with "a sample mean and standard deviation of the training dataset".
    For chest X-rays that is the right call, because radiographs are grayscale and nothing like
    the web photos CLIP was trained on. For COCO the opposite is true -- COCO images ARE the
    kind of data CLIP saw -- so the CLIP statistics are the better default here, and passing
    dataset statistics is offered only for faithfulness to the paper's recipe.
    """
    # OpenAI CLIP's normalisation constants
    mean = mean if mean is not None else (0.48145466, 0.4578275, 0.40821073)
    std  = std  if std  is not None else (0.26862954, 0.26130258, 0.27577711)
    return T.Compose([
        ResizeAndZeroPad(size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------------------------
# Reading the COCO annotation JSONs
# ---------------------------------------------------------------------------------------------
def load_caption_pairs(split="train2017", root=COCO_ROOT):
    """Returns [(image_path, [caption, caption, ...]), ...].

    COCO gives ~5 captions per image. The paper uses ONE text per study (the impressions
    section of the report), so during training we will sample one caption per image per epoch
    rather than treating all five as separate examples -- see CocoCaptionPairs below.
    """
    root = pathlib.Path(root)
    ann  = json.load(open(root / "annotations" / f"captions_{split}.json"))

    id_to_file = {im["id"]: im["file_name"] for im in ann["images"]}
    by_image   = {}
    for a in ann["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a["caption"].strip())

    img_dir = root / "images" / split
    pairs   = [(str(img_dir / id_to_file[i]), caps) for i, caps in by_image.items()]
    pairs.sort(key=lambda p: p[0])                       # deterministic order
    return pairs


def load_multilabel(split="val2017", root=COCO_ROOT):
    """Returns (image_paths, multihot[N, 80], category_names).

    These labels are the EVALUATION targets only. They are derived from COCO's object
    instance annotations: an image gets a 1 for every category that has at least one annotated
    object in it -- which makes this a genuine multi-label problem, exactly like a chest X-ray
    that shows several findings at once.
    """
    root = pathlib.Path(root)
    ann  = json.load(open(root / "annotations" / f"instances_{split}.json"))

    cats  = sorted(ann["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in cats]
    cat_index = {c["id"]: k for k, c in enumerate(cats)}   # category_id -> column 0..79

    id_to_file = {im["id"]: im["file_name"] for im in ann["images"]}
    labels = {i: np.zeros(len(cats), dtype=np.float32) for i in id_to_file}
    for a in ann["annotations"]:
        labels[a["image_id"]][cat_index[a["category_id"]]] = 1.0

    img_dir = root / "images" / split
    ids     = sorted(id_to_file)
    paths   = [str(img_dir / id_to_file[i]) for i in ids]
    Y       = np.stack([labels[i] for i in ids])
    return paths, Y, names


# ---------------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------------
class CocoCaptionPairs(torch.utils.data.Dataset):
    """TRAINING data: yields (image_tensor, caption_string). No labels anywhere.

    caption_mode:
      'sample' -- pick one of the image's captions at random each time it is drawn. Mirrors the
                  paper's one-text-per-study setup while still exposing the model to the
                  variety of phrasings across epochs.
      'first'  -- always use the first caption (fully deterministic, useful for debugging).
    """
    def __init__(self, pairs, transform, caption_mode="sample", seed=42):
        self.pairs = pairs
        self.transform = transform
        self.caption_mode = caption_mode
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, caps = self.pairs[i]
        img = PIL.Image.open(path).convert("RGB")
        cap = caps[0] if self.caption_mode == "first" else caps[self.rng.integers(len(caps))]
        return self.transform(img), cap


class CocoMultiLabel(torch.utils.data.Dataset):
    """EVALUATION data: yields (image_tensor, multihot[80]). Used only for scoring."""
    def __init__(self, paths, Y, transform):
        self.paths = paths
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = PIL.Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), torch.from_numpy(self.Y[i])


# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # A quick self-check so this file can be run on its own before anything depends on it.
    tf = build_transform()

    pairs = load_caption_pairs("val2017")
    print(f"caption pairs (val2017): {len(pairs)} images")
    print(f"  captions on the first image: {len(pairs[0][1])}")
    print(f"  example text: \"{pairs[0][1][0]}\"")

    ds = CocoCaptionPairs(pairs, tf)
    img, cap = ds[0]
    print(f"\nTRAIN item -> image {tuple(img.shape)}  text: \"{cap[:60]}\"")
    print(f"  pixel range after normalise: [{img.min():.2f}, {img.max():.2f}]")

    paths, Y, names = load_multilabel("val2017")
    print(f"\nEVAL split: {len(paths)} images, {Y.shape[1]} categories")
    print(f"  labels per image: mean {Y.sum(1).mean():.2f}, max {int(Y.sum(1).max())}")
    print(f"  rarest 3 : {[names[k] for k in Y.sum(0).argsort()[:3]]}")
    print(f"  commonest 3: {[names[k] for k in Y.sum(0).argsort()[::-1][:3]]}")

    eds = CocoMultiLabel(paths, Y, tf)
    im, y = eds[0]
    print(f"\nEVAL item -> image {tuple(im.shape)}  label vector {tuple(y.shape)}, "
          f"positives: {[names[k] for k in torch.nonzero(y).flatten().tolist()]}")
