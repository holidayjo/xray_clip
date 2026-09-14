"""Rank training checkpoints on the CheXpert validation set.

Follows the model-selection procedure in Tiu et al. (2022): score every checkpoint by
mean AUC over the five CheXpert competition tasks on the CheXpert validation set, then
ensemble the top ten.

The scoring is mathematically the same as zero_shot.run_softmax_eval, but restructured
for speed: upstream runs the image encoder once for the positive prompt and again for
the negative one, on CPU, at batch size 1. Image features do not depend on the prompt,
so here each image is encoded once per checkpoint on the GPU and both prompt sets are
applied to the cached features. --verify checks this against upstream directly.
"""

import argparse
import os
import re
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

import clip
from zero_shot import CXRTestDataset, load_clip

# The five CheXpert competition tasks used for checkpoint selection.
CXR_LABELS_5: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
CXR_PAIR_TEMPLATE = ("{}", "no {}")


def build_groundtruth(valid_csv: str, paths_csv: str, labels: List[str]) -> np.ndarray:
    """Build a label matrix whose row order matches the h5 row order.

    The h5 rows follow the sorted image paths recorded by run_preprocess.py, which are
    absolute. valid.csv paths are relative to the CheXpert release root. Both are keyed
    on the 'patientX/studyY/viewZ.jpg' suffix so the join cannot silently misalign.
    """
    key = r"(patient\d+/study\d+/view\d+[^/]*\.jpg)$"

    order = pd.read_csv(paths_csv)["Path"].str.extract(key, expand=False)
    if order.isna().any():
        raise ValueError(f"{paths_csv}: {int(order.isna().sum())} paths did not match {key}")

    gt = pd.read_csv(valid_csv)
    gt["key"] = gt["Path"].str.extract(key, expand=False)
    gt = gt.dropna(subset=["key"]).set_index("key")

    missing = set(order) - set(gt.index)
    if missing:
        raise ValueError(f"{len(missing)} h5 rows have no label row, e.g. {sorted(missing)[:3]}")

    y_true = gt.loc[order, labels].to_numpy(dtype=float)
    if np.isnan(y_true).any():
        raise ValueError("NaN in ground truth; the radiologist-labelled valid set should be 0/1")
    return y_true


def make_loader(cxr_filepath: str, batch_size: int) -> torch.utils.data.DataLoader:
    transform = Compose([
        # means computed from sample in `cxr_stats` notebook
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(224, interpolation=InterpolationMode.BICUBIC),
    ])
    dset = CXRTestDataset(img_path=cxr_filepath, transform=transform)
    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)


def text_weights(model, labels: List[str], template: str, device, context_length: int) -> torch.Tensor:
    """Normalised text embedding per label for one prompt template. (embed_dim, num_labels)"""
    with torch.no_grad():
        embs = []
        for name in labels:
            tokens = clip.tokenize([template.format(name)], context_length=context_length).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            # upstream averages over templates then renormalises; with a single
            # template both steps are identity, so the result is the same vector.
            embs.append(emb[0])
        return torch.stack(embs, dim=1)


def score_checkpoint(model, loader, labels, device, context_length: int) -> np.ndarray:
    """Softmax probabilities, (num_images, num_labels)."""
    model.eval()
    pos_w = text_weights(model, labels, CXR_PAIR_TEMPLATE[0], device, context_length)
    neg_w = text_weights(model, labels, CXR_PAIR_TEMPLATE[1], device, context_length)

    pos_logits, neg_logits = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device)
            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            pos_logits.append((feats @ pos_w).cpu().numpy())
            neg_logits.append((feats @ neg_w).cpu().numpy())

    pos = np.concatenate(pos_logits, axis=0)
    neg = np.concatenate(neg_logits, axis=0)
    return np.exp(pos) / (np.exp(pos) + np.exp(neg))


def mean_auc(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]):
    aucs = [roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(len(labels))]
    return float(np.mean(aucs)), aucs


def pick_spaced(df: pd.DataFrame, k: int, min_gap: int) -> pd.DataFrame:
    """Best k rows by mean_auc, requiring at least min_gap steps between members.

    df must already be sorted best-first. With min_gap == 0 this is plain top-k.
    """
    if min_gap <= 0:
        return df.head(k)
    chosen = []
    for _, row in df.iterrows():
        if all(abs(row.step - c.step) >= min_gap for c in chosen):
            chosen.append(row)
        if len(chosen) == k:
            break
    return pd.DataFrame(chosen).reset_index(drop=True)


def checkpoint_step(path: str) -> int:
    m = re.search(r"checkpoint_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", default="checkpoints/pt-imp",
                   help="Directory containing checkpoint_*.pt files.")
    p.add_argument("--cxr_filepath", default="data/chexpert_valid.h5",
                   help="h5 of the CheXpert validation images (chexpert-valid preprocessing).")
    p.add_argument("--paths_csv", default="data/chexpert_valid_paths.csv",
                   help="Ordered image paths written alongside the validation h5.")
    p.add_argument("--valid_csv", default="/mnt/My_Doc/dataset/CheXpert/kaggle/valid.csv",
                   help="CheXpert valid.csv with radiologist labels.")
    p.add_argument("--out_csv", default="data/checkpoint_ranking.csv")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--context_length", type=int, default=77)
    p.add_argument("--topk", type=int, default=10, help="How many checkpoints to report as the ensemble.")
    p.add_argument("--min_gap", type=int, default=0,
                   help="Minimum step distance between ensemble members. Checkpoints saved a "
                        "few hundred steps apart are near-identical, so ensembling them adds "
                        "no diversity; in the 4-epoch run the ensemble scored worse than its "
                        "own best member. 0 disables spacing (plain top-k).")
    p.add_argument("--verify", action="store_true",
                   help="Cross-check the fast scorer against zero_shot.run_softmax_eval on the "
                        "first checkpoint, then exit.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    y_true = build_groundtruth(args.valid_csv, args.paths_csv, CXR_LABELS_5)
    print(f"Ground truth: {y_true.shape[0]} images x {y_true.shape[1]} labels")
    print("Positives per label: "
          + ", ".join(f"{l}={int(y_true[:, i].sum())}" for i, l in enumerate(CXR_LABELS_5)))

    ckpts = sorted(
        (os.path.join(args.checkpoint_dir, f) for f in os.listdir(args.checkpoint_dir)
         if f.endswith(".pt")),
        key=checkpoint_step,
    )
    if not ckpts:
        raise SystemExit(f"No .pt checkpoints in {args.checkpoint_dir}")
    print(f"Found {len(ckpts)} checkpoints")

    loader = make_loader(args.cxr_filepath, args.batch_size)

    if args.verify:
        from zero_shot import make, run_softmax_eval
        print(f"\nVerifying fast scorer against upstream on {ckpts[0]} ...")
        model = load_clip(model_path=ckpts[0], pretrained=True, context_length=args.context_length)
        ref_model, ref_loader = make(model_path=ckpts[0], cxr_filepath=args.cxr_filepath,
                                     pretrained=True, context_length=args.context_length)
        reference = run_softmax_eval(ref_model, ref_loader, CXR_LABELS_5, CXR_PAIR_TEMPLATE,
                                     context_length=args.context_length)
        fast = score_checkpoint(model.to(device), loader, CXR_LABELS_5, device, args.context_length)
        diff = float(np.abs(reference - fast).max())
        print(f"max |upstream - fast| = {diff:.3e}")
        r_auc, _ = mean_auc(y_true, reference, CXR_LABELS_5)
        f_auc, _ = mean_auc(y_true, fast, CXR_LABELS_5)
        print(f"mean AUC  upstream={r_auc:.6f}  fast={f_auc:.6f}  delta={abs(r_auc - f_auc):.3e}")
        return

    rows = []
    for i, ckpt in enumerate(ckpts, 1):
        model = load_clip(model_path=ckpt, pretrained=True, context_length=args.context_length)
        model = model.to(device)
        y_pred = score_checkpoint(model, loader, CXR_LABELS_5, device, args.context_length)
        mean, aucs = mean_auc(y_true, y_pred, CXR_LABELS_5)
        rows.append({"checkpoint": ckpt, "step": checkpoint_step(ckpt), "mean_auc": mean,
                     **{l: a for l, a in zip(CXR_LABELS_5, aucs)}})
        print(f"[{i}/{len(ckpts)}] step {checkpoint_step(ckpt):>7}  mean AUC {mean:.4f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)

    # Consumers (eval_nih.py) just take the first `topk` rows, so the spaced selection has to
    # be reflected in the file order, not only in what is printed: put the chosen members
    # first, then everything else, still ranked.
    selected = pick_spaced(df, args.topk, args.min_gap)
    rest = df[~df["checkpoint"].isin(selected["checkpoint"])]
    pd.concat([selected, rest], ignore_index=True).to_csv(args.out_csv, index=False)
    print(f"\nWrote ranking to {args.out_csv}")

    top = selected
    if args.min_gap > 0:
        print(f"(ensemble members spaced >= {args.min_gap} steps apart)")
    print(f"\nTop {args.topk} checkpoints (the ensemble):")
    print(top[["step", "mean_auc"] + CXR_LABELS_5].to_string(index=False))
    print(f"\nEnsemble member mean AUC: {top['mean_auc'].mean():.4f} "
          f"(best single {df['mean_auc'].iloc[0]:.4f})")


if __name__ == "__main__":
    main()
