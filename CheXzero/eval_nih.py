"""Zero-shot evaluation of a CheXzero checkpoint ensemble on ChestX-ray14.

Takes the ranking produced by select_checkpoints.py, ensembles the top-k checkpoints by
averaging their softmax probability outputs (the combination rule used in Tiu et al.),
and reports per-pathology AUC on the NIH test split.

Prompt wording: CheXzero's template pair is ("{}", "no {}"), so the label string is
literally the prompt. Two ChestX-ray14 label names are not usable as-is -- "Effusion"
is ambiguous and "Pleural_Thickening" carries an underscore that tokenises poorly -- so
LABEL_PROMPTS maps them to their radiological phrasing. Everything else is unchanged.
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from select_checkpoints import make_loader, score_checkpoint
from zero_shot import load_clip

# The 14 ChestX-ray14 pathologies, in the canonical order used by the benchmark.
NIH_LABELS: List[str] = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

# Dataset label -> prompt text.
#
# BASELINE (paper-faithful): only the two names that are not usable as-is.
# The paper's Methods use bare '<label>' / 'no <label>' prompts, so anything richer is a
# deviation and must be justified by tuning on held-out data -- never on the test set.
# Override this dict from prompt_search_results.csv to evaluate tuned prompts.
LABEL_PROMPTS = {
    "Effusion": "Pleural Effusion",
    "Pleural_Thickening": "Pleural Thickening",
}



def build_groundtruth(data_entry_csv: str, paths_csv: str, labels: List[str]) -> np.ndarray:
    """Multi-hot label matrix whose row order matches the h5 row order."""
    order = pd.read_csv(paths_csv)["Path"].map(os.path.basename)

    entry = pd.read_csv(data_entry_csv).set_index("Image Index")
    missing = set(order) - set(entry.index)
    if missing:
        raise ValueError(f"{len(missing)} images absent from {data_entry_csv}, "
                         f"e.g. {sorted(missing)[:3]}")

    findings = entry.loc[order, "Finding Labels"].str.split("|")
    y = np.zeros((len(order), len(labels)), dtype=float)
    index = {l: i for i, l in enumerate(labels)}
    for row, items in enumerate(findings):
        for item in items:
            if item in index:          # silently skips "No Finding"
                y[row, index[item]] = 1.0
    return y


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ranking_csv", default="data/checkpoint_ranking.csv",
                   help="Output of select_checkpoints.py, sorted best-first.")
    p.add_argument("--topk", type=int, default=10, help="Ensemble size.")
    p.add_argument("--cxr_filepath", default="data/nih_test.h5")
    p.add_argument("--paths_csv", default="data/nih_test_paths.csv")
    p.add_argument("--data_entry_csv",
                   default="/mnt/My_Doc/github/xray_clip/data/cxr8/Data_Entry_2017_v2020.csv")
    p.add_argument("--out_csv", default="data/nih_zeroshot_results.csv")
    p.add_argument("--labels", default=None,
                   help="Comma-separated subset of NIH_LABELS to evaluate, e.g. to match a "
                        "benchmark that reports only some pathologies. Default: all 14.")
    p.add_argument("--prompts_csv", default=None,
                   help="Optional prompt_search_results.csv from search_prompts.py. When given, "
                        "the best-scoring prompt per label (by held-out AUC) overrides "
                        "LABEL_PROMPTS. Those prompts must have been selected on train/val "
                        "data, never on the test set.")
    p.add_argument("--pred_npy", default="data/nih_zeroshot_preds.npy",
                   help="Where to save the ensembled probabilities.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--context_length", type=int, default=77)
    p.add_argument("--n_bootstrap", type=int, default=1000,
                   help="Bootstrap replicates for confidence intervals; 0 to skip.")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def bootstrap_ci(y_true, y_pred, labels, n: int, seed: int):
    """Percentile CIs over resampled images."""
    rng = np.random.default_rng(seed)
    n_rows = y_true.shape[0]
    draws = np.empty((n, len(labels)))
    for b in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        yt, yp = y_true[idx], y_pred[idx]
        for j in range(len(labels)):
            col = yt[:, j]
            # a resample can be single-class, leaving AUC undefined
            draws[b, j] = roc_auc_score(col, yp[:, j]) if 0 < col.sum() < len(col) else np.nan
    lo = np.nanpercentile(draws, 2.5, axis=0)
    hi = np.nanpercentile(draws, 97.5, axis=0)
    return lo, hi


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels = [l.strip() for l in args.labels.split(",")] if args.labels else list(NIH_LABELS)
    unknown = [l for l in labels if l not in NIH_LABELS]
    if unknown:
        raise SystemExit(f"unknown labels: {unknown}")
    if args.labels:
        print(f"Evaluating {len(labels)} of {len(NIH_LABELS)} labels: {labels}")

    if args.prompts_csv:
        sr = pd.read_csv(args.prompts_csv)
        best = sr.loc[sr.groupby("label")["auc"].idxmax()].set_index("label")["prompt"]
        prompts = [best.get(l, LABEL_PROMPTS.get(l, l)) for l in labels]
        print(f"Prompts from {args.prompts_csv} (tuned on held-out train/val)")
    else:
        prompts = [LABEL_PROMPTS.get(l, l) for l in labels]
        print("Prompts: paper baseline (bare label names)")
    print("Prompt mapping: " + ", ".join(f"{l!r}->{p!r}" for l, p in zip(labels, prompts)
                                         if l != p))

    y_true = build_groundtruth(args.data_entry_csv, args.paths_csv, labels)
    print(f"Ground truth: {y_true.shape[0]} images x {y_true.shape[1]} labels")

    ranking = pd.read_csv(args.ranking_csv)
    ckpts = ranking["checkpoint"].tolist()[:args.topk]
    print(f"Ensembling {len(ckpts)} checkpoints from {args.ranking_csv}")

    loader = make_loader(args.cxr_filepath, args.batch_size)

    # Average the per-model probability outputs, as the paper does.
    total = None
    for i, ckpt in enumerate(ckpts, 1):
        model = load_clip(model_path=ckpt, pretrained=True, context_length=args.context_length).to(device)
        y_pred = score_checkpoint(model, loader, prompts, device, args.context_length)
        total = y_pred if total is None else total + y_pred
        mean_auc = np.mean([roc_auc_score(y_true[:, j], y_pred[:, j]) for j in range(len(labels))])
        print(f"  [{i}/{len(ckpts)}] {os.path.basename(ckpt)}  mean AUC {mean_auc:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    y_pred = total / len(ckpts)
    np.save(args.pred_npy, y_pred)

    aucs = [roc_auc_score(y_true[:, j], y_pred[:, j]) for j in range(len(labels))]
    out = pd.DataFrame({
        "label": labels,
        "prompt": prompts,
        "n_positive": y_true.sum(axis=0).astype(int),
        "auc": aucs,
    })

    if args.n_bootstrap:
        print(f"Bootstrapping ({args.n_bootstrap} replicates) ...")
        lo, hi = bootstrap_ci(y_true, y_pred, labels, args.n_bootstrap, args.seed)
        out["ci_lower"], out["ci_upper"] = lo, hi

    out.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}\n")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nEnsemble mean AUC over {len(labels)} pathologies: {np.mean(aucs):.4f}")


if __name__ == "__main__":
    main()
