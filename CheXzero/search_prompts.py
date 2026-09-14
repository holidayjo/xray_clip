"""Select zero-shot prompts on held-out NIH train/val data.

Prompt choice matters enormously for zero-shot AUC (swings of +-0.19 per label were
observed on ChestX-ray14), so it must be tuned -- but tuning on the test set would
invalidate the reported numbers. This script searches candidate prompts against a split
drawn from `train_val_list.txt`, which is disjoint from `test_list.txt`.

Efficiency: image features do not depend on the prompt. The ensemble encodes every image
once, the features are cached, and each candidate prompt is then scored as a matmul. That
makes a wide search cost roughly the same as a single evaluation.
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from eval_nih import NIH_LABELS, build_groundtruth
from select_checkpoints import CXR_PAIR_TEMPLATE, make_loader, text_weights
from zero_shot import load_clip

# Candidate prompts per label. The first entry in each list is the paper's baseline
# (the bare dataset name), so the search can only improve on it or tie.
CANDIDATES: Dict[str, List[str]] = {
    "Atelectasis":        ["Atelectasis", "atelectasis", "basilar atelectasis",
                           "a chest x-ray with atelectasis", "lung collapse"],
    "Cardiomegaly":       ["Cardiomegaly", "cardiomegaly", "enlarged heart",
                           "a chest x-ray with cardiomegaly", "enlarged cardiac silhouette"],
    "Effusion":           ["Effusion", "pleural effusion", "Pleural Effusion",
                           "a chest x-ray with pleural effusion", "fluid in the pleural space"],
    "Infiltration":       ["Infiltration", "infiltrate", "pulmonary infiltrate",
                           "a chest x-ray with pulmonary infiltrate", "airspace opacity"],
    "Mass":               ["Mass", "mass", "lung mass",
                           "a chest x-ray with a lung mass", "pulmonary mass"],
    "Nodule":             ["Nodule", "nodule", "pulmonary nodule",
                           "a chest x-ray with a nodule", "small nodular opacity"],
    "Pneumonia":          ["Pneumonia", "pneumonia", "a chest x-ray with pneumonia",
                           "infectious consolidation", "focal pneumonia"],
    "Pneumothorax":       ["Pneumothorax", "pneumothorax", "a chest x-ray with pneumothorax",
                           "collapsed lung", "air in the pleural space"],
    "Consolidation":      ["Consolidation", "consolidation", "a chest x-ray with consolidation",
                           "airspace consolidation", "dense opacity"],
    "Edema":              ["Edema", "edema", "pulmonary edema",
                           "a chest x-ray with pulmonary edema", "fluid overload"],
    "Emphysema":          ["Emphysema", "emphysema", "pulmonary emphysema",
                           "a chest x-ray with emphysema", "hyperinflated lungs",
                           "emphysematous changes"],
    "Fibrosis":           ["Fibrosis", "fibrosis", "pulmonary fibrosis",
                           "a chest x-ray with pulmonary fibrosis", "interstitial fibrosis",
                           "reticular interstitial opacities"],
    "Pleural_Thickening": ["Pleural_Thickening", "pleural thickening", "Pleural Thickening",
                           "a chest x-ray with pleural thickening", "thickened pleura"],
    "Hernia":             ["Hernia", "hernia", "hiatal hernia",
                           "a chest x-ray with a hiatal hernia", "diaphragmatic hernia"],
}


def encode_all_images(ckpts, loader, device, context_length):
    """Encode every image once per checkpoint. Returns list of (N, D) L2-normalised features."""
    feats = []
    for i, ckpt in enumerate(ckpts, 1):
        model = load_clip(model_path=ckpt, pretrained=True, context_length=context_length).to(device)
        model.eval()
        chunks = []
        with torch.no_grad():
            for batch in loader:
                f = model.encode_image(batch["img"].to(device))
                f = f / f.norm(dim=-1, keepdim=True)
                chunks.append(f.cpu())
        feats.append(torch.cat(chunks, dim=0))
        print(f"  encoded [{i}/{len(ckpts)}] {os.path.basename(ckpt)} -> {tuple(feats[-1].shape)}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return feats


def score_prompt(feats, models_text, device):
    """Ensemble probability for one candidate prompt, averaged over checkpoints."""
    total = None
    for f, (pos_w, neg_w) in zip(feats, models_text):
        pos = (f.to(device) @ pos_w).squeeze(-1)
        neg = (f.to(device) @ neg_w).squeeze(-1)
        p = (torch.exp(pos) / (torch.exp(pos) + torch.exp(neg))).cpu().numpy()
        total = p if total is None else total + p
    return total / len(feats)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ranking_csv", default="data/checkpoint_ranking.csv")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--cxr_filepath", default="data/nih_promptdev.h5")
    p.add_argument("--paths_csv", default="data/nih_promptdev_paths.csv")
    p.add_argument("--data_entry_csv",
                   default="/mnt/My_Doc/github/xray_clip/data/cxr8/Data_Entry_2017_v2020.csv")
    p.add_argument("--out_csv", default="data/prompt_search_results.csv")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--context_length", type=int, default=77)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    y_true = build_groundtruth(args.data_entry_csv, args.paths_csv, NIH_LABELS)
    print(f"Prompt-dev set: {y_true.shape[0]} images x {y_true.shape[1]} labels")
    print("Positives: " + ", ".join(f"{l}={int(y_true[:, i].sum())}"
                                    for i, l in enumerate(NIH_LABELS)))

    ckpts = pd.read_csv(args.ranking_csv)["checkpoint"].tolist()[:args.topk]
    loader = make_loader(args.cxr_filepath, args.batch_size)

    print(f"\nEncoding images with {len(ckpts)} checkpoints (once each) ...")
    feats = encode_all_images(ckpts, loader, device, args.context_length)

    # Text embeddings are cheap, but they need the models again. Rather than reloading,
    # cache the per-checkpoint text weights for every candidate string in one pass.
    all_strings = sorted({s for v in CANDIDATES.values() for s in v})
    print(f"\nEmbedding {len(all_strings)} candidate strings x {len(ckpts)} checkpoints ...")
    text_cache = {s: [] for s in all_strings}
    for ckpt in ckpts:
        model = load_clip(model_path=ckpt, pretrained=True, context_length=args.context_length).to(device)
        model.eval()
        for s in all_strings:
            pos = text_weights(model, [s], CXR_PAIR_TEMPLATE[0], device, args.context_length)
            neg = text_weights(model, [s], CXR_PAIR_TEMPLATE[1], device, args.context_length)
            text_cache[s].append((pos, neg))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    print("\nScoring candidates:")
    for li, label in enumerate(NIH_LABELS):
        for cand in CANDIDATES[label]:
            p = score_prompt(feats, text_cache[cand], device)
            auc = roc_auc_score(y_true[:, li], p)
            rows.append({"label": label, "prompt": cand, "auc": auc,
                         "is_baseline": cand == CANDIDATES[label][0]})
        best = max((r for r in rows if r["label"] == label), key=lambda r: r["auc"])
        base = next(r for r in rows if r["label"] == label and r["is_baseline"])
        flag = "  <-- improved" if best["prompt"] != base["prompt"] else ""
        print(f"  {label:20s} baseline {base['auc']:.4f} ({base['prompt']!r})")
        print(f"  {'':20s} best     {best['auc']:.4f} ({best['prompt']!r}){flag}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    best_rows = df.loc[df.groupby("label")["auc"].idxmax()]
    base_rows = df[df.is_baseline]
    print(f"\nWrote {args.out_csv}")
    print(f"\nprompt-dev mean AUC  baseline {base_rows.auc.mean():.4f} "
          f"-> tuned {best_rows.auc.mean():.4f}")
    print("\nLABEL_PROMPTS to paste into eval_nih.py:")
    print("LABEL_PROMPTS = {")
    for _, r in best_rows.iterrows():
        if r.prompt != r.label:
            print(f'    {r.label!r}: {r.prompt!r},')
    print("}")


if __name__ == "__main__":
    main()
