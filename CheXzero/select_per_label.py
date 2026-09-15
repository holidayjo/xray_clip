"""Per-label checkpoint selection + prompt ensembling, tuned on held-out NIH train/val.

Motivation: the paper's procedure picks ONE ensemble by mean AUC over the five CheXpert
competition tasks, then applies it to every pathology. But per-label AUC varies by 0.31-0.51
across our 236 checkpoints, and the best single checkpoint beats the top-10 ensemble mean by
0.02-0.03 on every label -- so a globally-chosen ensemble is leaving per-label headroom unused.

Two changes, both selected on `nih_promptdev.h5` (20,000 images drawn from train_val_list.txt,
verified disjoint from test_list.txt):

1. per-label checkpoint subset  - each pathology gets the checkpoints that serve IT best
2. per-label prompt ensembling  - average probabilities over the top-N prompts rather than
                                  betting on the single best, which reduces selection variance

As in search_prompt_templates.py the prompt-dev set is halved: choose on split A, confirm on
split B. Only confirm-split numbers are unbiased. Test data is touched once, at the end, by
eval_per_label.py.
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import clip
from eval_nih import NIH_LABELS, build_groundtruth
from select_checkpoints import make_loader
from zero_shot import load_clip


def encode_images(ckpts, loader, device, ctx):
    feats = []
    for i, c in enumerate(ckpts, 1):
        m = load_clip(model_path=c, pretrained=True, context_length=ctx).to(device).eval()
        chunks = []
        with torch.no_grad():
            for b in loader:
                f = m.encode_image(b["img"].to(device))
                chunks.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
        feats.append(torch.cat(chunks))
        if i % 10 == 0 or i == len(ckpts):
            print(f"  encoded {i}/{len(ckpts)}", flush=True)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return feats


def encode_texts(ckpts, strings, device, ctx):
    cache = {s: [] for s in strings}
    for i, c in enumerate(ckpts, 1):
        m = load_clip(model_path=c, pretrained=True, context_length=ctx).to(device).eval()
        with torch.no_grad():
            toks = clip.tokenize(list(strings), context_length=ctx).to(device)
            e = m.encode_text(toks)
            e = (e / e.norm(dim=-1, keepdim=True)).float().cpu()
        for s, v in zip(strings, e):
            cache[s].append(v)
        if i % 10 == 0 or i == len(ckpts):
            print(f"  text {i}/{len(ckpts)}", flush=True)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return cache


def probs(feat, pos_v, neg_v):
    p, n = feat @ pos_v, feat @ neg_v
    return (torch.exp(p) / (torch.exp(p) + torch.exp(n))).numpy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ranking_csv", default="data/checkpoint_ranking_v2.csv")
    p.add_argument("--pool", type=int, default=60,
                   help="Consider the best N checkpoints from the global ranking. Scoring all "
                        "236 on 20k images is far slower for little extra headroom.")
    p.add_argument("--topk", type=int, default=10, help="Checkpoints per label.")
    p.add_argument("--prompt_topn", type=int, default=3, help="Prompts averaged per label.")
    p.add_argument("--template_csv", default="data/prompt_template_search.csv")
    p.add_argument("--cxr_filepath", default="data/nih_promptdev.h5")
    p.add_argument("--paths_csv", default="data/nih_promptdev_paths.csv")
    p.add_argument("--data_entry_csv",
                   default="/mnt/My_Doc/github/xray_clip/data/cxr8/Data_Entry_2017_v2020.csv")
    p.add_argument("--out_csv", default="data/per_label_selection.csv")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--context_length", type=int, default=77)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    y = build_groundtruth(a.data_entry_csv, a.paths_csv, list(NIH_LABELS))
    n = len(y)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(n)
    A, B = perm[: n // 2], perm[n // 2:]
    print(f"prompt-dev {n:,} -> select A={len(A):,}, confirm B={len(B):,}")

    ckpts = pd.read_csv(a.ranking_csv)["checkpoint"].tolist()[: a.pool]
    print(f"candidate pool: {len(ckpts)} checkpoints")

    ts = pd.read_csv(a.template_csv)
    strings = sorted(set(ts.prompt_pos) | set(ts.prompt_neg))

    loader = make_loader(a.cxr_filepath, a.batch_size)
    print(f"\nEncoding images x {len(ckpts)} checkpoints...")
    feats = encode_images(ckpts, loader, dev, a.context_length)
    print(f"Encoding {len(strings)} prompt strings...")
    tcache = encode_texts(ckpts, strings, dev, a.context_length)

    rows = []
    for li, label in enumerate(NIH_LABELS):
        cand = ts[ts.label == label].sort_values("auc_select", ascending=False).head(a.prompt_topn)
        # per-checkpoint probability, already prompt-ensembled
        per_ck = []
        for k in range(len(ckpts)):
            pr = np.mean([probs(feats[k], tcache[r.prompt_pos][k], tcache[r.prompt_neg][k])
                          for r in cand.itertuples()], axis=0)
            per_ck.append(pr)
        per_ck = np.stack(per_ck)                                  # (n_ckpt, n_img)

        scores = [roc_auc_score(y[A, li], per_ck[k][A]) for k in range(len(ckpts))]
        order = np.argsort(scores)[::-1][: a.topk]                 # best checkpoints for THIS label

        ens_sel = per_ck[order].mean(0)
        base_sel = per_ck[: a.topk].mean(0)                        # global top-k, the current method
        rows.append({
            "label": label,
            "auc_global_confirm": roc_auc_score(y[B, li], base_sel[B]),
            "auc_perlabel_confirm": roc_auc_score(y[B, li], ens_sel[B]),
            "auc_perlabel_select": roc_auc_score(y[A, li], ens_sel[A]),
            "checkpoints": "|".join(ckpts[i] for i in order),
            "prompts_pos": "|".join(cand.prompt_pos),
            "prompts_neg": "|".join(cand.prompt_neg),
        })
        r = rows[-1]
        print(f"  {label:<20} global {r['auc_global_confirm']:.4f} -> per-label "
              f"{r['auc_perlabel_confirm']:.4f}  ({r['auc_perlabel_confirm']-r['auc_global_confirm']:+.4f})",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out_csv, index=False)
    print(f"\nWrote {a.out_csv}")
    print(f"CONFIRM-split mean: global {df.auc_global_confirm.mean():.4f} -> "
          f"per-label {df.auc_perlabel_confirm.mean():.4f} "
          f"({df.auc_perlabel_confirm.mean()-df.auc_global_confirm.mean():+.4f})")


if __name__ == "__main__":
    main()
