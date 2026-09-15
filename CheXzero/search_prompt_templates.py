"""Search over prompt TEMPLATE PAIRS as well as label strings.

search_prompts.py tunes only the label string inside CheXzero's fixed ("{}", "no {}") frame.
That frame produces malformed negatives for any multi-word prompt -- the tuned Nodule prompt
becomes the literal string "no a chest x-ray with a nodule". This script searches the frame
too, so both polarities can be phrased in radiology register:

    ("findings consistent with {}", "no evidence of {}")

Method notes
------------
* Image features do not depend on the prompt, so each checkpoint encodes the images once and
  every (template, label-string) pair is then a matmul. A ~700-combination search costs about
  the same as one plain evaluation.
* The search space is large enough to overfit the selection split, so the prompt-dev set is
  halved: winners are chosen on split A and *confirmed* on split B, which is never used for
  selection. Only the confirmed numbers should be quoted.
* Everything here runs on images drawn from train_val_list.txt, disjoint from test_list.txt.
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import clip
from eval_nih import NIH_LABELS, build_groundtruth
from search_prompts import CANDIDATES
from select_checkpoints import make_loader
from zero_shot import load_clip

# (positive, negative) frames. The first is CheXzero's, so the search can only tie or improve.
TEMPLATE_PAIRS: List[Tuple[str, str]] = [
    ("{}", "no {}"),                                        # paper baseline
    ("a chest x-ray with {}", "a chest x-ray with no {}"),
    ("findings consistent with {}", "no evidence of {}"),
    ("there is {}", "there is no {}"),
    ("{} is present", "{} is absent"),
]


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
        print(f"  [{i}/{len(ckpts)}] {os.path.basename(c)} -> {tuple(feats[-1].shape)}", flush=True)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return feats


def encode_texts(ckpts, strings, device, ctx, batch=128):
    """{string: [ (D,) tensor per checkpoint ]} -- batched, unlike search_prompts.py."""
    cache = {s: [] for s in strings}
    for i, c in enumerate(ckpts, 1):
        m = load_clip(model_path=c, pretrained=True, context_length=ctx).to(device).eval()
        with torch.no_grad():
            for k in range(0, len(strings), batch):
                chunk = strings[k:k + batch]
                tok = clip.tokenize(chunk, context_length=ctx).to(device)
                e = m.encode_text(tok)
                e = (e / e.norm(dim=-1, keepdim=True)).float().cpu()
                for s, v in zip(chunk, e):
                    cache[s].append(v)
        print(f"  [{i}/{len(ckpts)}] text done", flush=True)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return cache


def ensemble_prob(feats, pos_vecs, neg_vecs):
    """Mean over checkpoints of softmax(pos, neg). Matches eval_nih's combination rule."""
    tot = None
    for f, p, n in zip(feats, pos_vecs, neg_vecs):
        pos, neg = f @ p, f @ n
        pr = (torch.exp(pos) / (torch.exp(pos) + torch.exp(neg))).numpy()
        tot = pr if tot is None else tot + pr
    return tot / len(feats)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ranking_csv", default="data/checkpoint_ranking_v2.csv")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--cxr_filepath", default="data/nih_promptdev.h5")
    p.add_argument("--paths_csv", default="data/nih_promptdev_paths.csv")
    p.add_argument("--data_entry_csv",
                   default="/mnt/My_Doc/github/xray_clip/data/cxr8/Data_Entry_2017_v2020.csv")
    p.add_argument("--out_csv", default="data/prompt_template_search.csv")
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
    A, B = perm[: n // 2], perm[n // 2:]       # A selects, B confirms
    print(f"prompt-dev {n:,} images -> select on A={len(A):,}, confirm on B={len(B):,}")

    ckpts = pd.read_csv(a.ranking_csv)["checkpoint"].tolist()[: a.topk]
    loader = make_loader(a.cxr_filepath, a.batch_size)

    print(f"\nEncoding images ({len(ckpts)} checkpoints, once each)...")
    feats = encode_images(ckpts, loader, dev, a.context_length)

    strings = sorted({t.format(s) for v in CANDIDATES.values() for s in v
                      for pair in TEMPLATE_PAIRS for t in pair})
    print(f"\nEncoding {len(strings)} prompt strings x {len(ckpts)} checkpoints...")
    tcache = encode_texts(ckpts, strings, dev, a.context_length, )

    rows = []
    print("\nScoring...")
    for li, label in enumerate(NIH_LABELS):
        yA, yB = y[A, li], y[B, li]
        for pos_t, neg_t in TEMPLATE_PAIRS:
            for cand in CANDIDATES[label]:
                pr = ensemble_prob(feats, tcache[pos_t.format(cand)], tcache[neg_t.format(cand)])
                rows.append({
                    "label": label, "template_pos": pos_t, "template_neg": neg_t,
                    "label_string": cand,
                    "prompt_pos": pos_t.format(cand), "prompt_neg": neg_t.format(cand),
                    "auc_select": roc_auc_score(yA, pr[A]),
                    "auc_confirm": roc_auc_score(yB, pr[B]),
                    "is_baseline": (pos_t, neg_t) == TEMPLATE_PAIRS[0] and cand == CANDIDATES[label][0],
                })
        print(f"  {label:20s} done ({len(TEMPLATE_PAIRS)*len(CANDIDATES[label])} combos)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out_csv, index=False)
    print(f"\nWrote {a.out_csv}  ({len(df)} combinations)")

    best = df.loc[df.groupby("label")["auc_select"].idxmax()].set_index("label")
    base = df[df.is_baseline].set_index("label")

    print("\n" + "=" * 104)
    print(f"{'label':<20}{'baseline':>9}{'tuned(A)':>10}{'CONFIRM(B)':>12}{'d_vs_base':>11}  winning positive prompt")
    print("-" * 104)
    for l in NIH_LABELS:
        d = best.loc[l, "auc_confirm"] - base.loc[l, "auc_confirm"]
        print(f"{l:<20}{base.loc[l,'auc_confirm']:>9.4f}{best.loc[l,'auc_select']:>10.4f}"
              f"{best.loc[l,'auc_confirm']:>12.4f}{d:>+11.4f}  {best.loc[l,'prompt_pos']!r}")
    print("-" * 104)
    print(f"{'MEAN':<20}{base.auc_confirm.mean():>9.4f}{best.auc_select.mean():>10.4f}"
          f"{best.auc_confirm.mean():>12.4f}{best.auc_confirm.mean()-base.auc_confirm.mean():>+11.4f}")
    print("=" * 104)
    print("Quote the CONFIRM column: auc_select is the split the winner was chosen on and is "
          "optimistically biased.")

    print("\nTemplate pairs chosen:")
    print(best.groupby(["template_pos", "template_neg"]).size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
