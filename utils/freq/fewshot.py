"""Section 8: SS-Text few-shot adaptation (Silva-Rodriguez et al., Full Conformal
Adaptation of Medical VLMs, Eq. 10 with lambda = 1/(N tau)):

    w_pos = sum_{i in pos} v_i + t_pos        w_neg = sum_{i in neg} v_i + t_neg

A SUM, not a mean: with K images per class the text prototype is about 1/(K+1) of the
classifier. Nothing in the encoder is trained.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .clip_eval import pair_prob, text_pair


def ss_text(d, K, rng):
    """{label: test probabilities} after adding K positive and K negative support images."""
    probs = {}
    for i, l in enumerate(d["labels"]):
        t_pos, t_neg = text_pair(d, l)
        c = d["y_supp"][:, i]
        # CheXpert convention: blank means "not mentioned", which the official evaluation and
        # test_labels.csv both treat as negative. -1 (uncertain) is excluded from both pools.
        # NIH has neither blanks nor -1, so this reduces to the obvious rule there.
        pool_p = np.flatnonzero(c == 1)
        pool_n = np.flatnonzero((c == 0) | np.isnan(c))
        k = min(K, len(pool_p), len(pool_n))
        if k == 0:
            probs[l] = pair_prob(d["F_test"], t_pos, t_neg)
            continue
        ip = rng.choice(pool_p, k, replace=False)
        inn = rng.choice(pool_n, k, replace=False)
        probs[l] = pair_prob(d["F_test"],
                             d["F_supp"][ip].sum(0) + t_pos,
                             d["F_supp"][inn].sum(0) + t_neg)
    return probs


def k0_matches_zero_shot(d):
    """Max |p| difference between ss_text(K=0) and zero-shot (NB: K=0 takes the k == 0
    branch, so this only checks the text path, not the support-sum path)."""
    p0 = ss_text(d, 0, np.random.default_rng(0))
    return max(abs(p0[l] - d["zs_prob"][l]).max() for l in d["labels"])


def sweep(tasks, k_sweep, seeds, cache):
    """Per-label AUC for every (dataset, K, seed). Cached at `cache`."""
    if os.path.exists(cache):
        fs = pd.read_csv(cache)
        print(f"cache hit: {len(fs):,} rows")
        return fs
    rows = []
    for name, d in tasks.items():
        for K in k_sweep:
            for s in range(seeds):
                pr = ss_text(d, K, np.random.default_rng(s))
                for i, l in enumerate(d["labels"]):
                    rows.append({"dataset": name, "K": K, "seed": s, "label": l,
                                 "auc": roc_auc_score(d["y_test"][:, i], pr[l])})
            print(f"  {name:<9} K={K:<3} done", flush=True)
    fs = pd.DataFrame(rows)
    fs.to_csv(cache, index=False)
    print(f"\n{len(fs):,} rows -> {cache}")
    return fs


def summarize(fs):
    """Mean and std (over seeds) of the label-averaged AUC, per dataset and K."""
    return (fs.groupby(["dataset", "K", "seed"])["auc"].mean().reset_index()
              .groupby(["dataset", "K"])["auc"].agg(["mean", "std"]).reset_index())


def per_label(d, ks, seeds):
    """Mean few-shot AUC per label (over seeds), for each K in ks: {K: {label: auc}}."""
    out = {}
    for K in ks:
        acc = {l: [] for l in d["labels"]}
        for s in range(seeds):
            pr = ss_text(d, K, np.random.default_rng(s))
            for i, l in enumerate(d["labels"]):
                acc[l].append(roc_auc_score(d["y_test"][:, i], pr[l]))
        out[K] = {l: float(np.mean(v)) for l, v in acc.items()}
    return out
