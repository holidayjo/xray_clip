"""Sections 4-6: checkpoint ranking, feature encoding and zero-shot scoring.

Zero-shot score for a label:  p = sigmoid(v.t_pos - v.t_neg), which equals the 2-way
softmax over ("{label}", "no {label}") but is numerically stable.
"""
import hashlib
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import clip                                    # CheXzero/clip.py
from select_checkpoints import make_loader     # CheXzero
from zero_shot import load_clip                # CheXzero

CXR_PAIR = ("{}", "no {}")                     # CheXzero's prompt frame, untuned


def prompt_strings(labels, pmap):
    """Every positive and negative prompt needed for these labels."""
    return sorted({t.format(pmap[l]) for l in labels for t in CXR_PAIR})


def _load(ckpt, ctx, device):
    return load_clip(model_path=ckpt, pretrained=True, context_length=ctx).to(device).eval()


def _free(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def text_vecs(model, strings, ctx, device):
    with torch.no_grad():
        e = model.encode_text(clip.tokenize(list(strings), context_length=ctx).to(device))
        return (e / e.norm(dim=-1, keepdim=True)).float()


def checkpoint_step(path):
    """Training step from 'checkpoint_32200.pt'; the final 'checkpoint.pt' is step -1."""
    m = re.search(r"_(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


# ---- section 4: checkpoint selection -------------------------------------------------
def score_checkpoint(ckpt, loader, labels, pmap, ctx, device, rows=None):
    """(n_images, n_labels) zero-shot probabilities for one checkpoint."""
    m = _load(ckpt, ctx, device)
    pos = text_vecs(m, [CXR_PAIR[0].format(pmap[l]) for l in labels], ctx, device)
    neg = text_vecs(m, [CXR_PAIR[1].format(pmap[l]) for l in labels], ctx, device)
    chunks = []
    with torch.no_grad():
        for b in loader:
            f = m.encode_image(b["img"].to(device))
            f = (f / f.norm(dim=-1, keepdim=True)).float()
            chunks.append(torch.sigmoid(f @ pos.T - f @ neg.T).cpu())
    del m
    _free(device)
    p = torch.cat(chunks).numpy()
    return p if rows is None else p[rows]


def rank_checkpoints(ckpt_paths, h5, labels, pmap, y, cache, batch, ctx, device, rows=None):
    """Mean AUC of every checkpoint on a selection split, best first. Cached at `cache`."""
    tag = os.path.basename(cache)
    if os.path.exists(cache):
        r = pd.read_csv(cache)
        print(f"{tag}: cache hit ({len(r)} checkpoints)")
        return r
    loader = make_loader(h5, batch)
    out = []
    for i, c in enumerate(ckpt_paths, 1):
        p = score_checkpoint(c, loader, labels, pmap, ctx, device, rows)
        aucs = [roc_auc_score(y[:, j], p[:, j]) for j in range(len(labels))]
        out.append({"checkpoint": c, "step": checkpoint_step(c), "mean_auc": float(np.mean(aucs))})
        if i % 40 == 0 or i == len(ckpt_paths):
            print(f"  {tag} [{i}/{len(ckpt_paths)}] best so far "
                  f"{max(o['mean_auc'] for o in out):.4f}", flush=True)
    r = pd.DataFrame(out).sort_values("mean_auc", ascending=False).reset_index(drop=True)
    r.to_csv(cache, index=False)
    return r


def best_per_epoch(ranking, steps_per_epoch, n_epochs):
    """Best checkpoint and mean AUC within each epoch."""
    r = ranking.assign(epoch=np.ceil(ranking.step.clip(lower=0) / steps_per_epoch)
                       .astype(int).clip(1, n_epochs))
    best = r.loc[r.groupby("epoch").mean_auc.idxmax(), ["epoch", "step", "mean_auc"]]
    return best.set_index("epoch").assign(epoch_mean=r.groupby("epoch").mean_auc.mean())


# ---- section 5: encode once ------------------------------------------------------------
def encode_images(h5, ckpt, cache_dir, tag, batch, ctx, device, rows=None):
    """L2-normalised image features, cached per (checkpoint, h5, rows-or-not)."""
    key = hashlib.md5((ckpt + h5 + str(rows is not None)).encode()).hexdigest()[:10]
    cache = os.path.join(cache_dir, f"feat_{tag}_{key}.npy")
    if os.path.exists(cache):
        a = np.load(cache)
        print(f"  {tag}: cache hit {a.shape}")
        return torch.from_numpy(a)
    m = _load(ckpt, ctx, device)
    chunks = []
    with torch.no_grad():
        for b in make_loader(h5, batch):
            f = m.encode_image(b["img"].to(device))
            chunks.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
    del m
    _free(device)
    a = torch.cat(chunks).numpy()
    if rows is not None:
        a = a[rows]
    np.save(cache, a)
    print(f"  {tag}: encoded {a.shape}")
    return torch.from_numpy(a)


def encode_texts(ckpt, strings, ctx, device):
    """{prompt string: L2-normalised text vector}"""
    m = _load(ckpt, ctx, device)
    v = text_vecs(m, strings, ctx, device).cpu()
    del m
    _free(device)
    return {s: v[i] for i, s in enumerate(strings)}


def make_task(ckpt, labels, pmap, y_test, y_supp, F_test, F_supp, ctx, device):
    """Everything zero-shot and few-shot need for one dataset, in one dict."""
    assert len(F_test) == len(y_test) and len(F_supp) == len(y_supp)
    return dict(ckpt=ckpt, labels=labels, pmap=pmap, y_test=y_test, y_supp=y_supp,
                F_test=F_test, F_supp=F_supp,
                T=encode_texts(ckpt, prompt_strings(labels, pmap), ctx, device))


# ---- section 6: zero-shot ----------------------------------------------------------------
def pair_prob(F, w_pos, w_neg):
    """sigmoid(a-b) equals the 2-way softmax, but is numerically stable."""
    return torch.sigmoid(F @ w_pos - F @ w_neg).numpy()


def text_pair(d, label):
    return (d["T"][CXR_PAIR[0].format(d["pmap"][label])],
            d["T"][CXR_PAIR[1].format(d["pmap"][label])])


def zero_shot(d):
    """Adds zs_prob, zs_auc (per label) and ZS (mean AUC) to the task dict."""
    d["zs_prob"] = {l: pair_prob(d["F_test"], *text_pair(d, l)) for l in d["labels"]}
    d["zs_auc"] = {l: roc_auc_score(d["y_test"][:, i], d["zs_prob"][l])
                   for i, l in enumerate(d["labels"])}
    d["ZS"] = float(np.mean(list(d["zs_auc"].values())))
    return d
