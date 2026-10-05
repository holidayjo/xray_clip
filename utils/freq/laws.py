"""Sections 6-12: bootstrap CIs, Law 1 / Law 2 fits, the crossing point and the policies.

Law 1:  zero-shot AUC   = a * log10(f) + b
Law 2:  few-shot delta  = c * log10(f) + d        (delta = few-shot AUC - zero-shot AUC)
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def boot_mean_auc(y, probs, labels, n_boot=1000, seed=0):
    """(mean, 2.5%, 97.5%) of the label-averaged AUC over bootstrap resamples of images.
    Labels with a single class in a resample are skipped for that resample."""
    rng, n, out = np.random.default_rng(seed), len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = [roc_auc_score(y[idx, i], probs[l][idx]) for i, l in enumerate(labels)
             if y[idx, i].min() != y[idx, i].max()]
        if a:
            out.append(np.mean(a))
    d = np.array(out)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def fit(x, y):
    """Pearson, Spearman and the least-squares line of y on x."""
    r, p = stats.pearsonr(x, y)
    rho, ps = stats.spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    return dict(r=r, p=p, rho=rho, ps=ps, slope=slope, intercept=intercept)


def partial_r(x, y, z):
    """Pearson r between x and y after removing a linear fit on z from both."""
    res = lambda v: v - np.poly1d(np.polyfit(z, v, 1))(z)
    return stats.pearsonr(res(x), res(y))


def median_split(t, freq_col="f_%"):
    """(rare, common): rows below / at-or-above the median mention rate."""
    med = t[freq_col].median()
    return t[t[freq_col] < med], t[t[freq_col] >= med]


# ---- section 7: Law 1 --------------------------------------------------------------------
def law1(freq, zs_auc, n_pos, labels):
    """Law 1 on one dataset: fit, rare/common split, and the test-prevalence confound.
    freq needs columns f_% and log_f (mentions.frequency)."""
    t = freq.loc[labels, ["f_%", "log_f"]].copy()
    t["auc"] = pd.Series(zs_auc)
    t["n_pos"] = n_pos
    t = t.dropna(subset=["log_f"])              # labels without a frequency drop out
    out = dict(table=t, **fit(t.log_f, t.auc))
    rare, common = median_split(t)
    out["rare"], out["common"] = rare, common
    out["split_p"] = stats.mannwhitneyu(common.auc, rare.auc, alternative="greater")[1]
    log_n = np.log10(t.n_pos)
    out["confound_r"], out["confound_p"] = stats.pearsonr(log_n, t.auc)
    out["partial_r"], out["partial_p"] = partial_r(t.log_f, t.auc, log_n)
    return out


# ---- section 9: Law 2 --------------------------------------------------------------------
def delta_table(fs, name, K, d, freq):
    """Per label: zero-shot AUC, mean few-shot AUC at K, delta, mention rate."""
    m = fs[(fs.dataset == name) & (fs.K == K)].groupby("label")["auc"].mean()
    t = pd.DataFrame({"zero_shot": pd.Series(d["zs_auc"]), "few_shot": m})
    t["delta"] = t.few_shot - t.zero_shot
    t["f_%"] = freq.loc[t.index, "f_%"]
    t["log_f"] = freq.loc[t.index, "log_f"]
    return t.loc[d["labels"]].dropna(subset=["log_f"])     # labels without a frequency drop out


def law2(fs, tasks, k_sweep, freq, min_labels=5):
    """{(dataset, K): fit of delta on log_f, plus its table}, for every dataset and K."""
    out = {}
    for name, d in tasks.items():
        for K in k_sweep:
            t = delta_table(fs, name, K, d, freq)
            if len(t) >= min_labels:
                out[(name, K)] = dict(table=t, **fit(t.log_f, t.delta))
    return out


def sign_test(t):
    """Do rare concepts gain more than common ones? One-sided Mann-Whitney on delta."""
    rare, common = median_split(t)
    try:
        p = stats.mannwhitneyu(rare.delta, common.delta, alternative="greater")[1]
    except ValueError:
        p = float("nan")
    return dict(rare_helped=f"{int((rare.delta > 0).sum())}/{len(rare)}",
                common_helped=f"{int((common.delta > 0).sum())}/{len(common)}",
                rare_mean=rare.delta.mean(), common_mean=common.delta.mean(), p=p)


# ---- sections 10-11: crossing point and policies -----------------------------------------
def crossing(law2_fit):
    """Mention rate (%) at which the Law 2 line crosses zero."""
    return 10 ** (-law2_fit["intercept"] / law2_fit["slope"])


def rule_vs_outcome(t, cross):
    """Rule 'adapt if f < cross' against what actually helped."""
    return t.assign(
        predicted=np.where(t["f_%"] < cross, "adapt", "keep zero-shot"),
        actual=np.where(t.delta > 0, "adapt", "keep zero-shot"))


def policies(law2_fits, tasks, ks, cross):
    """Mean AUC under zero-shot, uniform adaptation, the frequency rule, and the oracle."""
    rows = []
    for name, d in tasks.items():
        zs = pd.Series(d["zs_auc"]).mean()
        for K in ks:
            t = law2_fits[(name, K)]["table"]
            rows.append({"dataset": name, "K": K, "zero_shot": zs,
                         "uniform": t.few_shot.mean(),
                         "rule": np.where(t["f_%"] < cross, t.few_shot, t.zero_shot).mean(),
                         "oracle": np.maximum(t.few_shot, t.zero_shot).mean()})
    p = pd.DataFrame(rows)
    p["rule_gain"] = p.rule - p.zero_shot
    return p
