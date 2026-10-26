"""Generate every table, figure and number in the paper from the result files, so nothing is typed by hand.

    /home/hj/anaconda3/envs/hj/bin/python latex/scripts/make_tables.py

Inputs
  runs/freq_v1/results/: asserted_nih.csv, asserted_nih_summary.txt, direct_label_counts_exact_vs_stem.csv,
                         zero_shot_nih.csv, chexpert_13.csv, ranking_nih.csv, ranking_chexpert.csv
  claude_s/:             negation_CheXpert.csv (labeler counts, CheXpert classes), negation_summary.txt (agreement),
                         rules_summary.csv / .txt and rules_per_class_ci_*.csv (few-shot image weight)
Outputs: latex/tables/*.tex, latex/tables/numbers.tex (\newcommand macros), latex/figures/relationships.png
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "CheXzero")]
from utils import config                      # noqa: E402
from utils.freq import laws                   # noqa: E402

CFG = config.load_experiment(os.path.join(REPO, "config/freq_v1.yaml"))
R = os.path.join(CFG["experiment"]["dir"], "results")
S = os.path.join(REPO, "claude_s")
OUT = os.path.join(REPO, "latex", "tables")
FIG = os.path.join(REPO, "latex", "figures")
os.makedirs(OUT, exist_ok=True)
NIH = CFG["files"]["nih"]["classes"]
CXP = CFG["cxp_labels_all"]
K_REF = CFG["eval"]["k_ref"]
nice = lambda l: l.replace("_", " ")
fmt = lambda n: f"{n:,}".replace(",", "{,}")
macros = {}


def p_fmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write(name, text):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text)


# ---- per-class data -----------------------------------------------------------------------------
nih = pd.read_csv(os.path.join(R, "asserted_nih.csv"), index_col=0).loc[NIH]          # asserted_%, word_%, zero_shot, K1..K64
exact = pd.read_csv(os.path.join(R, "direct_label_counts_exact_vs_stem.csv"), index_col="label")["% (exact)"]
nih["exact_%"] = exact
nih["pos"] = pd.read_csv(os.path.join(R, "zero_shot_nih.csv"), index_col=0)["test positives"]
nih["delta"] = nih[f"K{K_REF}"] - nih.zero_shot
c13 = pd.read_csv(os.path.join(R, "chexpert_13.csv"))
cxp = c13[c13.K == K_REF].set_index("label").loc[CXP]
cxp["asserted_%"] = pd.read_csv(os.path.join(S, "negation_CheXpert.csv"), index_col=0)["asserted_%"]
cxp["delta"] = cxp.few_shot - cxp.zero_shot

# ---- Table: NIH classes -------------------------------------------------------------------------
t = nih.sort_values("asserted_%", ascending=False)
rows = "\n".join(f"{nice(l)} & {r['asserted_%']:.2f} & {r['word_%']:.2f} & {r['exact_%']:.2f} & {int(r.pos):,} & "
                 f"{r.zero_shot:.3f} & {r.delta:+.3f} \\\\" for l, r in t.iterrows())
write("tab_classes.tex", rf"""\begin{{table}}[!t]
\centering
\footnotesize
\caption{{ChestX-ray14 classes: concept frequency $f$ in the MIMIC-CXR training impressions (\% of 377{{,}}110
image--text pairs; asserted = labelled positive by the CheXpert labeler; word = class name, stemmed or exact),
zero-shot AUC on the test set, and few-shot (SS-Text) gain at $K={K_REF}$ (mean of 20 draws).}}
\label{{tab:classes}}
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
Class & \multicolumn{{3}}{{c}}{{$f$ (\%)}} & Test & Zero-shot & $\Delta$AUC \\
 & assert. & word & exact & pos. & AUC & $K{{=}}{K_REF}$ \\
\midrule
{rows}
\midrule
Mean & & & & & {nih.zero_shot.mean():.3f} & {nih.delta.mean():+.3f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""")

# ---- Table: CheXpert classes ---------------------------------------------------------------------
t = cxp.sort_values("asserted_%", ascending=False)
rows = "\n".join(f"{l} & {r['asserted_%']:.2f} & {int(r.n_pos)} & {r.zero_shot:.3f} & {r.delta:+.3f} \\\\"
                 for l, r in t.iterrows())
write("tab_cxp_classes.tex", rf"""\begin{{table}}[!t]
\centering
\footnotesize
\caption{{CheXpert classes: asserted frequency $f$ in the training impressions, positives in the 500-study
test set, zero-shot AUC and few-shot (SS-Text) gain at $K={K_REF}$ (mean of 20 draws).}}
\label{{tab:cxpclasses}}
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lrrrr}}
\toprule
Class & $f$ & Test & Zero-shot & $\Delta$AUC \\
 & (\%) & pos. & AUC & $K{{=}}{K_REF}$ \\
\midrule
{rows}
\midrule
Mean & & & {cxp.zero_shot.mean():.3f} & {cxp.delta.mean():+.3f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""")

# ---- Table: both relationships, both datasets ----------------------------------------------------
def rel(df, col, k_cols):
    x = np.log10(df[col]); a = laws.fit(x, df.zero_shot); b = laws.fit(x, df.delta)
    pr, pp = laws.partial_r(x, df.delta, df.zero_shot)
    trend = {K: stats.pearsonr(x, df[c])[0] for K, c in k_cols}
    return dict(a=a, b=b, pr=pr, pp=pp, trend=trend)

nih_k = [(0, "zero_shot")] + [(K, f"K{K}") for K in CFG["eval"]["k_sweep"]]
cxp_k = [(0, "zero_shot"), (K_REF, "few_shot")]
REL = {("ChestX-ray14", "asserted"): rel(nih, "asserted_%", nih_k),
       ("ChestX-ray14", "word (stemmed)"): rel(nih, "word_%", nih_k),
       ("ChestX-ray14", "word (exact)"): rel(nih, "exact_%", nih_k),
       ("CheXpert", "asserted"): rel(cxp, "asserted_%", cxp_k)}
rp = lambda r, p: f"{r:+.2f} ({p_fmt(p)})"
rows = "\n".join(f"{ds} ({14 if ds == 'ChestX-ray14' else 12}) & {f} & {rp(v['a']['r'], v['a']['p'])} & {rp(v['a']['rho'], v['a']['ps'])} & "
                 f"{rp(v['b']['r'], v['b']['p'])} & {rp(v['pr'], v['pp'])} & {v['trend'][0]:+.2f} $\\to$ {v['trend'][K_REF]:+.2f} \\\\"
                 for (ds, f), v in REL.items())
write("tab_rel.tex", rf"""\begin{{table*}}[!t]
\centering
\footnotesize
\caption{{Relationships (1) and (2) on both test sets (one point per class). Relationship (1): zero-shot AUC vs.\
$\log_{{10}} f$. Relationship (2): few-shot gain $\Delta$AUC (SS-Text, $K={K_REF}$) vs.\ $\log_{{10}} f$; the partial
correlation removes a linear effect of the zero-shot AUC from both. Last column: Pearson $r$ of $\log_{{10}} f$ with
AUC at $K=0$ (zero-shot) and at $K={K_REF}$. Entries are $r$ or $\rho$ with two-sided $p$ in parentheses.}}
\label{{tab:rel}}
\begin{{tabular}}{{llrrrrr}}
\toprule
Test set ($n$) & $f$ & \multicolumn{{2}}{{c}}{{Rel.~(1)}} & \multicolumn{{2}}{{c}}{{Rel.~(2), $K={K_REF}$}} & $r(\log f,\mathrm{{AUC}})$ \\
 & & Pearson $r$ & Spearman $\rho$ & Pearson $r$ & partial $r$ & $K{{=}}0 \to {K_REF}$ \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")
A, C = REL[("ChestX-ray14", "asserted")], REL[("CheXpert", "asserted")]
W = REL[("ChestX-ray14", "word (stemmed)")]
macros.update({
    "NihROne": f"{A['a']['r']:+.2f}", "NihPOne": p_fmt(A['a']['p']), "NihRhoOne": f"{A['a']['rho']:+.2f}", "NihPsOne": p_fmt(A['a']['ps']),
    "NihSlopeOne": f"{A['a']['slope']:+.3f}",
    "CxpROne": f"{C['a']['r']:+.2f}", "CxpPOne": p_fmt(C['a']['p']), "CxpRhoOne": f"{C['a']['rho']:+.2f}", "CxpPsOne": p_fmt(C['a']['ps']),
    "CxpSlopeOne": f"{C['a']['slope']:+.3f}",
    "NihWordROne": f"{W['a']['r']:+.2f}", "NihWordPOne": p_fmt(W['a']['p']),
    "NihRTwo": f"{A['b']['r']:+.2f}", "NihPTwo": p_fmt(A['b']['p']), "NihPartial": f"{A['pr']:+.2f}", "NihPartialP": p_fmt(A['pp']),
    "CxpRTwo": f"{C['b']['r']:+.2f}", "CxpPTwo": p_fmt(C['b']['p']), "CxpPartial": f"{C['pr']:+.2f}", "CxpPartialP": p_fmt(C['pp']),
    "NihWordRTwo": f"{W['b']['r']:+.2f}", "NihWordPTwo": p_fmt(W['b']['p']),
    "NihWordPartial": f"{W['pr']:+.2f}", "NihWordPartialP": p_fmt(W['pp']),
    "NihTrendZero": f"{A['trend'][0]:+.2f}", "NihTrendRef": f"{A['trend'][K_REF]:+.2f}",
    "CxpTrendZero": f"{C['trend'][0]:+.2f}", "CxpTrendRef": f"{C['trend'][K_REF]:+.2f}",
    "NihTrendAll": ", ".join(f"{A['trend'][K]:+.2f}" for K, _ in nih_k),
    "NihImproved": str(int((nih.delta > 0).sum())), "CxpImproved": str(int((cxp.delta > 0).sum())),
    "ZSMeanNih": f"{nih.zero_shot.mean():.3f}", "ZSMeanCxp": f"{cxp.zero_shot.mean():.3f}",
})

# ---- labeler agreement ----------------------------------------------------------------------------
ag_nih = [float(x) for x in re.findall(r"label agreement\s+([\d.]+)%", open(os.path.join(R, "asserted_nih_summary.txt")).read())]
ag_cxp = [float(x) for x in re.findall(r"agreement\s+([\d.]+)%", open(os.path.join(S, "negation_summary.txt")).read())]
for df, tag in ((nih, "Nih"), (cxp, "Cxp")):          # does the starting AUC predict the gain?
    r_, p_ = stats.pearsonr(df.zero_shot, df.delta); rho_, ps_ = stats.spearmanr(df.zero_shot, df.delta)
    macros.update({f"{tag}ZsGainR": f"{r_:+.2f}", f"{tag}ZsGainP": p_fmt(p_), f"{tag}ZsGainRho": f"{rho_:+.2f}", f"{tag}ZsGainPs": p_fmt(ps_)})
macros.update({"PneuWord": f"{nih.loc['Pneumonia', 'word_%']:.1f}", "PneuAsserted": f"{nih.loc['Pneumonia', 'asserted_%']:.1f}"})
macros.update({"AgreeCxpMin": f"{min(ag_cxp):.2f}"})      # ag_nih is printed to 1 decimal only

# ---- Table: few-shot image weight ------------------------------------------------------------------
rs = pd.read_csv(os.path.join(S, "rules_summary.csv"))
COLS = [("SS-Text (FCA, a=1)", "SS-Text"), ("global a", "global $\\alpha$"), ("frequency rule", "frequency rule")]
lines = []
for K in [0] + CFG["eval"]["k_sweep"]:
    cells = []
    for ds in ("NIH", "CheXpert"):
        r = rs[(rs.dataset == ds) & (rs.K == max(K, 1))].iloc[0]
        cells += [f"{r['zero-shot']:.3f}"] * 3 if K == 0 else [f"{r[c]:.3f}" for c, _ in COLS]
    lines.append(f"{'0 (zero-shot)' if K == 0 else K} & " + " & ".join(cells) + " \\\\")
hdr = " & ".join(h for _, h in COLS)
write("tab_weight.tex", rf"""\begin{{table*}}[!t]
\centering
\footnotesize
\caption{{Few-shot adaptation with three image weights: mean AUC over classes (each class: mean of 20 draws of the
support images). SS-Text: $\alpha=1$ as in~\cite{{silvarodriguez2025fca}}. Global $\alpha$ and the frequency rule
were chosen on the ChestX-ray14 selection set and applied to CheXpert without retuning.}}
\label{{tab:weight}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
 & \multicolumn{{3}}{{c}}{{ChestX-ray14 test (14 classes)}} & \multicolumn{{3}}{{c}}{{CheXpert test (12 classes)}} \\
$K$ & {hdr} & {hdr} \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")
for ds, tag in (("NIH", "Nih"), ("CheXpert", "Cxp")):
    r = rs[(rs.dataset == ds) & (rs.K == K_REF)].iloc[0]
    macros.update({f"{tag}FcaRef": f"{r['SS-Text (FCA, a=1)']:.3f}", f"{tag}GlobalRef": f"{r['global a']:.3f}",
                   f"{tag}RuleRef": f"{r['frequency rule']:.3f}", f"{tag}ZsRef": f"{r['zero-shot']:.3f}",
                   f"{tag}FcaBelow": str(int(r["R1 below zs"])), f"{tag}GlobalBelow": str(int(r["R2 below zs"])),
                   f"{tag}RuleBelow": str(int(r["R3 below zs"]))})
fw = pd.read_csv(os.path.join(S, "fw_summary.csv")).drop_duplicates("K").set_index("K")   # tuned a_global, a_rule, b_rule
macros.update({"AlphaGlobalRef": f"{10 ** fw.loc[K_REF, 'a_global']:.3f}", "RuleB": f"{fw.loc[K_REF, 'b_rule']:+.1f}",
               "RuleBNeg": str(int((fw.b_rule < 0).sum())), "NK": str(len(fw))})
pcl = pd.read_csv(os.path.join(S, "fw_per_class_K64.csv"))
al = pcl[pcl.dataset == "NIH"]["alpha_rule"]
macros.update({"AlphaRuleMin": f"{al.min():.3f}", "AlphaRuleMax": f"{al.max():.3f}"})

# bootstrap (K = 64): differences in mean AUC and per-class significance
txt = open(os.path.join(S, "rules_summary.txt")).read()
blocks = dict(zip(["NIH", "CheXpert"], re.split(r"=== (?:NIH|CheXpert), K=64: paired bootstrap", txt)[1:]))
def diff(block, a, b):
    m = re.search(re.escape(a) + r"\s+-\s+" + re.escape(b) + r"\s+([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", block)
    return [float(x) for x in m.groups()]
def sig(block, a):
    m = re.search(re.escape(a) + r"\s+significantly better: (\d+)\s+significantly worse: (\d+)", block)
    return [int(x) for x in m.groups()]
brow = []
for ds, tag in (("NIH", "Nih"), ("CheXpert", "Cxp")):
    b = blocks[ds]
    for a_, b_, key in [("global a", "SS-Text (FCA, a=1)", "GlobalVsFca"), ("frequency rule", "global a", "RuleVsGlobal")]:
        m, lo, hi = diff(b, a_, b_)
        macros[f"{tag}{key}"] = f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    for a_, key in [("SS-Text (FCA, a=1)", "Fca"), ("global a", "Global"), ("frequency rule", "Rule")]:
        bt, wo = sig(b, a_); macros[f"{tag}{key}SigBetter"], macros[f"{tag}{key}SigWorse"] = str(bt), str(wo)
    nb = re.search(r"\((\d+) resamples", txt[txt.index(f"=== {ds}, K=64"):]).group(1)
    macros[f"{tag}NBoot"] = nb

# ---- Table: datasets and splits --------------------------------------------------------------------
from eval_nih import build_groundtruth as nih_gt          # noqa: E402
from select_checkpoints import build_groundtruth as cxp_val_gt   # noqa: E402
from utils.freq import data as fdata                      # noqa: E402
cz = CFG["files"]["chexzero_data"]; NIHF = CFG["files"]["nih"]; CX = CFG["files"]["chexpert"]
imp_rows = pd.read_csv(cz["impressions"])
mimic_paths = pd.read_csv(os.path.join(REPO, "CheXzero/data/cxr_paths.csv"))["Path"]
assert len(mimic_paths) == len(imp_rows)
mimic_pat = mimic_paths.str.extract(r"/(p\d{8})/")[0].nunique()
entry = pd.read_csv(NIHF["labels"]).set_index("Image Index")["Patient ID"]
pool_names = pd.read_csv(cz["nih_pool_paths"])["Path"].map(os.path.basename)
test_names = pd.read_csv(cz["nih_test_paths"])["Path"].map(os.path.basename)
SEL, SUPP = fdata.split_pool(len(pool_names), CFG["eval"]["n_sel"], CFG["eval"]["seed"])
y_pool = nih_gt(NIHF["labels"], cz["nih_pool_paths"], NIH); y_test = nih_gt(NIHF["labels"], cz["nih_test_paths"], NIH)
def nih_row(names, y):
    return len(names), entry.reindex(names).nunique(), 100 * (y.sum(1) > 0).mean()
pat = lambda paths: paths.str.extract(r"(patient\d+)")[0].nunique()
cv_paths = pd.read_csv(cz["cxp_val_paths"])["Path"]
cv_y = cxp_val_gt(CX["valid_csv"], cz["cxp_val_paths"], CFG["cxp_labels_5"])
ct_paths_csv = os.path.join(CFG["experiment"]["dir"], "chexpert_test_paths.csv")
cs_paths_csv = os.path.join(CFG["experiment"]["dir"], "chexpert_support_paths.csv")
ct_y = fdata.chexpert_test_labels(CX["test_labels"], ct_paths_csv, CFG["cxp_labels_5"])[CXP].to_numpy(float)
cs_y = fdata.chexpert_support_labels(CX["train_csv"], cs_paths_csv)[CXP].to_numpy(float)
ct_paths, cs_paths = pd.read_csv(ct_paths_csv)["Path"], pd.read_csv(cs_paths_csv)["Path"]
anypos = lambda y: 100 * (np.nan_to_num(y) == 1).any(1).mean()
sets = [("MIMIC-CXR", "training", fmt(len(imp_rows)), fmt(mimic_pat), "impression text", "--"),
        *[(f"ChestX-ray14 {n}", role, fmt(a), fmt(b), "14 labels", f"{c:.1f}") for n, role, (a, b, c) in [
            ("selection", "checkpoint choice", nih_row(pool_names.iloc[SEL], y_pool[SEL])),
            ("support", "few-shot examples", nih_row(pool_names.iloc[SUPP], y_pool[SUPP])),
            ("test", "evaluation", nih_row(test_names, y_test))]],
        ("CheXpert validation", "checkpoint choice", fmt(len(cv_paths)), fmt(pat(cv_paths)), "5 labels", f"{anypos(cv_y):.1f}"),
        ("CheXpert train subset", "few-shot examples", fmt(len(cs_paths)), fmt(pat(cs_paths)), "12 labels", f"{anypos(cs_y):.1f}"),
        ("CheXpert test", "evaluation", fmt(len(ct_paths)), fmt(pat(ct_paths)), "12 labels", f"{anypos(ct_y):.1f}")]
rows = "\n".join(" & ".join(s) + " \\\\" for s in sets)
write("tab_data.tex", rf"""\begin{{table*}}[!t]
\centering
\footnotesize
\caption{{Datasets and splits. MIMIC-CXR provides image--text pairs (one row per image; a study's impression is
repeated for each of its images); no labels are used for training. ChestX-ray14 selection and support are drawn from
the official train/validation split and test is the official test split. CheXpert: one frontal image per study for
validation and test; the support images are a fixed random subset of the training set. Last column: images
positive for at least one of the labels used in that set.}}
\label{{tab:data}}
\begin{{tabular}}{{llrrlr}}
\toprule
Set & Role & Images & Patients & Labels used & $\geq$1 positive (\%) \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")
macros.update({"MimicCxrPatients": fmt(mimic_pat), "NTest": fmt(len(test_names)),
               "NTestPatients": fmt(entry.reindex(test_names).nunique()),
               "NCxpVal": fmt(len(cv_paths)), "NCxpTest": fmt(len(ct_paths)), "NCxpSupp": fmt(len(cs_paths))})

# ---- figure: both relationships, both datasets (asserted f) ----------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for j, (name, df) in enumerate([("ChestX-ray14", nih), ("CheXpert", cxp)]):
    x = np.log10(df["asserted_%"])
    for i, (y, ylab) in enumerate([(df.zero_shot, "zero-shot AUC"), (df.delta, f"$\\Delta$AUC, K={K_REF}")]):
        a = ax[i, j]; r = laws.fit(x, y); xs = np.linspace(x.min() - .1, x.max() + .1, 50)
        a.scatter(x, y, s=40, zorder=3); a.plot(xs, r["slope"] * xs + r["intercept"], lw=1.2)
        for l in df.index:
            a.annotate(nice(l), (x[l], y[l]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        if i == 1:
            a.axhline(0, color="k", lw=.7)
        a.set_title(f"{name}: r = {r['r']:+.2f} (p = {p_fmt(r['p'])})", fontsize=10)
        a.set_xlabel("log10 f (asserted, % of training pairs)"); a.set_ylabel(ylab); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "relationships.png"), dpi=200)

# ---- numbers used in the text ------------------------------------------------------------------------
rank, rank_c = pd.read_csv(os.path.join(R, "ranking_nih.csv")), pd.read_csv(os.path.join(R, "ranking_chexpert.csv"))
best, best_c = rank.iloc[0], rank_c.iloc[0]
macros.update({
    "MimicCxrPairs": fmt(len(imp_rows)),
    "MimicCxrStudies": fmt(imp_rows["filename"].str.extract(r"s(\d+)")[0].nunique()),
    "MimicCxrUniqueTexts": fmt(imp_rows["impression"].nunique(dropna=False)),   # the 42 empty impressions count as one text
    "MimicCxrNoImpression": fmt((imp_rows["impression"] == "NO IMPRESSION").sum()),
    "NCkpt": str(len(rank)), "BestStep": fmt(int(best.step)), "BestEpoch": f"{best.step / CFG['steps_per_epoch']:.2f}",
    "SelAUC": f"{best.mean_auc:.3f}", "CxpBestStep": fmt(int(best_c.step)),
    "CxpBestEpoch": f"{best_c.step / CFG['steps_per_epoch']:.2f}", "CxpSelAUC": f"{best_c.mean_auc:.3f}",
    "KRef": str(K_REF),
})
write("numbers.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
for old in ("tab_rel1.tex", "tab_rel2.tex", "tab_fewshot.tex", "tab_asserted.tex"):
    if os.path.exists(os.path.join(OUT, old)):
        os.remove(os.path.join(OUT, old))
print("written:", sorted(os.listdir(OUT)))
