"""Section 2: concept frequency f, how often each pathology is asserted in the training text.

Two sources: the CheXpert labeler output (labeler_table, citable) and our hand regex
(mention_table). frequency() puts every definition side by side and picks the primary one."""
import hashlib
import os
import re

import numpy as np
import pandas as pd
import yaml


def load_vocabulary(path):
    """(synonyms, negation_regex) from config/concept_synonyms.yaml."""
    with open(path) as f:
        v = yaml.safe_load(f)
    neg = re.compile(r"\b(" + "|".join(v["negation"]["cues"]) + r")\b[^.;]{0,%d}$"
                     % v["negation"]["window"])
    return v["synonyms"], neg


def mention_stats(reports, patterns, neg):
    """Counts for one concept.

    reports  : lower-cased pd.Series of report texts
    patterns : list of regex patterns for the concept
    neg      : negation regex; a report counts as negated if ANY mention has a cue just before it
    """
    pat = re.compile("|".join(patterns))
    hit = reports.str.contains(pat, regex=True)
    n = int(hit.sum())
    n_neg = sum(any(neg.search(s[:m.start()]) for m in pat.finditer(s)) for s in reports[hit])
    return {"n_mentions": n, "negated_%": 100 * n_neg / max(n, 1),
            "affirmative_n": n - n_neg, "affirmative_%": 100 * (n - n_neg) / len(reports)}


def mention_table(reports_csv, vocab_path, cache_dir):
    """
    One row per concept, plus log_f = log10(affirmative %).
    Cached as mention_rates_<hash>.csv, 
    where <hash> comes from the vocabulary file,
    so editing a synonym forces a recount. 
    Returns (table, n_reports, cache_path).
    """
    
    key     = hashlib.md5(open(vocab_path, "rb").read()).hexdigest()[:8]
    # print("mention_table: vocab hash", key, "from", vocab_path)
    cache   = os.path.join(cache_dir, f"mention_rates_{key}.csv")
    reports = pd.read_csv(reports_csv)["impression"].fillna("").str.lower()
    # print("mention_table: counting mentions in", len(reports), "reports ...")
    # print("mention_table: caching to", cache)
    # print("report sample:\n", reports)
    if os.path.exists(cache):
        table = pd.read_csv(cache, index_col=0)
    else:
        synonyms, neg = load_vocabulary(vocab_path)
        table         = pd.DataFrame({l: mention_stats(reports, p, neg) for l, p in synonyms.items()}).T
        table.to_csv(cache)
    table["log_f"] = np.log10(table["affirmative_%"])
    
    return table, len(reports), cache


# ---- f from the CheXpert labeler (MIMIC's mimic-cxr-2.0.0-chexpert.csv.gz) -------------------
# NIH label name -> CheXpert labeler column, only where it is the same finding. Mass and Nodule
# are merged into CheXpert's "Lung Lesion" and cannot be separated, so they are not mapped.
TO_CHEXPERT = {"Effusion": "Pleural Effusion"}
LABELER_COLS = ["labeler_pos_%", "labeler_pos_unc_%", "labeler_any_%"]


def labeler_table(impressions_csv, labels_csv, labels, negbio_csv=None):
    """
    Share (%) of training rows whose study the CheXpert labeler marks, per label.

    Unit = training rows (image-text pairs), as the model saw them.
    A row whose training text is NO IMPRESSION counts as NOT exposed whatever its label: 
    MIMIC labelled those studies from the findings section (Johnson et al. 2019, p.3), 
    which the model never saw. 
    Labels the labeler does not cover get NaN.
    
        labeler_pos_%      label 1                        (Irvin et al. 2019, aggregation, p.3)
        labeler_pos_unc_%  label 1 or -1                  (U-Ones mapping, p.4)
        labeler_any_%      label 1, 0 or -1, i.e. named   (mention extraction, p.3)
        negbio_pos_%       label 1 in MIMIC's NegBio file: same mention phrases, NegBio's own
                           negation/uncertainty rules (Johnson et al. 2019, p.3). Only if negbio_csv.
    """
    imp   = pd.read_csv(impressions_csv)
    seen  = (imp["impression"].astype(str) != "NO IMPRESSION").to_numpy()
    # print(f"labeler_table: {seen.sum()} of {len(imp)} training rows have an impression")
    study = imp["filename"].str.extract(r"s(\d+)")[0].astype(int)
    # print("study =", study)
    # print(f"labeler_table: {len(study.unique())} unique studies in {len(imp)} training rows")
    
    lab   = pd.read_csv(labels_csv).set_index("study_id").reindex(study)
    nb    = pd.read_csv(negbio_csv).set_index("study_id").reindex(study) if negbio_csv else None
    n     = len(imp)
    rows  = {}
    for label in labels:
        col = TO_CHEXPERT.get(label, label)
        if col not in lab.columns:
            rows[label] = dict.fromkeys(LABELER_COLS + (["negbio_pos_%"] if nb is not None else []), np.nan)
            continue
        v = lab[col].to_numpy()
        pct = lambda m: 100 * (m & seen).sum() / n
        rows[label] = {"labeler_pos_%": pct(v == 1),
                       "labeler_pos_unc_%": pct((v == 1) | (v == -1)),
                       "labeler_any_%": pct(~np.isnan(v))}
        if nb is not None:
            rows[label]["negbio_pos_%"] = pct(nb[col].to_numpy() == 1)
    return pd.DataFrame(rows).T


def frequency(regex, labeler, primary, udandarao=None):
    """Every definition of f side by side; f_% and log_f are the `primary` one.
    log_f is NaN where f is missing or zero, so those labels drop out of the fits.
    udandarao: optional {label: udandarao_%} (labels mapped from prompt strings by the caller)."""
    t = labeler.join(regex["affirmative_%"].rename("regex_%"), how="outer")
    if udandarao is not None:
        t["udandarao_%"] = pd.Series(udandarao)
    t["f_%"] = t[primary]
    t["log_f"] = np.log10(t["f_%"].where(t["f_%"] > 0))
    return t


# ---- f exactly as Udandarao et al. (2024) count it (see udandarao_count.py) ------------------
def udandarao_table(impressions_csv, concepts, out_csv, python):
    """{concept: udandarao_%}. Runs udandarao_count.py in the spaCy-3.7.2 environment `python`
    unless out_csv already holds every concept. `concepts` are the class names as prompted."""
    import subprocess, sys
    if os.path.exists(out_csv) and set(concepts) <= set(pd.read_csv(out_csv)["concept"]):
        return pd.read_csv(out_csv).set_index("concept")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "udandarao_count.py")
    print("counting as Udandarao et al. (spaCy en_core_web_lg over every training row) ...", flush=True)
    subprocess.run([os.path.expanduser(python), script, "--impressions", impressions_csv,
                    "--out", out_csv, "--concepts", *concepts], check=True, stdout=sys.stdout)
    return pd.read_csv(out_csv).set_index("concept")
