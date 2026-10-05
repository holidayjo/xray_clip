"""Section 3: test / support / selection splits, and labels aligned to each h5's row order.

Every h5 stores only pixels. Its *_paths.csv is the only record of which image sits in
which row, so labels are always re-ordered through that file, never assumed.
"""
import importlib.util
import os
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd

STUDY = r"(patient\d+/study\d+)"
TRAIN_IMAGE = r"(train/patient\d+/study\d+/view\d+_\w+\.jpg)"


def study_key(paths):
    return paths.str.extract(STUDY)[0]


# ---- building h5 files (skipped when they exist) ------------------------------------
def build_chexpert_test(test_img_dir, out_h5, out_paths):
    """500 frontal images, one per study, via CheXzero's own preprocessing (view1 filter)."""
    if os.path.exists(out_h5):
        return
    print("preprocessing CheXpert test ...")
    script = importlib.util.find_spec("run_preprocess").origin    # CheXzero/run_preprocess.py
    subprocess.run([sys.executable, "-u", script,
                    "--dataset_type", "chexpert-valid",
                    "--chest_x_ray_path", test_img_dir,
                    "--csv_out_path", out_paths, "--cxr_out_path", out_h5],
                   cwd=os.path.dirname(script),                   # run it from its own folder
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)


def build_chexpert_support(train_csv, out_h5, out_paths, seed, n=10000):
    """n frontal CheXpert training images, drawn by a fixed permutation. Image paths in
    train.csv are relative to the folder that holds train.csv (CheXpert/kaggle/)."""
    if os.path.exists(out_h5):
        return
    print("building CheXpert support pool ...")
    from data_process import img_to_hdf5
    tr = pd.read_csv(train_csv)
    tr = tr[tr["Frontal/Lateral"] == "Frontal"].copy()
    tr["rel"] = tr.Path.str.extract(TRAIN_IMAGE)[0]
    tr = tr.dropna(subset=["rel"])
    rng = np.random.default_rng(seed)
    image_dir = os.path.dirname(train_csv)
    cand = [f"{image_dir}/{r}" for r in tr.iloc[rng.permutation(len(tr))].rel.head(2 * n)]
    paths = [p for p in cand if os.path.exists(p)][:n]
    pd.DataFrame({"Path": paths}).to_csv(out_paths, index=False)
    img_to_hdf5(paths, out_h5)


# ---- splits --------------------------------------------------------------------------
def split_pool(n_pool, n_sel, seed):
    """Disjoint (selection, support) row indices of a pool, by a fixed permutation."""
    perm = np.random.default_rng(seed).permutation(n_pool)
    sel, supp = np.sort(perm[:n_sel]), np.sort(perm[n_sel:])
    assert not set(sel) & set(supp)
    return sel, supp


# ---- labels, in h5 row order ----------------------------------------------------------
def chexpert_test_labels(labels_csv, paths_csv, check_labels):
    """All label columns, one row per study, in the order of the test h5."""
    tl = pd.read_csv(labels_csv)
    tl["study"] = study_key(tl.Path)
    assert (tl.groupby("study")[check_labels].nunique() <= 1).all().all(), "labels differ within a study"
    tl = tl.drop_duplicates("study").set_index("study")            # image rows -> study rows
    order = study_key(pd.read_csv(paths_csv)["Path"])
    assert len(order) == len(set(order)) == 500
    return tl.loc[order]


def chexpert_support_labels(train_csv, paths_csv):
    """All label columns of the CheXpert training csv, in the order of the support h5."""
    trl = pd.read_csv(train_csv)
    trl["k"] = trl.Path.str.extract(TRAIN_IMAGE)[0]
    trl = trl.dropna(subset=["k"]).drop_duplicates("k").set_index("k")
    key = pd.read_csv(paths_csv)["Path"].str.extract(TRAIN_IMAGE)[0]
    assert key.notna().all() and key.isin(trl.index).all()
    return trl.loc[key]


# ---- checks ---------------------------------------------------------------------------
def check_aligned(items):
    """items: [(name, h5_path, n_label_rows)]. Asserts and prints h5 rows == label rows."""
    for name, h5, n_lab in items:
        with h5py.File(h5, "r") as f:
            n = f["cxr"].shape[0]
        assert n == n_lab, f"{name}: h5 {n} vs labels {n_lab}"
        print(f"  {name:<14} {n:>7,} rows  aligned")


def overlap(paths_a, paths_b):
    """(shared full paths, shared patient ids) between two *_paths.csv files.

    CheXpert files are ALL named view1_frontal.jpg, so basenames collide meaninglessly;
    compare full paths and patient ids, never basenames."""
    full = lambda p: set(pd.read_csv(p)["Path"])
    pid = lambda p: set(pd.read_csv(p)["Path"].str.extract(r"(patient\d+)")[0].dropna())
    return len(full(paths_a) & full(paths_b)), len(pid(paths_a) & pid(paths_b))
