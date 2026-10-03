#!/usr/bin/env bash
# CheXzero reproduction + few-shot on CheXpert test.
#   impression-only MIMIC training -> checkpoint selection -> CheXpert test zero-shot -> few-shot sweep
# Everything lives under EXP. Delete that folder to undo the run.
set -u

EXP=/mnt/My_Doc/experiments/chexpert_v1
CZ=/mnt/My_Doc/github/xray_clip/CheXzero
PY=/home/hj/anaconda3/envs/hj/bin/python
JUP=/home/hj/anaconda3/envs/hj/bin/jupyter
LOG=$EXP/run_all.log

CXR=/home/hj/cxr_cache/cxr.h5                 # 377,110, all views, already on NVMe
TXT=$CZ/data/mimic_impressions.csv            # IMPRESSION ONLY - the paper's recipe
CKPT_DIR=$EXP/checkpoints
RANKING=$EXP/checkpoint_ranking.csv
CXP=/mnt/My_Doc/dataset/CXR_dataset/CheXpert
N_SUPPORT=10000

mkdir -p "$CKPT_DIR" "$EXP/results"
cd "$CZ" || exit 1
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== CheXpert run started ==="
say "text   : mimic_impressions.csv (impression only)"
say "images : cxr.h5 (377,110, frontal + lateral)"
say "recipe : 8 epochs, batch 64, sgd lr 1e-4 momentum 0.9  (selection picks the best epoch)"

# ---------------------------------------------------------------- 1. gates
say "[1/6] gates"
$PY - <<'PYGATE' 2>&1 | tee -a "$LOG"
import h5py, pandas as pd
with h5py.File("/home/hj/cxr_cache/cxr.h5","r") as f:
    d=f["cxr"]; rows,side,dt=d.shape[0],d.shape[1],d.dtype
t=pd.read_csv("data/mimic_impressions.csv")
noimp=(t["impression"].astype(str)=="NO IMPRESSION").mean()*100
print(f"      h5 {rows:,} x {side} {dt}   text {len(t):,}   aligned={rows==len(t)}")
print(f"      NO IMPRESSION {noimp:.1f}%   nan={int(t['impression'].isna().sum())}")
assert rows==len(t), "ALIGNMENT FAILURE"
PYGATE
[ ${PIPESTATUS[0]} -ne 0 ] && { say "GATE FAILED"; exit 1; }

# ---------------------------------------------------------------- 2. train
say "[2/6] training 8 epochs (impression-only text)"
$PY -u run_train.py --cxr_filepath "$CXR" --txt_filepath "$TXT" \
    --batch_size 64 --epochs 8 --lr 1e-4 --optimizer sgd --momentum 0.9 \
    --save_interval 200 --save_dir "$CKPT_DIR" --model_name pt-imp-cxp \
    >> "$EXP/train.log" 2>&1
say "      exit=$?  checkpoints=$(ls "$CKPT_DIR/pt-imp-cxp" 2>/dev/null | wc -l)"

# ---------------------------------------------------------------- 3. select
say "[3/6] checkpoint selection on CheXpert val (200 studies, radiologist labels)"
$PY -u select_checkpoints.py --checkpoint_dir "$CKPT_DIR/pt-imp-cxp" \
    --topk 10 --min_gap 1000 --out_csv "$RANKING" >> "$EXP/select.log" 2>&1
say "      exit=$?"
[ -f "$RANKING" ] && say "      best: $(sed -n 2p "$RANKING" | cut -d, -f2,3)"

# ---------------------------------------------------------------- 4. CheXpert test h5
say "[4/6] preprocessing CheXpert test (view1 filter -> 500 frontal, one per study)"
$PY -u run_preprocess.py --dataset_type chexpert-valid \
    --chest_x_ray_path "$CXP/chexlocalize/chexlocalize/CheXpert/test" \
    --csv_out_path "$EXP/chexpert_test_paths.csv" \
    --cxr_out_path  "$EXP/chexpert_test.h5" >> "$EXP/prep.log" 2>&1
say "      exit=$?  rows=$($PY -c "import h5py;print(h5py.File('$EXP/chexpert_test.h5','r')['cxr'].shape[0])" 2>/dev/null)"

# ---------------------------------------------------------------- 5. support pool
say "[5/6] building support pool: $N_SUPPORT frontal images from CheXpert train"
$PY - <<PYSUPP >> "$EXP/prep.log" 2>&1
import sys, pandas as pd, numpy as np
sys.path.insert(0, "$CZ")
from data_process import img_to_hdf5
root = "$CXP/kaggle"
tr = pd.read_csv(f"{root}/train.csv")
tr = tr[tr["Frontal/Lateral"] == "Frontal"].copy()
tr["rel"] = tr.Path.str.extract(r"(train/patient\d+/study\d+/view\d+_\w+\.jpg)")[0]
tr = tr.dropna(subset=["rel"])
rng = np.random.default_rng(0)
sel = tr.iloc[rng.permutation(len(tr))].head($N_SUPPORT * 2)      # oversample, some files may be absent
paths = [f"{root}/{r}" for r in sel.rel]
import os
paths = [p for p in paths if os.path.exists(p)][:$N_SUPPORT]
print(f"support images on disk: {len(paths)}")
pd.DataFrame({"Path": paths}).to_csv("$EXP/chexpert_support_paths.csv", index=False)
img_to_hdf5(paths, "$EXP/chexpert_support.h5")
PYSUPP
say "      exit=$?  rows=$($PY -c "import h5py;print(h5py.File('$EXP/chexpert_support.h5','r')['cxr'].shape[0])" 2>/dev/null)"

# ---------------------------------------------------------------- 6. notebook
say "[6/6] zero-shot + few-shot sweep on CheXpert test"
cd /mnt/My_Doc/github/xray_clip || exit 1
cp 5_zeroshot_to_fewshot.ipynb "$EXP/5_chexpert.ipynb"
CX_EXP_DIR="$EXP/" CX_RANKING="$RANKING" \
$JUP nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
    "$EXP/5_chexpert.ipynb" >> "$EXP/notebook.log" 2>&1
say "      exit=$?"
say "=== run finished ==="
