#!/usr/bin/env bash
# Full pipeline, from scratch:
#   MIMIC contrastive training -> checkpoint selection -> zero-shot -> few-shot -> test
# Everything it writes lives under EXP. Delete that one folder to undo the whole run.
set -u

EXP=/mnt/My_Doc/experiments/fewshot_v1
CZ=/mnt/My_Doc/github/xray_clip/CheXzero
PY=/home/hj/anaconda3/envs/hj/bin/python
JUP=/home/hj/anaconda3/envs/hj/bin/jupyter
LOG=$EXP/run_all.log

CXR=/home/hj/cxr_cache/cxr.h5                 # NVMe copy, already staged
TXT=$CZ/data/mimic_impressions_v2.csv         # impression + FINDINGS fallback (v2 recipe)
CKPT_DIR=$EXP/checkpoints
RANKING=$EXP/checkpoint_ranking.csv

mkdir -p "$CKPT_DIR" "$EXP/results"
cd "$CZ" || exit 1

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== run_all started ==="
say "experiment dir : $EXP"
say "recipe         : v2 (8 epochs, batch 64, sgd lr 1e-4 momentum 0.9, save_interval 200)"

# ---------------------------------------------------------------- 1. gates
say "[1/4] gates"
$PY - <<'PYGATE' 2>&1 | tee -a "$LOG"
import h5py, pandas as pd
h5, txt = "/home/hj/cxr_cache/cxr.h5", "data/mimic_impressions_v2.csv"
with h5py.File(h5, "r") as f:
    d = f["cxr"]; rows, side, dt = d.shape[0], d.shape[1], d.dtype
t = pd.read_csv(txt)
col = "impression" if "impression" in t.columns else t.columns[-1]
noimp = (t[col].astype(str) == "NO IMPRESSION").mean() * 100
print(f"      h5 {rows:,} x {side} {dt}   text {len(t):,}   aligned={rows == len(t)}")
print(f"      NO IMPRESSION {noimp:.1f}%   nan={int(t[col].isna().sum())}")
assert rows == len(t), "ALIGNMENT FAILURE - stop"
PYGATE
[ ${PIPESTATUS[0]} -ne 0 ] && { say "GATE FAILED - aborting"; exit 1; }

# ---------------------------------------------------------------- 2. train
say "[2/4] training 8 epochs from scratch"
$PY -u run_train.py \
    --cxr_filepath "$CXR" \
    --txt_filepath "$TXT" \
    --batch_size 64 --epochs 8 --lr 1e-4 \
    --optimizer sgd --momentum 0.9 \
    --save_interval 200 \
    --save_dir "$CKPT_DIR" --model_name pt-imp-fs \
    >> "$EXP/train.log" 2>&1
say "      train exit=$?  checkpoints=$(ls "$CKPT_DIR/pt-imp-fs" 2>/dev/null | wc -l)"

# ---------------------------------------------------------------- 3. select
say "[3/4] checkpoint selection on CheXpert validation"
$PY -u select_checkpoints.py \
    --checkpoint_dir "$CKPT_DIR/pt-imp-fs" \
    --topk 10 --min_gap 1000 \
    --out_csv "$RANKING" \
    >> "$EXP/select.log" 2>&1
say "      select exit=$?"
[ -f "$RANKING" ] && say "      best: $(sed -n 2p "$RANKING" | cut -d, -f1,3)"

# ---------------------------------------------------------------- 4. notebook
say "[4/4] zero-shot -> few-shot -> test"
cd /mnt/My_Doc/github/xray_clip || exit 1
cp 5_zeroshot_to_fewshot.ipynb "$EXP/5_zeroshot_to_fewshot.ipynb"
FS_EXP_DIR="$EXP/" FS_RANKING="$RANKING" FS_RUN_NAME="fs-retrain" \
$JUP nbconvert --to notebook --execute --inplace --allow-errors \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    "$EXP/5_zeroshot_to_fewshot.ipynb" \
    >> "$EXP/notebook.log" 2>&1
say "      notebook exit=$?"

say "=== run_all finished ==="
