# Few-shot SS-Text — fresh retrain, 2026-09-28

First zero-shot → few-shot → test run. Encoder **retrained from scratch** (not `pt-imp-v2`)
so nothing here depends on an earlier model.

## Results

| | NIH-14 mean AUC | Labeled images |
|---|---|---|
| zero-shot, baseline prompts | 0.6737 | 0 |
| zero-shot, v2-tuned prompts | 0.6819 | 0 |
| SS-Text **K=16** | 0.6951 | 448 |
| SS-Text **K=32** | 0.7030 | 896 |
| SS-Text **K=64** | **0.7120** | 1,792 |
| CheXpert val, 5 tasks (best single ckpt) | **0.8650** | — |

K=64 is the highest NIH-14 in the project (Arm B 0.7105, v2 0.7035). CheXpert val 0.8650
beats v2's 0.8617 — but it is a selection maximum over 236 checkpoints, not an unbiased
estimate. CheXzero's comparable single-model figure is 0.878.

## Findings

1. **Few-shot hurts below K=8** (K=1 is −0.066). Contradicts the FCA paper, which gains at
   K=1. Independently confirmed by Shakeri et al. MICCAI 2024, whose linear probe on CXR
   drops 14 ACA points at S=1 and only ties zero-shot at S=16.
2. **Gains are frequency-dependent.** Only 4/14 labels improve at K=16, and three are the
   rare-in-MIMIC ones: Fibrosis +0.1996, Emphysema +0.1708, Hernia +0.1093. Labels the text
   encoder already handles get worse (Cardiomegaly −0.021, Effusion −0.042, Atelectasis
   −0.055). This is the mechanism behind the flat few-shot curves other CXR papers report.
3. **Support-draw variance is large.** std 0.0103 at K=16; averaging probabilities over 20
   draws gives +0.0431 vs +0.0137 for the per-seed mean.

## Caveat

`prompt_template_search.csv` was tuned on the **v2** encoder and reused unchanged. The +0.035
prompt gain collapsed to +0.008. Comparisons against 0.7035 / 0.7105 are **not like-for-like**;
re-run the template search on this encoder first. The clean within-run line is baseline prompts
throughout: 0.6737 → 0.6951 → 0.7120.

## Recipe

v2 recipe, 8 epochs, batch 64, SGD lr 1e-4 momentum 0.9, save_interval 200, on all 377,110
MIMIC pairs with `mimic_impressions_v2.csv`. Loss 4.589 → 1.828. Reproduce with `run_all.sh`.

## Files here

| File | What |
|---|---|
| `REPORT.md` | full write-up |
| *(notebook)* | `../../5_zeroshot_to_fewshot.ipynb` at repo root — single copy, outputs embedded |
| `run_all.sh` | the orchestrator that produced everything |
| `checkpoint_ranking.csv` | 236 checkpoints ranked on CheXpert val |
| `fewshot_sweep.csv` | 3,920 rows: 2 prompt sets × 7 K × 20 seeds × 14 labels |
| `comparison.csv` | against all previous runs |
| `k_sweep.png`, `per_label_delta.png` | figures |
| `train_loss.log`, `run_all.log`, `select.log`, `notebook.log` | logs |

## Not in this snapshot (too large, regenerable)

| Path | Size | What |
|---|---|---|
| `/mnt/My_Doc/experiments/fewshot_v1/checkpoints/pt-imp-fs/` | 78 GB | 236 checkpoints |
| `/mnt/My_Doc/experiments/fewshot_v1/results/feat_*.npy` | 892 MB | cached image features — re-runs skip encoding |

Delete `/mnt/My_Doc/experiments/fewshot_v1/` to reclaim 79 GB; `run_all.sh` rebuilds it in
about 2 hours.
