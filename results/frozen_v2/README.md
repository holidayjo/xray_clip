# Frozen state — v2 (8-epoch model), 2026-09-15

Restore point taken before switching the training text to impression+findings.

## Best results at this point

| Metric | Value |
|---|---|
| NIH full test, 14 labels (template prompts) | **0.7035** |
| Benchmark split (13,871), 9 labels | **0.6468** |
| Benchmark ViT (ASL), supervised, same split | 0.8119 → gap **-0.165** |
| CheXpert val, 5 tasks (best checkpoint) | 0.8617 |

## Models on disk (gitignored, not in this snapshot)

| Dir | What |
|---|---|
| `CheXzero/checkpoints/pt-imp-v1/` | 4 epochs, impression-only text. NIH 0.6738 |
| `CheXzero/checkpoints/pt-imp-v2/` | 8 epochs, impression+FINDINGS-fallback text. **Best.** |
| `CheXzero/checkpoints/pt-imp/` | empty — next run writes here |

Rankings in this folder point at the archived dirs, so evaluation can be re-run
without retraining.

## Training data

- `CheXzero/data/cxr.h5` and `/home/hj/cxr_cache/cxr.h5` — 377,110 x 320 x 320 uint8, whole image
- `CheXzero/data/mimic_impressions_v2.csv` — impression, FINDINGS as fallback, 4.4% "NO IMPRESSION"

## Findings established here

1. **8 epochs > 4**, but only on the selection domain. CheXpert val 0.845 -> 0.862 while
   NIH *fell* 0.6738 -> 0.6685. Checkpoint steps spread across epochs 4-8, so 8 is enough.
2. **Prompts matter more than training.** Template-pair search +0.035; doubling epochs -0.005.
   CheXzero's own ("{}", "no {}") frame yields malformed negatives
   ("no a chest x-ray with a nodule"); fixing it gave Pneumothorax +0.051.
3. **Lung cropping does not help** (-0.0001 on the benchmark split). Hernia fell -0.035
   (subdiaphragmatic, cropped away) and Cardiomegaly rose +0.032, confirming the crop
   was applied correctly. The resolution hypothesis is dead.
4. **Emphysema / Fibrosis / Nodule are prevalence-limited, not resolution-limited.**
   Measured three ways: those terms appear in 2.6% / 7.5% / 3.9% of reports even when
   the ENTIRE report text is searched. The words are simply not written.

## Next experiment (what this freeze protects against)

Impression-first concatenation with FINDINGS -> `mimic_impressions_v3.csv`, retrain 8 epochs.
Expected: gains on Pneumothorax (mentions 12% -> 61%) and Effusion (24% -> 75%);
little or nothing on Emphysema / Fibrosis / Nodule.

## To roll back

```bash
git checkout frozen-v2
# evaluation needs no retraining:
cd CheXzero && python -u eval_nih.py \
    --ranking_csv data/checkpoint_ranking_v2.csv --topk 10 \
    --template_csv data/prompt_template_search.csv \
    --out_csv /tmp/restore_check.csv --n_bootstrap 0
# expect mean AUC 0.7035
```
