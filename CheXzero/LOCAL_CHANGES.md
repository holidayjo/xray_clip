# Local changes to CheXzero

This folder is a copy of the official code, <https://github.com/rajpurkarlab/CheXzero> (`main`),
added to this repo in commit `8fa0c70` (2026-09-13). It holds three kinds of files:
1. **upstream files**, unchanged or patched;
2. **our own scripts**, added next to them;
3. **generated data**, which git ignores.

Compared against upstream `main` on 2026-10-05.

## 1. Upstream files

| file | status | what changed |
|---|---|---|
| `clip.py`, `model.py`, `eval.py`, `metrics.py`, `simple_tokenizer.py`, `preprocess_padchest.py`, `requirements.txt`, `README.md` | **unchanged** | none |
| `zero_shot.py` | patched | `.float()` on images, because our `cxr.h5` is stored as uint8 |
| `train.py` | patched | h5 file opened lazily per DataLoader worker; `num_workers` 0 → 8 (plus `pin_memory`, `persistent_workers`, `prefetch_factor`); `.float()` on uint8 images |
| `run_train.py` | whitespace only | alignment of `=` signs |
| `run_preprocess.py` | extended | new `--dataset_type` values `chexpert-valid` and `nih-test`, each writing a `*_paths.csv` that records h5 row order; new `--image_list` argument |
| `data_process.py` | patched | (a) `img_to_hdf5`: process pool, uint8 storage (4× smaller, same values), `Image.LANCZOS` instead of `ANTIALIAS`, which Pillow 10 removed. **(b) `write_report_csv`: two blocks added that are not in upstream:** a findings-section fallback when no impression exists (commit `501f3e6`), and appending findings after the impression (commit `1830b90`). See the warning below. |

### ⚠ Warning: `write_report_csv` no longer matches the paper's recipe

Upstream code and the CheXzero paper use the **impression only**; a report with no
impression becomes the literal string `NO IMPRESSION`. The two added blocks change that.
`data/mimic_impressions.csv` (2026-09-13) was built **before** these blocks existed, so the
trained models are unaffected. Rerunning preprocessing now would produce different
training text, though. Decide whether to remove the blocks or put them behind a flag.

## 2. Our own scripts (not in upstream)

| file | purpose |
|---|---|
| `__init__.py` | makes the folder importable |
| `eval_nih.py` | zero-shot evaluation on ChestX-ray14; `NIH_LABELS`, `LABEL_PROMPTS`, `build_groundtruth` |
| `select_checkpoints.py` | ranks checkpoints by validation AUC; `make_loader` |
| `select_per_label.py` | per-label checkpoint and prompt selection on `nih_promptdev` |
| `search_prompts.py`, `search_prompt_templates.py` | prompt and template search on `nih_promptdev` |
| `preprocess_cropped.py` | lung-cropped preprocessing (crop ablation) |

## 3. Generated data (ignored by git)

| folder | size (2026-10-05) | contents |
|---|---|---|
| `data/` | about 70 GB | preprocessed h5 files, path csvs, impressions csvs, predictions, rankings |
| `checkpoints/` | about 362 GB | `pt-imp`, `pt-imp-mask`, `pt-imp-v1` … `v3` |

Training and evaluation logs that used to sit here were moved to
`results/legacy_logs/chexzero/` (see `results/legacy_logs/MOVES.md`).
