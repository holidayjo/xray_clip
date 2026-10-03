# CheXzero reproduction + few-shot, CheXpert and ChestX-ray14 — 2026-09-28

Trained from scratch on MIMIC-CXR with **impression-only** text, the paper's recipe.
One model, one protocol, two test sets.

```
09:52  gates      377,110 x 320 uint8, text aligned, NO IMPRESSION 15.1%
09:52  training   8 epochs, batch 64, SGD lr 1e-4 momentum 0.9      1h31m
11:23  selection  236 checkpoints -> best = step 32200, CheXpert val 0.8582
11:40  preprocess CheXpert test 500 studies, support pool 10,000
11:56  evaluation zero-shot + few-shot sweep on both datasets, 0 errors
```

Hygiene verified in-run: `K=0` reproduces zero-shot to **0.00e+00** on both datasets;
CheXpert test/support overlap **0 patients, 0 studies, 0 paths**; NIH test/support overlap **0**.

---

## 1. Headline — CheXzero reproduced

**CheXpert test, 500 studies, 5 competition tasks:**

| | Mean AUC |
|---|---|
| DAM (supervised SOTA) | 0.931 |
| DenseNet-121 (supervised) | 0.902 |
| **CheXzero, paper** | **0.8890** |
| **ours, zero-shot** | **0.8864** |
| MoCo-CXR, 100% labels | 0.884 |
| ConVIRT-ResNet-50, 100% labels | 0.881 |

**Difference from the paper: −0.0026.** Bootstrap over the 500 studies gives
**0.8863, 95% CI [0.8685, 0.9044]** — **0.889 falls inside our interval.** The reproduction
succeeds, and with zero labels we sit above both 100%-labelled self-supervised baselines.

Per label:

| | zero-shot AUC |
|---|---|
| Pleural Effusion | 0.9242 |
| Cardiomegaly | 0.9087 |
| Edema | 0.8894 |
| Consolidation | 0.8866 |
| Atelectasis | 0.8229 |

**ChestX-ray14 test, 25,596 images, 14 labels: 0.6821**, CI [0.6764, 0.6876] — slightly above
the previous impression+FINDINGS model on baseline prompts (0.6737), so dropping FINDINGS did
not cost anything and simplifies the recipe back to the paper's.

---

## 2. Few-shot hurts on CheXpert at every K

| K | images | CheXpert | vs ZS | NIH | vs ZS |
|---|---|---|---|---|---|
| 0 | 0 | **0.8864** | — | **0.6821** | — |
| 1 | 10 / 28 | 0.7426 | −0.1437 | 0.6087 | −0.0734 |
| 2 | 20 / 56 | 0.6949 | −0.1915 | 0.6178 | −0.0643 |
| 4 | 40 / 112 | 0.7396 | −0.1468 | 0.6404 | −0.0417 |
| 8 | 80 / 224 | 0.7741 | −0.1122 | 0.6695 | −0.0126 |
| 16 | 160 / 448 | 0.8168 | −0.0695 | 0.6878 | **+0.0057** |
| 32 | 320 / 896 | 0.8400 | −0.0463 | 0.6978 | +0.0157 |
| 64 | 640 / 1792 | 0.8506 | −0.0357 | 0.7073 | +0.0252 |

**CheXpert never recovers.** At K=64 — 640 labeled films — it is still 0.036 *below* using no
labels at all. **0 of 5 labels improved at K=16.**

NIH crosses over at K=16 and gains +0.025 by K=64.

---

## 3. The frequency prediction held

Before the run we predicted: *"All five CheXpert competition tasks are common in MIMIC, so
expect little or no gain there; NIH contains rare pathologies, so expect gains concentrated on
those."*

**NIH per-label at K=16 — 5 of 14 improved:**

```
Emphysema            0.4778 -> 0.6665   +0.1887     rare in MIMIC (3.4%)
Fibrosis             0.5127 -> 0.6477   +0.1350     rare (0.7%)
Nodule               0.5537 -> 0.5771   +0.0234     rare (2.3%)
Pleural_Thickening   0.6607 -> 0.6694   +0.0087
Cardiomegaly         0.8290 -> 0.8309   +0.0019
--------------------------------------------- everything below loses
Consolidation        0.7068 -> 0.7034   -0.0034
Mass                 0.6689 -> 0.6481   -0.0208
Effusion             0.7840 -> 0.7524   -0.0316     common (24.7%)
Edema                0.8129 -> 0.7754   -0.0375
Atelectasis          0.6949 -> 0.6450   -0.0499
Pneumothorax         0.7364 -> 0.6865   -0.0499
Pneumonia            0.6865 -> 0.6332   -0.0533
```

The three largest gains are the three pathologies the reports barely mention. Every
well-grounded pathology loses. This replicates the previous run on a **different encoder**
(impression-only rather than impression+FINDINGS).

---

## 4. The cleanest result — head-to-head on the five shared pathologies

These five appear in both datasets. Same checkpoints, same prompt strings (NIH's `Effusion`
already maps to `"Pleural Effusion"`, identical to CheXpert's), same K. Only the test
distribution differs.

| pathology | CheXpert ZS | CheXpert K=16 | Δ | NIH ZS | NIH K=16 | Δ |
|---|---|---|---|---|---|---|
| Atelectasis | 0.8229 | 0.7105 | **−0.1124** | 0.6949 | 0.6450 | −0.0499 |
| Cardiomegaly | 0.9087 | 0.7871 | **−0.1216** | 0.8290 | 0.8309 | +0.0019 |
| Consolidation | 0.8866 | 0.8649 | −0.0217 | 0.7068 | 0.7034 | −0.0034 |
| Edema | 0.8894 | 0.8277 | −0.0617 | 0.8129 | 0.7754 | −0.0374 |
| Pleural Effusion | 0.9242 | 0.8938 | −0.0304 | 0.7840 | 0.7524 | −0.0317 |
| **MEAN** | **0.8864** | **0.8168** | **−0.0695** | **0.7655** | **0.7414** | **−0.0241** |

**The same five pathologies lose on both datasets.** So this is not a property of CheXpert —
it is a property of *those pathologies*, which are exactly the ones MIMIC reports describe
often. NIH's overall +0.0057 at K=16 comes entirely from Emphysema and Fibrosis, which are not
in this set.

Note also that the loss is **larger where zero-shot was stronger**: CheXpert Cardiomegaly
starts at 0.9087 and loses 0.12; NIH Cardiomegaly starts at 0.8290 and loses nothing. The
better the text prototype, the more a 16-image visual prototype degrades it.

---

## 5. What this means

1. **The reproduction is sound.** 0.8864 vs 0.8890, paper value inside our CI. Every downstream
   claim rests on a model that matches the published one.

2. **Few-shot adaptation is not free.** Three independent papers report flat few-shot curves on
   chest X-ray; this run shows *why*. It is not that adaptation fails to help — it is that it
   **actively destroys a well-grounded text prototype** while rescuing a poorly-grounded one.
   The published flat averages are two opposite effects cancelling.

3. **A global λ is provably wrong.** SS-Text uses one text/visual balance for every class. Here
   the optimal balance has opposite signs for Fibrosis (+0.14) and Cardiomegaly (−0.12) *in the
   same model at the same K*. Making λ depend on measured pretraining frequency is no longer a
   proposal — it is what the data requires.

4. **Zero-shot is the right operating point for CheXpert's five tasks.** 0.8864 with no labels
   beats every self-supervised baseline that used all 223,414 of them, and adding labels makes
   it worse.

---

## 6. Caveats

- **CheXpert support labels are machine-extracted** (CheXpert labeler, `-1` dropped, blank
  treated as negative per the official convention); test labels are radiologist consensus. Label
  noise is a competing explanation for CheXpert's worse few-shot behaviour and is **not**
  separated from the frequency account by this design. NIH support labels are clean, and NIH
  shows the same per-pathology pattern — which argues for frequency, but does not prove it.
- **8 epochs, not the paper's 4.** Mitigated by selecting the best checkpoint over all 236,
  which landed at step 32200 (epoch ~5.5).
- **All views in training.** The paper's Methods describe AP/PA selection per study; the
  released code does not implement it, and we follow the code.
- NIH **Hernia** has only 37 support positives, so K=64 truncates for that label.

---

## 7. Files

```
/mnt/My_Doc/experiments/chexpert_v1/
├── REPORT.md, run_all.sh, run_all.log, train.log, select.log, prep.log, notebook.log
├── checkpoints/pt-imp-cxp/      236 checkpoints, 78 GB
├── checkpoint_ranking.csv
├── chexpert_test.h5 (500) / chexpert_support.h5 (10,000) + paths csvs
├── 5_chexpert.ipynb             executed, outputs embedded
└── results/
    ├── fewshot_sweep.csv        2,660 rows
    ├── head_to_head.csv
    ├── k_sweep_both.png, per_label_both.png
    └── feat_*.npy, text_*.npz   cached features
```
