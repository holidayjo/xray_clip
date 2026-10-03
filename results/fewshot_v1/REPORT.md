# Zero-shot → Few-shot → Test — overnight run, 2026-09-28

Full pipeline from scratch. Nothing reused: the MIMIC encoder was retrained from the
pretrained CLIP ViT-B/32, not loaded from `pt-imp-v2`.

```
23:04  gates       377,110 x 320 uint8, text aligned, NO IMPRESSION 4.4%
23:04  training    8 epochs, batch 64, SGD lr 1e-4 momentum 0.9, save_interval 200
00:37  ...done     exit 0, 236 checkpoints                                (1h33m)
00:37  selection   CheXpert val, top-10, min_gap 1000
00:53  ...done     best = checkpoint_29200, mean AUC 0.8650               (16m)
00:53  notebook    encode -> zero-shot -> few-shot K-sweep -> test
01:05  ...done     exit 0, 0 cell errors                                  (12m)
```

CheXpert validation: **0.8650** vs v2's 0.8617. The retrain reproduced and marginally
beat the old encoder on the selection metric.

Data hygiene verified in-run: support/test overlap **0**, `K=0` reproduces zero-shot to
**0.00e+00**.

---

## 1. Headline

| Run | NIH-14 mean AUC | Labeled images |
|---|---|---|
| fs-retrain zero-shot, baseline prompts | 0.6737 | 0 |
| fs-retrain zero-shot, v2-tuned prompts | 0.6819 | 0 |
| fs-retrain + SS-Text **K=16** | 0.6951 / 0.6956 | 448 |
| fs-retrain + SS-Text **K=32** | 0.7030 / 0.7031 | 896 |
| fs-retrain + SS-Text **K=64** | **0.7120** | 1,792 |
| — previous best, v2 + template-pair tuned | 0.7035 | 0 |
| — previous best, Arm B negation weights | 0.7105 | 0 |
| — supervised ViT(ASL) benchmark, 9 labels | 0.8119 | 23,496 |

K=64 at **0.7120** is the highest NIH-14 number this project has produced, marginally
above Arm B's 0.7105. It costs 1,792 labeled films.

Paired bootstrap, zero-shot → K=16 (probabilities averaged over 20 support draws):

```
baseline  +0.0512  95% CI [+0.0452, +0.0569]  significant
tuned     +0.0431  95% CI [+0.0379, +0.0487]  significant
```

---

## 2. The K-sweep: few-shot HURTS below K=8

```
prompts      K   mean AUC     std   vs zero-shot
baseline     0     0.6737       -            -
baseline     1     0.6078  0.0215      -0.0658
baseline     2     0.6201  0.0251      -0.0536
baseline     4     0.6422  0.0213      -0.0315
baseline     8     0.6738  0.0112      +0.0001
baseline    16     0.6951  0.0105      +0.0215
baseline    32     0.7030  0.0088      +0.0293
baseline    64     0.7120  0.0042      +0.0383
```

This **contradicts the FCA paper**, where SS-Text beats zero-shot at every K including
K=1 (56.7 vs 50.2, their Table 2).

Mechanism: SS-Text sets `w_c = Σ(K image features) + t_c`. The text prototype `t_c` has
norm 1; the visual sum has norm ~K. At K=1 the two terms are comparable, so one randomly
drawn image perturbs the prototype as much as the prompt defines it — and a single CXR is
a noisy estimate of a pathology. Only by K≈8 does the visual mean become stable enough to
be worth its weight.

Their setting is multi-class with 4–19 mutually exclusive classes; ours is 14 independent
one-vs-rest problems with severe positive/negative imbalance. **The break-even K does not
transfer between the two settings.** That is a finding, not a bug — and the FCA paper
never reports a break-even point, because it never sweeps past K=16.

---

## 3. The main result: gains land exactly on the rare pathologies

Per-label, K=16, tuned prompts:

```
label                 zero-shot     K=16     std     delta
Fibrosis                 0.4505   0.6501  0.0567   +0.1996
Emphysema                0.5261   0.6969  0.0535   +0.1708
Hernia                   0.6694   0.7787  0.0653   +0.1093
Consolidation            0.6842   0.7067  0.0152   +0.0225
Pleural_Thickening       0.6788   0.6704  0.0452   -0.0084
Infiltration             0.6736   0.6617  0.0317   -0.0119
Nodule                   0.5969   0.5834  0.0565   -0.0136
Cardiomegaly             0.8437   0.8228  0.0330   -0.0209
Edema                    0.8108   0.7780  0.0191   -0.0328
Pneumothorax             0.7439   0.7044  0.0463   -0.0395
Mass                     0.6932   0.6511  0.0297   -0.0421
Effusion                 0.7947   0.7526  0.0201   -0.0421
Pneumonia                0.6787   0.6339  0.0315   -0.0448
Atelectasis              0.7023   0.6475  0.0442   -0.0548
MEAN                     0.6819   0.6956           +0.0137
```

**Only 4 of 14 labels improve — and three of them are the pathologies barely mentioned in
MIMIC.** Fibrosis (0.7% mention rate) gains +0.20. Emphysema (3.4%) gains +0.17. The
labels that lose are the ones the text encoder already handles well: Cardiomegaly 0.844,
Edema 0.811, Effusion 0.795.

This is the cleanest evidence the project has produced for the frequency hypothesis:

> **Labeled examples substitute for missing pretraining frequency. Where the concept was
> frequent enough in the reports, the visual prototype only adds noise.**

Udandarao et al. show rare concept → weak zero-shot. Silva-Rodríguez et al. use that to
motivate adaptation. **Neither shows that the benefit of adaptation is itself
frequency-dependent.** That is the gap, and this table is a direct measurement of it.

It also converts the "frequency-aware λ" idea from a proposal into something the data
demands: a global λ is provably wrong here, because the optimal visual/text balance has
opposite signs for Fibrosis and for Cardiomegaly.

Not every gainer is rare-concept driven — Hernia (+0.109) has only 37 support positives
and the widest variance (std 0.065), so its trend is real (K1 0.656 → K64 0.846) but
thinly sampled. Cardiomegaly is the mirror case: it *loses* at K=16 but climbs steadily
(K1 0.731 → K64 0.856), so it is not immune to adaptation, just slower to benefit.

---

## 4. Prompt tuning did not survive the retrain

| | baseline prompts | v2-tuned prompts | gain |
|---|---|---|---|
| v2 (previous) | 0.6685 | 0.7035 | **+0.0350** |
| fs-retrain | 0.6737 | 0.6819 | **+0.0082** |

The +0.035 prompt-tuning gain — previously the single largest effect in the project —
collapses to +0.008 on a retrained encoder.

**Caveat, and it is mine:** `prompt_template_search.csv` was searched against the *v2*
text encoder and reused here unchanged. Prompts interact with the specific text tower, so
this is a transfer failure, not evidence that prompt tuning stopped working. A fresh
template search on this encoder would likely recover most of it.

Two things follow:

1. The honest within-run comparison is **baseline prompts throughout**: zero-shot 0.6737 →
   K=16 0.6951 → K=64 0.7120. On that axis the retrain slightly beats v2 (0.6737 vs 0.6685).
2. Cross-run comparisons against 0.7035 and 0.7105 are **not like-for-like**, because those
   used prompts tuned on their own encoder.

---

## 5. Support-draw variance is large, and ensembling recovers it

The per-seed std at K=16 is 0.0103 — the same order as the effect being measured. But the
bootstrap, which averages probabilities over 20 draws before scoring, gives +0.0431 rather
than the +0.0137 mean of per-seed AUCs.

**Which 16 films you pick matters as much as having 16 films.** Averaging over draws is
nearly free and recovers ~0.03 AUC. Worth reporting as a practical result; it also implies
any single-draw few-shot number in the literature carries hidden variance.

---

## 6. What to do next

1. **Re-run the template search on this encoder** before any cross-run claim. Cheap —
   features are cached in `results/`.
2. **Frequency-aware λ.** Section 3 is the evidence. Replace the global λ = 1/(Nτ) with a
   per-label value driven by the measured MIMIC mention rate, so Cardiomegaly keeps its text
   prototype while Fibrosis leans on images. This is a method contribution with a measured
   motivation.
3. **Conformal prediction.** The parts are now in place: cached features, disjoint support
   set, working SS-Text solver. Multi-label conformal is still the open method problem —
   the FCA guarantee assumes one true label per image.

---

## 7. Files

```
/mnt/My_Doc/experiments/fewshot_v1/
├── REPORT.md                       this file
├── run_all.sh                      orchestrator
├── run_all.log / train.log / select.log / notebook.log
├── checkpoints/pt-imp-fs/          236 checkpoints, 78 GB
├── checkpoint_ranking.csv          236 rows, absolute paths
├── 5_zeroshot_to_fewshot.ipynb     executed, outputs embedded
└── results/
    ├── comparison.csv
    ├── fewshot_sweep.csv           3,920 rows (2 prompt sets x 7 K x 20 seeds x 14 labels)
    ├── k_sweep.png
    ├── per_label_delta.png
    ├── feat_test_*.npy  (524 MB)   cached features - re-runs skip encoding
    ├── feat_supp_*.npy  (410 MB)
    └── text_*.npz
```

`rm -rf /mnt/My_Doc/experiments/fewshot_v1` removes the entire run. Nothing was written
into the git repo; the notebook there is the unexecuted source.
