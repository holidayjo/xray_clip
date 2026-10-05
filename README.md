# xray_clip

Zero-shot and few-shot chest X-ray classification with CLIP-style models (CheXzero),
trained on MIMIC-CXR and tested on CheXpert and ChestX-ray14.

## Repository layout

```
xray_clip/
├── README.md               this file: layout + working notes
├── 6_frequency_predicts_adaptation.ipynb   CURRENT WORK: Law 1 / Law 2 (moves to notebooks/ when done)
├── notebooks/              finished analyses, in order
│   ├── 1_dataset_preparation.ipynb          MIMIC / CheXpert / NIH files, preprocessing
│   ├── 2_main.ipynb                         BiomedCLIP baselines (linear probe, adapters)
│   ├── 2_1_temp_chexzero_walkthrough.ipynb  CheXzero code walkthrough
│   ├── 3_frequency_analysis.ipynb           concept frequency vs zero-shot AUC (first look)
│   ├── 4_weighted_contrastive.ipynb         frequency-weighted contrastive training
│   └── 5_zeroshot_to_fewshot.ipynb          zero-shot -> SS-Text few-shot
├── config/                 YAML configs: all paths and settings live here, never in code
│   ├── datasets.yaml           WHERE the raw datasets are (MIMIC, CheXpert, NIH): the one place that says so
│   ├── freq_v1.yaml            notebook 6 experiment (points to datasets.yaml)
│   ├── concept_synonyms.yaml   concept vocabulary for mention counting
│   └── cxr_dataset.yaml        BiomedCLIP NIH training (train.py, val.py, 2_main)
├── utils/                  our Python code
│   ├── config.py               the one config loader (load_datasets, load_experiment, show)
│   ├── plot.py                 all plotting (training curves, two-laws figure)
│   ├── freq/                   notebook 6, one module per stage (see utils/freq/__init__.py)
│   └── dataset.py, models.py, loss.py, evaluation.py, utils.py   BiomedCLIP pipeline
├── train.py, val.py        BiomedCLIP training / evaluation scripts (YOLO-style, run from the root)
├── CheXzero/               official CheXzero code (rajpurkarlab), patched + our CheXzero-side scripts
│   ├── LOCAL_CHANGES.md        what is upstream, what we changed, what is ours  <- read this first
│   ├── data/                   preprocessed h5 / csv files       (git-ignored, ~70 GB)
│   └── checkpoints/            older training runs               (git-ignored, ~362 GB)
├── runs/                   experiment outputs: checkpoints, features, caches (git-ignored)
├── results/                small, tracked summaries of finished runs (REPORT.md, csv, png)
│   └── legacy_logs/            old CheXzero logs (see MOVES.md)
├── data/                   BiomedCLIP-era image features (*.npz; git-ignored). NIH images are in the dataset folder
├── files/                  papers and chat exports (git-ignored)
└── trash/                  retired experiments and configs, kept for reference (incl. utils_chexzero/ study copy)
```

Datasets live outside the repo, under `/mnt/My_Doc/dataset/CXR_dataset/`, and are referenced
only through `config/datasets.yaml` (and the full paths in `config/cxr_dataset.yaml`).

Not part of this project (git-ignored): `FCA/` (separate cloned repo, source of SS-Text) and
`missing_viewposition/` (X-ray images; never commit, MIMIC data use agreement).

## Working notes

Work to do.
* (20260627_to_do) To try.
  - Extract from claude or gemini that combining impression and findings and outputs the appropriate text for training. And then apply the zero shot training and apply the few shot learning and then test on the testset.
* (20260927_note) Few-shot learning. 
  - We can apply few-shot learning concept on LG tumor classification. It is a perfect application area.
  - Zero-shot has a limitation. It is hard to overcome the ViT performance.
  - CheXzero is not a complete zero-shot. There are papers addressing this issue. It is now a common issue.
  - We found that there are class name explicitly mentioned in the reports. And we also found that the more class named mentioned in the paper, the better performance.
  - We can now apply a few-shot learning. 
  - Is this few-shot learning affected by class imbalacement? Then we should solve this issue.
* (20260927_note) Foundation model
  - Foundation model problem - we train the model using the same domainn dataset for train, validation, and test. 
  - For example, we train the model on mimic cxr dataset and validate and test on the same dataset.
  - However, can we apply the model on other hospital dataset?
  - Different from ChestX-ray14 dataset only results, we train the model using mimic-cxr dataset, and use a few-shot learning and compare the result from the existing results.

  
* creating images features in cache to boost training speed. - Done
* adaptor design upate. (basic nn architecture approach for now.) - Done - Not working.
  * Adding skip-connection.
* Checking how the results change with different prompt - Not working.
* Zero-shot by applying different recent CLIP model - Not working.
* Previous better test results for overall accuracy. (JW) - 
* Metric for (1) only one class, (2) 2 classes or more. - Not working.

DONEs
* code check (inference phase) - Done 
* training curve check (with loss and val set results) - Done
* In main.ipynb,what does load_clip_model actually load? - Done


Meeting on 20260907
- train on mimic dataset --> test on chest 14 dataset (our old dataset)
- after that we can going to the llm
- svip q2 q3 target

Meeting on 20260914
- prompt change: "a photo of ...", 부정관사 확인.
- cos similarity
- (in training) understanding contrastive learning with shapes of each tensor
- (in inference) understaning constrastive learning with shapes of each tensor and if it finally outputs the probability.
- a better few labels are okay