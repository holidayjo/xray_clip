"""
Tree classifier on frozen-backbone features.
Reuses the existing pipeline end to end -- build_image_index, load_official_split, filter_dataset_*, summarize_splits, build_embedding_cache, lookup_embeddings,
compute_pos_weight, compute_multilabel_metrics, format_metrics_table, save_results_csv.
The only new thing here is the XGBoost fit, which cannot share train.py's epoch loop because it is not trained by SGD.
Backbones: any open_clip model (image branch only -- the text branch is unused here), or ImageNet ResNet50 as a plain feature extractor.
"""

import argparse
import pathlib
import yaml
import numpy as np
import xgboost as xgb
import torch

import utils.dataset
import utils.models
import utils.utils
import utils.loss
import utils.evaluation
from sklearn.linear_model import LogisticRegression


def main(opt):
    utils.utils.set_random_seeds(seed=opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(opt.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    label_names = cfg['top_labels']
    image_root  = pathlib.Path(cfg['image_root'])
    exp_dir     = utils.utils.increment_path("runs/xgb")

    with open(exp_dir / "opt.yaml", "w") as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    with open(exp_dir / "cfg.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # ---- backbone selection (models are loaded later, once per feature source) ----
    # "concat" uses BOTH caches: ImageNet ResNet50 (2048) + BiomedCLIP (512). They carry
    # complementary information -- concatenating beat either alone by ~0.03 macro AUC on
    # validation. Costs nothing extra once both .npz files exist.

    # --backbone accepts a comma-separated list; features from each are concatenated.
    # Concatenating complementary backbones was the single biggest AUC win on validation.
    # Cache paths are derived per backbone so switching never re-encodes an already-cached one.
    specs = []
    for kind in opt.backbone.split(","):
        kind = kind.strip()
        cache = opt.clip_cache if kind == "clip" else f"data/{kind}_embeddings.npz"
        specs.append((kind, cache))
    print(f"Backbones: {[k for k, _ in specs]}")


    # ---- splits (identical to train.py) ----
    id_to_path = utils.dataset.build_image_index(image_root)
    if opt.split_source == "official":
        splits = utils.dataset.load_official_split(cfg['data_entry_csv'], 
                                                   cfg['train_val_list'],
                                                   cfg['test_list'], 
                                                   image_root, 
                                                   label_names,
                                                   id_to_path   = id_to_path,
                                                   val_fraction = opt.val_fraction,
                                                   seed         = opt.seed, verbose=True)
        raw = {k: splits[k] for k in ('train', 'valid', 'test')}
    else:
        raw = {}
        for name, key in [('train', 'train_csv'), ('valid', 'valid_csv'), ('test', 'test_csv')]:
            df, paths, _ = utils.dataset.load_split(cfg[key], image_root, id_to_path=id_to_path)
            raw[name]    = (df, paths)
    print(f"Split source: {opt.split_source}")

    if opt.filter_mode == "any_positive":
        filt = lambda df, p: utils.dataset.filter_dataset_any_positive(df, p, label_names)
    else:
        filt = lambda df, p: utils.dataset.filter_dataset(df, p, label_names, cfg['all_labels'])
    filtered = {k: filt(df, p)[0] for k, (df, p) in raw.items()}
    print(f"Filter mode: {opt.filter_mode}")
    utils.dataset.summarize_splits(filtered, label_names)

    # ---- features (one-time encode, then cached) ----
    needed = set()
    for df in filtered.values():
        needed.update(df['id'].tolist())
    needed_paths = {i: id_to_path[i] for i in needed}
    print(f"Embedding cache: {len(needed_paths):,} of {len(id_to_path):,} images needed")

    X, Y = {}, {}
    for k, df in filtered.items():
        Y[k] = df[label_names].values.astype(int)

    parts = {k: [] for k in filtered}
    for kind, cache_path in specs:
        if kind != "clip":
            model, preprocess, _, device = utils.models.load_torchvision_feature_extractor(arch=kind, device=device)
            model_name                   = f"{kind}-imagenet"
        else:
            model, preprocess, _, device = utils.models.load_clip_model(model_name      = opt.clip_model,
                                                                        freeze_backbone = True,
                                                                        device          = device)
            model_name = opt.clip_model
        print(f"  loading {model_name}  ->  {cache_path}")

        cache = utils.dataset.build_embedding_cache(cache_path, model, preprocess, device,
                                                    needed_paths, model_name,
                                                    batch_size    = opt.encode_batch_size,
                                                    num_workers   = opt.num_workers,
                                                    force_rebuild = opt.rebuild_cache)
        for k, df in filtered.items():
            parts[k].append(utils.dataset.lookup_embeddings(cache, df['id'].values))

    for k in filtered:
        X[k] = np.concatenate(parts[k], axis=1) if len(parts[k]) > 1 else parts[k][0]
    model_name = "+".join(kind for kind, _ in specs)
    print(f"features: train {X['train'].shape}, valid {X['valid'].shape}, test {X['test'].shape}")


    # ---- XGBoost: one independent one-vs-rest model per label ----
    # scale_pos_weight is XGBoost's exact analogue of BCEWithLogitsLoss's pos_weight, so the
    # same helper supplies it; each label is its own binary problem, matching the multi-label
    # setup where the labels do not compete.
    pos_weight = utils.loss.compute_pos_weight(filtered['train'], label_names).numpy()
    X_all      = np.concatenate([X['train'], X['valid']], axis=0)
    Y_all      = np.concatenate([Y['train'], Y['valid']], axis=0)
    
    prob       = np.zeros_like(Y['test'], dtype=np.float32)
    refit_prob = np.zeros_like(prob) if opt.refit else None
    
    for j, name in enumerate(label_names):
        if opt.head == "logistic":
            # Beat XGBoost on every feature set tested (+0.02 macro AUC on ResNet features):
            # trees split on single dimensions, but these embeddings encode information
            # directionally across all 512/2048 of them. class_weight='balanced' replaces
            # scale_pos_weight and, unlike it, actually shifts the decision boundary, so
            # recall/F1 stop being degenerate. Do NOT standardise first -- that cost ~0.08 AUC.
            clf = LogisticRegression(C=opt.C, class_weight="balanced", max_iter=3000)
            clf.fit(X['train'], Y['train'][:, j])
            prob[:, j] = clf.predict_proba(X['test'])[:, 1]
            print(f"  [{name:<15s}] logistic C={opt.C}")

            if opt.refit:
                r = LogisticRegression(C=opt.C, class_weight="balanced", max_iter=3000)
                r.fit(X_all, Y_all[:, j])
                refit_prob[:, j] = r.predict_proba(X['test'])[:, 1]
        else:
            common = dict(max_depth=opt.max_depth, learning_rate=opt.lr,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=float(pos_weight[j]),
                          eval_metric="aucpr", tree_method="hist", device=opt.device,
                          n_jobs=opt.num_workers, random_state=opt.seed)

            clf = xgb.XGBClassifier(n_estimators=opt.rounds,
                                    early_stopping_rounds=opt.early_stopping, **common)
            clf.fit(X['train'], Y['train'][:, j],
                    eval_set=[(X['valid'], Y['valid'][:, j])], verbose=False)
            best_iter  = int(clf.best_iteration) + 1
            prob[:, j] = clf.predict_proba(X['test'])[:, 1]
            print(f"  [{name:<15s}] spw {pos_weight[j]:6.2f}  best_iteration {best_iter:>4d}")

            if opt.refit:
                # best_iteration plays the role best_epoch plays for the neural heads: the valid
                # split only picks the round count, then a fresh model trains on train+valid.
                r = xgb.XGBClassifier(n_estimators=best_iter, **common)
                r.fit(X_all, Y_all[:, j], verbose=False)
                refit_prob[:, j] = r.predict_proba(X['test'])[:, 1]


    # ---- evaluation (identical routines to train.py / val.py) ----
    for tag, p in [("best", prob)] + ([("refit", refit_prob)] if opt.refit else []):
        overall, per_label = utils.evaluation.compute_multilabel_metrics(Y['test'], p)
        table = utils.evaluation.format_metrics_table(label_names, overall, per_label)
        print(f"\n===== Test Set Results (xgboost / {model_name} / {tag}) =====")
        print(table)
        with open(exp_dir / f"xgb_{tag}_test_results.txt", "w") as f:
            f.write(f"backbone: {model_name}   variant: {tag}\n\n{table}\n")
        utils.evaluation.save_results_csv(exp_dir / f"xgb_{tag}_test_results.csv",
                                         overall, per_label, label_names,
                                         extra_fields={"backbone": model_name, "variant": tag})
    print(f"\nSaved to {exp_dir}")


def parse_opt():
    p = argparse.ArgumentParser(description="XGBoost on frozen-backbone features")
    p.add_argument("--cfg", type=str, default="config/cxr_dataset_9class.yaml")
    # p.add_argument("--backbone", type=str, default="concat", choices=["resnet50", "clip", "concat"])
    p.add_argument("--head", type=str, default="logistic", choices=["logistic", "xgboost"], help="logistic beat xgboost on these dense embeddings and fits in seconds")
    p.add_argument("--C", type=float, default=0.3, help="Inverse regularisation for --head logistic")
    # p.add_argument("--resnet-weights", type=str, default="IMAGENET1K_V2", choices=["IMAGENET1K_V1", "IMAGENET1K_V2"])
    p.add_argument("--clip_model", type=str, default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    p.add_argument("--resnet-cache", type=str, default="data/resnet50_embeddings.npz", help="Feature cache for the ResNet50 backbone")
    p.add_argument("--clip-cache",   type=str, default="data/biomedclip_embeddings.npz", help="Feature cache for the CLIP backbone")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--encode-batch-size", type=int, default=600)
    p.add_argument("--split-source", type=str, default="official", choices=["prunecxr", "official"])
    p.add_argument("--filter-mode", type=str, default="any_positive", choices=["purity", "any_positive"])
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--refit", action="store_true", help="Also refit on train+valid for best_iteration rounds")
    p.add_argument("--rounds", type=int, default=5000)
    p.add_argument("--early-stopping", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="XGBoost device")
    p.add_argument("--num_workers", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", type=str, default="clip,convnext_base", help="Comma-separated; features are concatenated. e.g. clip / resnet50 / swin_b / convnext_base / clip,convnext_base")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_opt())