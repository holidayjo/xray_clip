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

    # ---- backbone (frozen feature extractor) ----
    if opt.backbone == "resnet50":
        model, preprocess, _, device = utils.models.load_resnet_feature_extractor(weights=opt.resnet_weights, device=device)
        model_name                   = f"resnet50-{opt.resnet_weights}"
    else:
        model, preprocess, _, device = utils.models.load_clip_model(model_name      = opt.clip_model,
                                                                    freeze_backbone = True, 
                                                                    device          = device)
        model_name = opt.clip_model
    print(f"Backbone: {model_name}")

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

    cache = utils.dataset.build_embedding_cache(opt.embed_cache, model, preprocess, device,
                                                needed_paths, model_name,
                                                batch_size    = opt.encode_batch_size,
                                                num_workers   = opt.num_workers,
                                                force_rebuild = opt.rebuild_cache)
    X, Y = {}, {}
    for k, df in filtered.items():
        X[k] = utils.dataset.lookup_embeddings(cache, df['id'].values)
        Y[k] = df[label_names].values.astype(int)
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
    p.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "clip"])
    p.add_argument("--resnet-weights", type=str, default="IMAGENET1K_V2", choices=["IMAGENET1K_V1", "IMAGENET1K_V2"])
    p.add_argument("--clip_model", type=str, default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    p.add_argument("--embed-cache", type=str, default="data/resnet50_embeddings.npz", help="Must differ per backbone; the fingerprint check refuses a mismatched file")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--encode-batch-size", type=int, default=256)
    p.add_argument("--split-source", type=str, default="official", choices=["prunecxr", "official"])
    p.add_argument("--filter-mode", type=str, default="any_positive", choices=["purity", "any_positive"])
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--refit", action="store_true", help="Also refit on train+valid for best_iteration rounds")
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--early-stopping", type=int, default=50)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="XGBoost device")
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_opt())