import argparse
import pathlib
import yaml
import numpy as np
import torch
from tqdm import tqdm

import utils.dataset
import utils.models
import utils.utils
import utils.evaluation


@torch.no_grad()
def run(cfg            = "data/cxr_dataset.yaml",
        weights        = "muldiff.pth",
        clip_model     = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        split          = "test",
        batch_size     = 300,
        num_workers    = 20,
        context_length = 77,
        seed           = 42,
        device         = None,
        save_dir       = None,
        filter_mode    = None):
    """Evaluates a trained Adapter checkpoint on one dataset split and prints the metrics table.

    save_dir: where to save results.txt/results.csv. Pass an existing directory (e.g. train.py's
    own exp_dir) to save alongside a training run without creating a new folder -- matching how
    YOLO's val.run() reuses train.py's save_dir instead of also writing to runs/val. Leave as
    None for standalone CLI usage, which auto-creates a fresh runs/val/expN directory.

    filter_mode: leave as None to read it from the checkpoint (train.py records the mode it
    trained with), so a checkpoint reproduces its own data preparation instead of depending on
    the caller to pass a matching flag. Pass an explicit value to override, which warns if it
    disagrees with what the checkpoint recorded."""
    utils.utils.set_random_seeds(seed=seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_dir is None:
        save_dir = utils.utils.increment_path("runs/val/exp")
    else:
        save_dir = pathlib.Path(save_dir)

    with open(cfg, "r") as f:
        cfg = yaml.safe_load(f)
    image_root                           = pathlib.Path(cfg['image_root'])

    # Loaded up-front (not just before load_state_dict) because the checkpoint carries the
    # filter_mode and label set train.py used, and both are needed to prepare the data below.
    checkpoint = torch.load(weights, map_location=device)

    ckpt_filter_mode = checkpoint.get('filter_mode')
    if filter_mode is None:
        filter_mode = ckpt_filter_mode or "purity"
        source      = "checkpoint" if ckpt_filter_mode else "default (checkpoint predates this field)"
    else:
        source = "explicit argument"
        if ckpt_filter_mode and ckpt_filter_mode != filter_mode:
            print(f"WARNING: filter_mode='{filter_mode}' overrides checkpoint's '{ckpt_filter_mode}' "
                  f"-- results will not match how this model was trained.")
    print(f"Filter mode: {filter_mode}  (from {source})")

    # DualBranchAdapter emits one logit per (image, label) pair, so NO weight shape depends on
    # the number of labels -- a checkpoint trained on 9 labels loads cleanly against a 3-label
    # cfg without any error, then silently scores against the wrong prompts. Hence this check.
    ckpt_labels = checkpoint.get('top_labels')
    if ckpt_labels and list(ckpt_labels) != list(cfg['top_labels']):
        print(f"WARNING: checkpoint was trained on {list(ckpt_labels)}\n"
              f"         but cfg specifies    {list(cfg['top_labels'])}\n"
              f"         Adapter weight shapes are independent of label count, so this does NOT\n"
              f"         fail loudly -- the metrics below would be meaningless. Check --cfg.")

    model, preprocess, tokenizer, device = utils.models.load_clip_model(model_name=clip_model, freeze_backbone=True, device=device)
    csv_key                              = {"train": "train_csv", "valid": "valid_csv", "test": "test_csv"}[split]
    df, paths, _                         = utils.dataset.load_split(cfg[csv_key], image_root, verbose=True)
    if filter_mode == "any_positive":
        df_filtered, paths_filtered      = utils.dataset.filter_dataset_any_positive(df, paths, cfg['top_labels'])
    else:
        df_filtered, paths_filtered      = utils.dataset.filter_dataset(df, paths, cfg['top_labels'], cfg['all_labels'])
    dataset                              = utils.dataset.XrayDataset(image_paths=paths_filtered, df=df_filtered, label_cols=cfg['top_labels'], preprocess=preprocess)
    loader                               = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Created {split} loader with {len(dataset)} samples.")

    Adapter    = utils.models.DualBranchAdapter().to(device)
    Adapter.load_state_dict(checkpoint['adapter_state_dict'])
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded weights from {weights} (epoch {checkpoint.get('epoch', '?')})")

    model.eval()
    Adapter.eval()

    text_features  = utils.models.encode_label_prompts(model, tokenizer, cfg['top_labels'], context_length, device, cfg['prompt_template'])
    y_true, y_pred = [], []
    for images, labels in tqdm(loader, desc=f"Evaluating [{split}]"):
        images = images.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)

        predictions = Adapter(image_features, text_features)
        predictions = torch.sigmoid(predictions)

        y_pred.append(predictions.cpu().numpy())
        y_true.append(labels.cpu().numpy())

    y_pred_prob = np.concatenate(y_pred, axis=0)
    y_true      = np.concatenate(y_true, axis=0)

    overall, per_label = utils.evaluation.compute_multilabel_metrics(y_true, y_pred_prob)

    print(f"\n===== {split.capitalize()} Set Results =====")
    table_text = utils.evaluation.format_metrics_table(cfg['top_labels'], overall, per_label)
    print(table_text)

    # Prefix with the checkpoint's stem so evaluating multiple checkpoints (e.g. best vs.
    # refit weights) into the same save_dir doesn't overwrite each other's results.
    weights_stem      = pathlib.Path(weights).stem
    results_txt_path = save_dir / f"{weights_stem}_{split}_results.txt"
    with open(results_txt_path, "w") as f:
        f.write(f"{split.capitalize()} Set Results (weights: {weights})\n\n")
        f.write(table_text + "\n")

    results_csv_path = save_dir / f"{weights_stem}_{split}_results.csv"
    utils.evaluation.save_results_csv(results_csv_path, overall, per_label, cfg['top_labels'],
                                       extra_fields={"split": split, "weights": str(weights)})
    print(f"Saved results to {results_txt_path} and {results_csv_path}")

    return overall, per_label


def parse_opt():
    parser = argparse.ArgumentParser(description="CLIP-Based Chest X-Ray Multi-Label Classification - Validation/Test")
    parser.add_argument("--cfg", type=str, default="data/cxr_dataset.yaml", help="Path to dataset YAML file")
    parser.add_argument("--weights", type=str, default="muldiff.pth", help="Path to trained checkpoint (.pth)")
    parser.add_argument("--clip_model", type=str, default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", help="Pre-trained CLIP model name")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"], help="Which dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=16, help="Total batch size")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--context_length", type=int, default=77, help="the length of the prompt text.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--filter-mode", type=str, default=None, choices=["purity", "any_positive"], help="Omit to use the mode recorded in the checkpoint (recommended). purity: keep only images whose findings are entirely within top_labels. any_positive: keep any image with >=1 top_label")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(cfg=opt.cfg, weights=opt.weights, clip_model=opt.clip_model, split=opt.split,
        batch_size=opt.batch_size, num_workers=opt.num_workers,
        context_length=opt.context_length, seed=opt.seed, filter_mode=opt.filter_mode)
