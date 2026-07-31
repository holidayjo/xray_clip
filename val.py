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
        save_dir       = None):
    """Evaluates a trained Adapter checkpoint on one dataset split and prints the metrics table.

    save_dir: where to save results.txt/results.csv. Pass an existing directory (e.g. train.py's
    own exp_dir) to save alongside a training run without creating a new folder -- matching how
    YOLO's val.run() reuses train.py's save_dir instead of also writing to runs/val. Leave as
    None for standalone CLI usage, which auto-creates a fresh runs/val/expN directory."""
    utils.utils.set_random_seeds(seed=seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_dir is None:
        save_dir = utils.utils.increment_path("runs/val/exp")
    else:
        save_dir = pathlib.Path(save_dir)

    with open(cfg, "r") as f:
        cfg = yaml.safe_load(f)
    image_root                           = pathlib.Path(cfg['image_root'])
    model, preprocess, tokenizer, device = utils.models.load_clip_model(model_name=clip_model, freeze_backbone=True, device=device)
    csv_key                              = {"train": "train_csv", "valid": "valid_csv", "test": "test_csv"}[split]
    df, paths, _                         = utils.dataset.load_split(cfg[csv_key], image_root, verbose=True)
    df_filtered, paths_filtered          = utils.dataset.filter_dataset(df, paths, cfg['top_labels'], cfg['all_labels'])
    dataset                              = utils.dataset.XrayDataset(image_paths=paths_filtered, df=df_filtered, label_cols=cfg['top_labels'], preprocess=preprocess)
    loader                               = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Created {split} loader with {len(dataset)} samples.")

    Adapter    = utils.models.DualBranchAdapter().to(device)
    checkpoint = torch.load(weights, map_location=device)
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
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(cfg=opt.cfg, weights=opt.weights, clip_model=opt.clip_model, split=opt.split,
        batch_size=opt.batch_size, num_workers=opt.num_workers,
        context_length=opt.context_length, seed=opt.seed)
