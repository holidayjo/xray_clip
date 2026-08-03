"""Reproduces the ResNet101 baseline from:
Mezina & Burget (2024), "Detection of post-COVID-19-related pulmonary diseases in
X-ray images using Vision Transformer-based neural network," Biomedical Signal
Processing and Control 87, 105380.

Paper's reported ResNet101 result (Table 3): Accuracy 0.7249, AUC 0.7936,
Sensitivity 0.7345, Specificity 0.7215, Precision 0.3114, F1 0.4087.

This is a plain, fully fine-tuned ImageNet-pretrained ResNet101 -- unrelated to the
CLIP+Adapter pipeline in train.py/val.py, which is left untouched. Known
simplifications vs. the paper (documented, not hidden):
  - No U-Net lung-segmentation crop before resizing; images are resized directly.
  - "Steps per epoch 1000" (a Keras-specific knob) is treated as a standard full
    pass over the training set instead.
  - The exact activation/dropout choices inside the paper's 3-layer classifier head
    (64, 32, num_labels neurons) aren't fully specified in the paper; ReLU between
    layers is used as a reasonable default.
"""
import argparse
import pathlib
import yaml
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from tqdm import tqdm

import utils.dataset
import utils.utils
import utils.loss
import utils.evaluation

# Paper Section 3.1 / Fig. 2: the 9 selected post-COVID-19-related diseases.
PAPER_LABELS = ["Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema",
                "Fibrosis", "Infiltration", "Nodule", "Pneumothorax"]

# Paper's reported ResNet101 baseline (Table 3), printed at the end for comparison.
PAPER_RESNET101_RESULTS = {
    "accuracy": 0.7249, "auc": 0.7936, "sensitivity": 0.7345,
    "specificity": 0.7215, "precision": 0.3114, "f1": 0.4087,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_resnet101_classifier(num_labels):
    """ImageNet-pretrained ResNet101, fully fine-tuned, with the paper's head: the
    original 1000-way FC layer is replaced with 3 dense layers (64, 32, num_labels).
    Outputs raw logits -- AsymmetricLoss applies sigmoid internally, and callers
    apply torch.sigmoid() themselves for metrics, matching train.py's convention."""
    model = tv_models.resnet101(weights=tv_models.ResNet101_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, num_labels),
    )
    return model


def build_transforms(image_size, augment):
    """Standard ImageNet normalization -- this is a plain torchvision ResNet, not
    CLIP, so it needs ImageNet mean/std rather than CLIP's own preprocessing."""
    ops = [T.Resize((image_size, image_size))]
    if augment:
        # Approximates the paper's augmentation (height/width shift 0.1, rotation
        # range 10, zoom range 0.2 -- originally Keras ImageDataGenerator params).
        ops.append(T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)))
    ops += [T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return T.Compose(ops)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc):
    model.eval()
    total_loss = 0.0
    y_true, y_pred_prob = [], []
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()

        y_pred_prob.append(torch.sigmoid(logits).detach().cpu().numpy())
        y_true.append(labels.cpu().numpy())

    total_loss /= len(loader)
    y_true      = np.concatenate(y_true, axis=0)
    y_pred_prob = np.concatenate(y_pred_prob, axis=0)
    return total_loss, y_true, y_pred_prob


def main(opt):
    utils.utils.set_random_seeds(seed=opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(opt.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    image_root = pathlib.Path(cfg['image_root'])

    exp_dir   = utils.utils.increment_path("runs/resnet_exp")
    save_path = exp_dir / opt.save_path

    # 1. Load splits (reuses train.py's cached image index -- no re-scanning).
    id_to_path = utils.dataset.build_image_index(image_root)
    train_df, train_paths, _ = utils.dataset.load_split(cfg['train_csv'], image_root, id_to_path=id_to_path, verbose=True)
    valid_df, valid_paths, _ = utils.dataset.load_split(cfg['valid_csv'], image_root, id_to_path=id_to_path)
    test_df,  test_paths,  _ = utils.dataset.load_split(cfg['test_csv'],  image_root, id_to_path=id_to_path)

    # 2. No purity filtering, matching the paper -- keep any image with >=1 of the 9
    # target diseases, regardless of what else co-occurs.
    train_df_f, train_paths_f = utils.dataset.filter_dataset_any_positive(train_df, train_paths, PAPER_LABELS)
    valid_df_f, valid_paths_f = utils.dataset.filter_dataset_any_positive(valid_df, valid_paths, PAPER_LABELS)
    test_df_f,  test_paths_f  = utils.dataset.filter_dataset_any_positive(test_df,  test_paths,  PAPER_LABELS)

    print(f"Train: {len(train_df_f)} | Valid: {len(valid_df_f)} | Test: {len(test_df_f)} "
          f"samples (paper reports 23496 / 5874 / 13871 -- exact counts may differ "
          f"slightly since we don't have their precise pre-filtering pipeline).")

    # 3. Datasets/loaders (reuses XrayDataset -- generic over any preprocess/augment).
    train_transform = build_transforms(opt.image_size, augment=True)
    eval_transform  = build_transforms(opt.image_size, augment=False)

    train_dataset = utils.dataset.XrayDataset(train_paths_f, train_df_f, PAPER_LABELS, train_transform)
    valid_dataset = utils.dataset.XrayDataset(valid_paths_f, valid_df_f, PAPER_LABELS, eval_transform)
    test_dataset  = utils.dataset.XrayDataset(test_paths_f,  test_df_f,  PAPER_LABELS, eval_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # 4. Model, optimizer, loss -- matching the paper's hyperparameters.
    model     = build_resnet101_classifier(num_labels=len(PAPER_LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    criterion = utils.loss.AsymmetricLoss(gamma_neg=opt.gamma_neg, gamma_pos=opt.gamma_pos,
                                           clip=opt.clip, disable_torch_grad_focal_loss=True)

    best_val_loss = float("inf")
    counter, early_stop = 0, False
    best_epoch = None
    best_val_overall, best_val_per_label = None, None

    for epoch in tqdm(range(opt.epochs), desc="Training"):
        model.train()
        train_loss = 0.0
        y_train_true, y_train_pred = [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs} [Train]", leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            y_train_pred.append(torch.sigmoid(logits).detach().cpu().numpy())
            y_train_true.append(labels.cpu().numpy())

        train_loss /= len(train_loader)
        y_train_true      = np.concatenate(y_train_true, axis=0)
        y_train_pred_prob = np.concatenate(y_train_pred, axis=0)
        train_overall, _  = utils.evaluation.compute_multilabel_metrics(y_train_true, y_train_pred_prob)

        val_loss, y_val_true, y_val_pred_prob = evaluate(model, valid_loader, criterion, device,
                                                          desc=f"Epoch {epoch+1}/{opt.epochs} [Valid]")
        val_overall, val_per_label = utils.evaluation.compute_multilabel_metrics(y_val_true, y_val_pred_prob)

        tqdm.write(f"Epoch [{epoch+1}/{opt.epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"Val Acc: {val_overall['accuracy']:.4f} | Val mAP: {val_overall['mAP']:.4f}")
        utils.evaluation.log_epoch_to_csv(exp_dir / "results.csv", epoch + 1, train_loss, val_loss,
                                           train_overall, val_overall, val_per_label, PAPER_LABELS)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_epoch = epoch + 1
            best_val_overall, best_val_per_label = val_overall, val_per_label

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            tqdm.write("Validation loss improved. Model saved.")
        else:
            counter += 1
            tqdm.write(f"No improvement for {counter}/{opt.patience} epochs.")
            if counter >= opt.patience:
                tqdm.write("Early stopping triggered.")
                early_stop = True

        if early_stop:
            break

    print(f"\n===== Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f}) =====")
    print(utils.evaluation.format_metrics_table(PAPER_LABELS, best_val_overall, best_val_per_label))

    # 5. Final test evaluation, reloading the best checkpoint (not just the last epoch).
    print(f"\n===== Final Test Set Evaluation (Best Weights: {save_path}) =====")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, y_test_true, y_test_pred_prob = evaluate(model, test_loader, criterion, device, desc="Evaluating [test]")
    test_overall, test_per_label = utils.evaluation.compute_multilabel_metrics(y_test_true, y_test_pred_prob)

    table_text = utils.evaluation.format_metrics_table(PAPER_LABELS, test_overall, test_per_label)
    print(table_text)
    with open(exp_dir / "test_results.txt", "w") as f:
        f.write(f"Test Set Results (weights: {save_path})\n\n{table_text}\n")
    utils.evaluation.save_results_csv(exp_dir / "test_results.csv", test_overall, test_per_label, PAPER_LABELS,
                                       extra_fields={"split": "test", "weights": str(save_path)})

    print("\nPaper's reported ResNet101 baseline (Table 3):")
    print(f"  Accuracy {PAPER_RESNET101_RESULTS['accuracy']:.4f}  AUC {PAPER_RESNET101_RESULTS['auc']:.4f}  "
          f"Sensitivity {PAPER_RESNET101_RESULTS['sensitivity']:.4f}  Specificity {PAPER_RESNET101_RESULTS['specificity']:.4f}  "
          f"Precision {PAPER_RESNET101_RESULTS['precision']:.4f}  F1 {PAPER_RESNET101_RESULTS['f1']:.4f}")
    print("Our reproduction (test set):")
    print(f"  Accuracy {test_overall['subset_acc']:.4f}  Recall(~Sens) {test_overall['recall']:.4f}  "
          f"Specificity {test_overall['specificity']:.4f}  Precision {test_overall['precision']:.4f}  "
          f"F1 {test_overall['f1']:.4f}")


def parse_opt():
    parser = argparse.ArgumentParser(
        description="Reproduce Mezina & Burget (2024) ResNet101 baseline on the 9-disease post-COVID-19 subset")
    parser.add_argument("--cfg", type=str, default="data/cxr_dataset.yaml", help="Path to dataset YAML file")
    parser.add_argument("--image-size", type=int, default=384, help="Paper resizes to 384x384")
    parser.add_argument("--epochs", type=int, default=100, help="Paper trains for 100 epochs")
    parser.add_argument("--batch-size", type=int, default=10, help="Paper's batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Paper's AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Paper's AdamW weight decay")
    parser.add_argument("--gamma-neg", type=float, default=5, help="Paper's ASL gamma_negative")
    parser.add_argument("--gamma-pos", type=float, default=1, help="Paper's ASL gamma_positive")
    parser.add_argument("--clip", type=float, default=0.001, help="Paper's ASL clip")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience epochs")
    parser.add_argument("--seed", type=int, default=42, help="Global training random seed")
    parser.add_argument("--save-path", type=str, default="resnet101_baseline.pth", help="Checkpoint filename within the run's exp dir")
    parser.add_argument("--num-workers", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
