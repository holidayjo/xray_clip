import csv
import pathlib
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                              recall_score, f1_score, hamming_loss,
                              confusion_matrix, average_precision_score)
from tabulate import tabulate


def compute_multilabel_metrics(y_true, y_pred_prob, threshold=0.5):
    """Computes overall (micro/macro) and per-label multi-label classification metrics.

    Parameters
    ----------
    y_true      : array [N, num_labels], binary ground-truth labels
    y_pred_prob : array [N, num_labels], predicted probabilities (post-sigmoid)
    threshold   : decision threshold applied to y_pred_prob

    Returns
    -------
    overall   : dict of scalar metrics aggregated across all labels
    per_label : list of dicts, one per label, in label order
    """
    y_pred = (y_pred_prob > threshold).astype(int)
    num_labels = y_true.shape[1]

    overall = {}
    overall['subset_acc'] = accuracy_score(y_true, y_pred)
    overall['accuracy']   = accuracy_score(y_true.ravel(), y_pred.ravel())
    overall['precision']  = precision_score(y_true, y_pred, average="micro", zero_division=0)
    overall['recall']     = recall_score(y_true, y_pred, average="micro", zero_division=0)
    overall['f1']         = f1_score(y_true, y_pred, average="micro")
    overall['hamming']    = hamming_loss(y_true, y_pred)
    overall['mAP']        = average_precision_score(y_true, y_pred_prob, average="macro")

    tn, fp, fn, tp = confusion_matrix(y_true.ravel(), y_pred.ravel(), labels=[0, 1]).ravel()
    overall['specificity'] = tn / (tn + fp + 1e-7)

    per_label = []
    for i in range(num_labels):
        row = {}
        row['accuracy']  = accuracy_score(y_true[:, i], y_pred[:, i])
        row['precision'] = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        row['recall']    = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        row['f1']        = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        row['hamming']   = hamming_loss(y_true[:, i], y_pred[:, i])

        try:
            tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()
            row['specificity'] = tn / (tn + fp + 1e-7)
        except Exception:
            row['specificity'] = float('nan')

        try:
            row['auc'] = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
        except ValueError:
            row['auc'] = float('nan')

        try:
            row['ap'] = average_precision_score(y_true[:, i], y_pred_prob[:, i])
        except ValueError:
            row['ap'] = float('nan')

        per_label.append(row)

    return overall, per_label


def log_epoch_to_csv(csv_path, epoch, train_loss, val_loss, train_overall, val_overall, val_per_label, label_names):
    """Appends one epoch's metrics as a row to a results.csv file (YOLO-style running log),
    writing the header only the first time the file is created."""
    csv_path = pathlib.Path(csv_path)

    header = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "train_mAP", "val_mAP"]
    row    = [epoch, train_loss, val_loss, train_overall['accuracy'], val_overall['accuracy'],
              train_overall['mAP'], val_overall['mAP']]

    for name, label_row in zip(label_names, val_per_label):
        header += [f"{name}_val_acc", f"{name}_val_ap"]
        row    += [label_row['accuracy'], label_row['ap']]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def save_results_csv(csv_path, overall, per_label, label_names, extra_fields=None):
    """Writes a single-row CSV summarizing one evaluation (e.g. a val.py run),
    overwriting any existing file at csv_path. extra_fields is an optional dict
    of extra leading columns (e.g. {'split': 'test', 'weights': '...'})."""
    csv_path = pathlib.Path(csv_path)
    extra_fields = extra_fields or {}

    header = list(extra_fields.keys()) + ["accuracy", "mAP"]
    row    = list(extra_fields.values()) + [overall['accuracy'], overall['mAP']]

    for name, label_row in zip(label_names, per_label):
        header += [f"{name}_acc", f"{name}_ap"]
        row    += [label_row['accuracy'], label_row['ap']]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)


def format_metrics_table(label_names, overall, per_label, tablefmt="github"):
    """Builds a printable table: one row per label plus a final Overall row.

    The Overall row's "Accuracy" column uses overall['accuracy'] (the standard
    per-element accuracy, directly comparable to each label's own Accuracy column)
    -- NOT overall['subset_acc'] (exact-match accuracy, which requires every single
    label to be simultaneously correct and collapses toward 0 as label count grows).
    Subset accuracy is still reported, in its own "Subset Acc" column, so it isn't
    lost -- it's just no longer mislabeled as if it were the same metric as
    "Accuracy" elsewhere in the table."""
    headers = ["Label", "Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC", "AP", "Hamming", "Subset Acc"]

    rows = []
    for name, row in zip(label_names, per_label):
        rows.append([name, row['accuracy'], row['precision'], row['recall'],
                      row['specificity'], row['f1'], row['auc'], row['ap'], row['hamming'], "-"])

    rows.append(["Overall", overall['accuracy'], overall['precision'], overall['recall'],
                 overall['specificity'], overall['f1'], "-", overall['mAP'], overall['hamming'],
                 overall['subset_acc']])

    return tabulate(rows, headers=headers, floatfmt=".4f", tablefmt=tablefmt)


@torch.no_grad()
def evaluate(loader, model, Adapter, text_features, criterion, device, desc="Eval"):
    """Runs a no-grad forward pass over loader and returns (avg_loss, y_true, y_pred_prob)."""
    Adapter.eval()
    total_loss = 0.0
    y_true, y_pred_prob = [], []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        predictions    = Adapter(image_features, text_features)
        loss           = criterion(predictions, labels)
        total_loss    += loss.item()

        predictions = torch.sigmoid(predictions)
        y_pred_prob.append(predictions.detach().cpu().numpy())
        y_true.append(labels.cpu().numpy())

    total_loss  = total_loss / len(loader)
    y_true      = np.concatenate(y_true, axis=0)
    y_pred_prob = np.concatenate(y_pred_prob, axis=0)
    return total_loss, y_true, y_pred_prob 