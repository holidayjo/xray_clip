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


def format_metrics_table(label_names, overall, per_label, tablefmt="github"):
    """Builds a printable table: one row per label plus a final Overall row."""
    headers = ["Label", "Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC", "AP", "Hamming"]

    rows = []
    for name, row in zip(label_names, per_label):
        rows.append([name, row['accuracy'], row['precision'], row['recall'],
                      row['specificity'], row['f1'], row['auc'], row['ap'], row['hamming']])

    rows.append(["Overall", overall['subset_acc'], overall['precision'], overall['recall'],
                 overall['specificity'], overall['f1'], "-", overall['mAP'], overall['hamming']])

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
        y_pred_prob.append(predictions.cpu().numpy())
        y_true.append(labels.cpu().numpy())

    total_loss  = total_loss / len(loader)
    y_true      = np.concatenate(y_true, axis=0)
    y_pred_prob = np.concatenate(y_pred_prob, axis=0)
    return total_loss, y_true, y_pred_prob 