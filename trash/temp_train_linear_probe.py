import argparse
import yaml
import pathlib
import numpy as np
import torch
from tqdm import tqdm

# Import your custom modules
import utils.dataset
import utils.models
import utils.utils
import utils.loss
import utils.evaluation
import utils.plot


@torch.no_grad()
def evaluate_image_only(loader, model, probe, criterion, device, desc="Eval"):
    """Local no-grad evaluation helper for the image-only linear-probe ablation.
    utils.evaluation.evaluate() can't be reused here -- its signature is hardcoded to
    the Adapter + text_features calling convention -- so this mirrors its structure
    closely, minus the text branch: no_grad, tqdm progress, accumulate y_true/
    y_pred_prob, return (avg_loss, y_true, y_pred_prob).

    NOTE on the .detach() below: utils/loss.py's AsymmetricLoss, when constructed with
    disable_torch_grad_focal_loss=True, calls the GLOBAL torch.set_grad_enabled(True)
    internally (not a properly scoped context manager) after computing the focal
    weight. That silently re-enables grad for the remainder of this no_grad-decorated
    function once criterion(...) returns. Any tensor-to-numpy conversion after the
    criterion(...) call in this function must therefore go through .detach() first,
    or it will crash with "Can't call numpy() on Tensor that requires grad". Do not
    "fix" this by removing the .detach() below even though the function is already
    @torch.no_grad()-decorated.
    """
    probe.eval()
    total_loss = 0.0
    y_true, y_pred_prob = [], []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        logits         = probe(image_features)
        loss           = criterion(logits, labels)
        total_loss    += loss.item()

        y_pred_prob.append(torch.sigmoid(logits).detach().cpu().numpy())
        y_true.append(labels.cpu().numpy())

    total_loss  = total_loss / len(loader)
    y_true      = np.concatenate(y_true, axis=0)
    y_pred_prob = np.concatenate(y_pred_prob, axis=0)
    return total_loss, y_true, y_pred_prob


def main(opt):
    # 1. Initialize settings and load config using CLI options
    utils.utils.set_random_seeds(seed=opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(opt.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    assert len(cfg['top_labels']) == 1, (
        "train_linear_probe.py expects a single-label cfg (top_labels must have exactly "
        "one entry, e.g. data/cxr_dataset_nodule.yaml) -- it trains target-label-vs-rest, "
        "balanced 1:1 every epoch, on frozen CLIP image embeddings only.")
    target_label = cfg['top_labels'][0]

    exp_dir   = utils.utils.increment_path("runs/exp")
    save_path = exp_dir / opt.save_path

    # 2. Load dataset splits
    image_root = pathlib.Path(cfg['image_root'])

    # 3. Load Model (and Tokenizer, unused -- this ablation has no text branch at all,
    # so the tokenizer returned by load_clip_model is received but never called)
    model, preprocess, tokenizer, device = utils.models.load_clip_model(model_name=opt.clip_model, freeze_backbone=True, device=device)

    # Load dataset splits (image index built once and reused across all three splits)
    id_to_path               = utils.dataset.build_image_index(image_root)
    train_df, train_paths, _ = utils.dataset.load_split(cfg['train_csv'], image_root, id_to_path=id_to_path, verbose=True)
    valid_df, valid_paths, _ = utils.dataset.load_split(cfg['valid_csv'], image_root, id_to_path=id_to_path)
    test_df,  test_paths,  _ = utils.dataset.load_split(cfg['test_csv'],  image_root, id_to_path=id_to_path)

    # Full (unfiltered) binary split on the target label: positives are every image where
    # target_label == 1 regardless of co-occurring findings; negatives are everything else
    # (other diseases and "No Finding" alike) -- a genuine positive/negative pool, unlike
    # filter_dataset()'s top_labels-positive-only rows.
    train_pos_df, train_pos_paths, train_neg_df, train_neg_paths = utils.dataset.split_binary_label(train_df, train_paths, target_label)
    print(f"[train] {len(train_pos_df)} positive ({target_label}) / {len(train_neg_df)} negative pool "
          f"-- {opt.epochs} epochs will each draw a fresh random {len(train_pos_df)}-sample negative subset.")

    train_augment = utils.dataset.build_train_augmentation() if opt.augment else None

    # Valid/test stay as the full, un-rebalanced split so reported accuracy reflects the
    # true (imbalanced) class distribution rather than the artificially balanced train set.
    valid_dataset = utils.dataset.XrayDataset(image_paths=valid_paths, df=valid_df, label_cols=[target_label], preprocess=preprocess)
    test_dataset  = utils.dataset.XrayDataset(image_paths=test_paths,  df=test_df,  label_cols=[target_label], preprocess=preprocess)
    valid_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    print(f"Created valid loader with {len(valid_dataset)} samples.")
    print(f"Created test loader with {len(test_dataset)} samples.")

    # 5. Initialize the image-only linear probe, Optimizer, and Loss using CLI learning rate.
    # num_labels is len(cfg['top_labels']) (not hardcoded to 1) so this stays correct if
    # someone later points --cfg at a multi-label yaml, though the assert above still
    # restricts this script to single-label use, like train_nodule_balanced.py.
    probe      = utils.models.ImageLinearProbe(num_labels=len(cfg['top_labels'])).to(device)
    optimizer  = torch.optim.Adam(probe.parameters(), lr=opt.lr)
    criterion  = utils.loss.AsymmetricLoss(gamma_neg=1, gamma_pos=1, clip=0, disable_torch_grad_focal_loss=True)

    # No text-feature pre-computation step -- this ablation is image-only (no
    # encode_label_prompts call, no prompt_template usage).

    # 6. Training Setup
    best_val_loss = float("inf")
    best_val_map  = 0.0
    counter       = 0
    early_stop    = False

    train_losses, val_losses, train_Accs, valid_Accs = [], [], [], []
    best_epoch = None
    best_train_overall, best_train_per_label = None, None
    best_val_overall,   best_val_per_label   = None, None

    # 7. Main Training Loop
    for epoch in tqdm(range(opt.epochs), desc="Training"):
        # Resample a fresh random negative subset every epoch, matched 1:1 to the (fixed)
        # positive set, so the classifier isn't stuck training on one arbitrary negative draw.
        epoch_train_df, epoch_train_paths = utils.dataset.sample_balanced_split(train_pos_df,
                                                                                train_pos_paths,
                                                                                train_neg_df,
                                                                                train_neg_paths,
                                                                                seed=opt.seed + epoch)
        train_dataset = utils.dataset.XrayDataset(image_paths=epoch_train_paths, df=epoch_train_df,
                                                   label_cols=[target_label], preprocess=preprocess,
                                                   augment=train_augment)
        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                                      shuffle=True, num_workers=opt.num_workers)
        if epoch == 0:
            print(f"Created train loader with {len(train_dataset)} samples (balanced 1:1, resampled every epoch).")

        probe.train()
        train_loss = 0.0
        y_train_true, y_train_pred = [], []

        # ---- Training Batch Loop ----
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs} [Train]", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            # Backbone is frozen and this ablation has no img_mlp branch needing
            # gradients to flow back through image_features, so compute it under
            # no_grad -- saves memory/compute (autograd would likely already skip
            # building a graph here anyway, since encode_image's own parameters all
            # have requires_grad=False, but this is explicit good practice).
            with torch.no_grad():
                image_features = model.encode_image(images)
                image_features = torch.nn.functional.normalize(image_features, dim=-1)

            predictions = probe(image_features)
            loss        = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.sigmoid(predictions)
            y_train_pred.append(predictions.detach().cpu().numpy())
            y_train_true.append(labels.cpu().numpy())

        # ---- Validation ----
        model.eval()
        val_loss, y_val_true, y_val_pred_prob = evaluate_image_only(
            valid_loader, model, probe, criterion, device,
            desc=f"Epoch {epoch+1}/{opt.epochs} [Valid]")

        y_train_pred_prob = np.concatenate(y_train_pred, axis=0)
        y_train_true      = np.concatenate(y_train_true, axis=0)

        train_overall, train_per_label = utils.evaluation.compute_multilabel_metrics(y_train_true, y_train_pred_prob)
        val_overall,   val_per_label   = utils.evaluation.compute_multilabel_metrics(y_val_true,   y_val_pred_prob)

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_Accs.append(train_overall['accuracy'])
        valid_Accs.append(val_overall['accuracy'])

        tqdm.write(f"Epoch [{epoch+1}/{opt.epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"Val Acc: {val_overall['accuracy']:.4f} | Val mAP: {val_overall['mAP']:.4f}")

        utils.evaluation.log_epoch_to_csv(exp_dir / "results.csv", epoch + 1, train_loss, val_loss,
                                           train_overall, val_overall, val_per_label, cfg['top_labels'])

        # ======= Early Stopping Check (selection criterion: val mAP, not val_loss --
        # val_loss is dominated by calibration collapse under 1:1-resampled training and
        # doesn't track genuine ranking improvement here; val_mAP is threshold-free) =======
        if val_overall['mAP'] > best_val_map:
            best_val_map  = val_overall['mAP']
            best_val_loss = val_loss  # kept for display only
            counter = 0
            best_epoch = epoch + 1
            best_train_overall, best_train_per_label = train_overall, train_per_label
            best_val_overall,   best_val_per_label   = val_overall,   val_per_label

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'probe_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'train_acc': train_Accs,
                'valid_loss': val_losses,
                'valid_acc': valid_Accs,
            }, save_path)
            tqdm.write("Validation mAP improved. Model saved.")

        else:
            counter += 1
            tqdm.write(f"No improvement for {counter}/{opt.patience} epochs.")
            if counter >= opt.patience:
                tqdm.write("Early stopping triggered.")
                early_stop = True

        if early_stop:
            break

    print(f"\n===== Best Epoch: {best_epoch} (Val mAP: {best_val_map:.4f}, Val Loss: {best_val_loss:.4f}) =====")

    print("\nTrain Metrics (Best Epoch)")
    print(utils.evaluation.format_metrics_table(cfg['top_labels'], best_train_overall, best_train_per_label))
    print("\nValidation Metrics (Best Epoch)")
    print(utils.evaluation.format_metrics_table(cfg['top_labels'], best_val_overall, best_val_per_label))

    utils.plot.plot_training_curves(train_losses, val_losses, train_Accs, valid_Accs,
                                      save_path=str(exp_dir / "training_curves.png"))

    # ---- Final Test Set Evaluation ----
    # (Not delegated to val.py's run(), since that internally calls filter_dataset(), which
    # for a single-label cfg only keeps target_label-positive rows and drops every true
    # negative -- exactly what this balanced-vs-rest experiment must NOT do at eval time.)
    def evaluate_and_save(weights_path, tag):
        print(f"\n===== Final Test Set Evaluation ({tag} Weights: {weights_path}) =====")
        checkpoint = torch.load(weights_path, map_location=device)
        probe.load_state_dict(checkpoint['probe_state_dict'])
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        test_loss, y_test_true, y_test_pred_prob = evaluate_image_only(
            test_loader, model, probe, criterion, device, desc="Evaluating [test]")
        test_overall, test_per_label = utils.evaluation.compute_multilabel_metrics(y_test_true, y_test_pred_prob)
        test_table = utils.evaluation.format_metrics_table(cfg['top_labels'], test_overall, test_per_label)
        print(test_table)

        weights_stem = pathlib.Path(weights_path).stem
        with open(exp_dir / f"{weights_stem}_test_results.txt", "w") as f:
            f.write(f"Test Set Results (weights: {weights_path})\n\n")
            f.write(test_table + "\n")
        utils.evaluation.save_results_csv(exp_dir / f"{weights_stem}_test_results.csv", test_overall, test_per_label,
                                           cfg['top_labels'], extra_fields={"split": "test", "weights": str(weights_path)})
        print(f"Saved results to {exp_dir / f'{weights_stem}_test_results.txt'} and {exp_dir / f'{weights_stem}_test_results.csv'}")

    evaluate_and_save(save_path, "Best")

    # 8. Optional: Refit on Train+Valid combined for best_epoch epochs, no early stopping
    if opt.refit and best_epoch is None:
        print("Skipping refit: no epoch ever improved val_loss (best_epoch is None).")
    elif opt.refit:
        tqdm.write(f"\n===== Refitting on Train+Valid combined for {best_epoch} epochs =====")

        valid_pos_df, valid_pos_paths, valid_neg_df, valid_neg_paths = utils.dataset.split_binary_label(valid_df, valid_paths, target_label)
        refit_pos_df,   refit_pos_paths   = utils.dataset.concat_splits(train_pos_df, train_pos_paths, valid_pos_df, valid_pos_paths)
        refit_neg_df,   refit_neg_paths   = utils.dataset.concat_splits(train_neg_df, train_neg_paths, valid_neg_df, valid_neg_paths)
        print(f"Refit pool: {len(refit_pos_df)} positive / {len(refit_neg_df)} negative (train+valid combined).")

        refit_probe     = utils.models.ImageLinearProbe(num_labels=len(cfg['top_labels'])).to(device)
        refit_optimizer = torch.optim.Adam(refit_probe.parameters(), lr=opt.lr)
        model.eval()  # backbone stays frozen/eval, same as during the main training loop

        for refit_epoch in tqdm(range(best_epoch), desc="Refit"):
            epoch_refit_df, epoch_refit_paths = utils.dataset.sample_balanced_split(
                refit_pos_df, refit_pos_paths, refit_neg_df, refit_neg_paths, seed=opt.seed + refit_epoch)
            refit_dataset = utils.dataset.XrayDataset(image_paths=epoch_refit_paths, df=epoch_refit_df,
                                                       label_cols=[target_label], preprocess=preprocess,
                                                       augment=train_augment)
            refit_loader  = torch.utils.data.DataLoader(refit_dataset, batch_size=opt.batch_size,
                                                         shuffle=True, num_workers=opt.num_workers)

            refit_probe.train()
            refit_loss = 0.0

            for images, labels in tqdm(refit_loader, desc=f"Refit Epoch {refit_epoch+1}/{best_epoch}", leave=False):
                images = images.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    image_features = model.encode_image(images)
                    image_features = torch.nn.functional.normalize(image_features, dim=-1)

                predictions = refit_probe(image_features)
                loss        = criterion(predictions, labels)

                refit_optimizer.zero_grad()
                loss.backward()
                refit_optimizer.step()

                refit_loss += loss.item()

            refit_loss /= len(refit_loader)
            tqdm.write(f"Refit Epoch [{refit_epoch+1}/{best_epoch}] Loss: {refit_loss:.4f}")

        refit_save_path = exp_dir / f"refit_{opt.save_path}"
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'probe_state_dict': refit_probe.state_dict(),
            'optimizer_state_dict': refit_optimizer.state_dict(),
        }, refit_save_path)
        print(f"Refit model saved to {refit_save_path}")

        probe = refit_probe  # so evaluate_and_save's load below matches its own state dict
        evaluate_and_save(refit_save_path, "Refit")


def parse_opt():
    parser = argparse.ArgumentParser(description="Image-only ablation: frozen CLIP image embeddings -> plain linear probe,"
                                     "single target label vs. everything else, resampled 1:1 balanced every epoch."
                                     "No text branch, no DualBranchAdapter, no tokenizer/prompt usage.")
    parser.add_argument("--cfg", type=str, default="data/cxr_dataset_nodule.yaml", help="Path to a single-label dataset YAML file (top_labels must contain exactly one label)")
    parser.add_argument("--clip_model", type=str, default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", help="Pre-trained CLIP model name")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1200, help="Total batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate for optimizer")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience epochs")
    parser.add_argument("--seed", type=int, default=42, help="Global training random seed")
    parser.add_argument("--save-path", type=str, default="nodule_linear_probe.pth", help="File path to save the best model checkpoint")
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--refit", action="store_true", help="After training, retrain a fresh ImageLinearProbe on train+valid combined (still resampled 1:1 every epoch) for best_epoch epochs and re-evaluate on the test set")
    parser.add_argument("--augment", action="store_true", help="Apply mild image augmentation (rotation, translation, brightness/contrast jitter) to the train split only")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
