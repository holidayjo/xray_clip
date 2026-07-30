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
import val as validate  # for end-of-epoch mAP


def main(opt):
    # 1. Initialize settings and load config using CLI options
    utils.utils.set_random_seeds(seed=opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(opt.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    exp_dir   = utils.utils.increment_path("runs/exp")
    save_path = exp_dir / opt.save_path

    # 2. Load and filter dataset splits
    image_root               = pathlib.Path(cfg['image_root'])
    
    # 3. Load Model and Tokenizer using CLI model option
    model, preprocess, tokenizer, device = utils.models.load_clip_model(model_name=opt.clip_model, freeze_backbone=True, device=device)
    
    # Load dataset splits (image index built once and reused across all three splits)
    id_to_path = utils.dataset.build_image_index(image_root)
    train_df, train_paths, train_labels = utils.dataset.load_split(cfg['train_csv'], image_root, id_to_path=id_to_path, verbose=True)
    valid_df, valid_paths, valid_labels = utils.dataset.load_split(cfg['valid_csv'], image_root, id_to_path=id_to_path)
    test_df,  test_paths,  test_labels  = utils.dataset.load_split(cfg['test_csv'],  image_root, id_to_path=id_to_path)

    # filtering
    train_df_filtered, train_paths_filtered = utils.dataset.filter_dataset(train_df, train_paths, cfg['top_labels'], cfg['all_labels'])
    valid_df_filtered, valid_paths_filtered = utils.dataset.filter_dataset(valid_df, valid_paths, cfg['top_labels'], cfg['all_labels'])
    test_df_filtered,  test_paths_filtered  = utils.dataset.filter_dataset(test_df,  test_paths,  cfg['top_labels'], cfg['all_labels'])
    # labels                                  = train_df_filtered[cfg['top_labels']].values.astype('float32')


    # 4. Create DataLoaders using CLI batch-size option
    paths_dict = {'train': train_paths_filtered, 'valid': valid_paths_filtered, 'test' : test_paths_filtered}
    df_dict    = {'train': train_df_filtered,    'valid': valid_df_filtered,    'test' : test_df_filtered}
    train_loader, valid_loader, test_loader = utils.dataset.create_dataloaders(paths_dict  = paths_dict, 
                                                                               df_dict     = df_dict, 
                                                                               top_labels  = cfg['top_labels'], 
                                                                               preprocess  = preprocess, 
                                                                               batch_size  = opt.batch_size,
                                                                               num_workers = opt.num_workers)

    # 5. Initialize Adapter, Optimizer, and Loss using CLI learning rate
    num_labels = len(cfg['top_labels'])
    Adapter    = utils.models.DualBranchAdapter().to(device)
    optimizer  = torch.optim.Adam(Adapter.parameters(), lr=opt.lr)
    criterion  = utils.loss.AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05, disable_torch_grad_focal_loss=True)

    # 6. Pre-compute Text Features
    text_features = utils.models.encode_label_prompts(model, tokenizer, cfg['top_labels'],
                                                        opt.context_length, device, cfg['prompt_template'])

    # 7. Training Setup using CLI options
    best_val_loss = float("inf")
    counter       = 0
    early_stop    = False

    train_losses, val_losses, train_Accs, valid_Accs = [], [], [], []
    best_epoch = None
    best_train_overall, best_train_per_label = None, None
    best_val_overall,   best_val_per_label   = None, None

    # 8. Main Training Loop using CLI epochs option
    for epoch in tqdm(range(opt.epochs), desc="Training"):
        # model.train()
        Adapter.train()
        train_loss = 0.0

        y_train_true, y_train_pred = [], []
        train_label_accuracies     = {f"label_{i}": [] for i in range(num_labels)}   # {'label_0': [], 'label_1': [], 'label_2': []}
        
        # ---- Training Batch Loop ----
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs} [Train]", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            # print(f"image_features.shape = {image_features.shape}, text_features.shape = {text_features.shape}") 
            # # image_features.shape = torch.Size([16, 512]), text_features.shape = torch.Size([3, 512])
            predictions = Adapter(image_features, text_features)
            # print(f"predictions = {predictions}") # (3, 16)

            loss        = criterion(predictions, labels)
            
            optimizer.zero_grad() # clears .grad on every parameter the optimizer tracks 
            loss.backward()  
            optimizer.step() # updating weights.

            train_loss += loss.item() # loss sum for this batch.
            predictions = torch.sigmoid(predictions)
            y_train_pred.append(predictions.detach().cpu().numpy())
            y_train_true.append(labels.cpu().numpy())

        # Evaluate Validation
        model.eval()
        Adapter.eval()
        val_loss               = 0.0
        y_val_true, y_val_pred = [], []
        val_label_accuracies   = {f"label_{i}": [] for i in range(num_labels)}

        # ---- Validation Batch Loop ----
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{opt.epochs} [Valid]", leave=False):
                images = images.to(device)
                labels = labels.to(device)

                image_features = model.encode_image(images)
                image_features = torch.nn.functional.normalize(image_features, dim=-1)

                predictions = Adapter(image_features, text_features)
                loss        = criterion(predictions, labels)
                val_loss   += loss.item()
                predictions = torch.sigmoid(predictions)

                y_val_pred.append(predictions.detach().cpu().numpy())
                y_val_true.append(labels.cpu().numpy())
                
                
                
        y_train_pred_prob = np.concatenate(y_train_pred, axis=0)
        y_train_true       = np.concatenate(y_train_true, axis=0)
        y_val_pred_prob    = np.concatenate(y_val_pred, axis=0)
        y_val_true         = np.concatenate(y_val_true, axis=0)

        train_overall, train_per_label = utils.evaluation.compute_multilabel_metrics(y_train_true, y_train_pred_prob)
        val_overall,   val_per_label   = utils.evaluation.compute_multilabel_metrics(y_val_true,   y_val_pred_prob)

        train_loss /= len(train_loader)
        val_loss   /= len(valid_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_Accs.append(train_overall['accuracy'])
        valid_Accs.append(val_overall['accuracy'])

        val_per_class_str = " | ".join(
            f"{name}: Acc {row['accuracy']:.4f} AP {row['ap']:.4f}"
            for name, row in zip(cfg['top_labels'], val_per_label)
        )
        tqdm.write(f"Epoch [{epoch+1}/{opt.epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"Val Acc: {val_overall['accuracy']:.4f} | Val mAP: {val_overall['mAP']:.4f} | {val_per_class_str}")

        """
        # ---- Metric Calculation and Printing ----
        y_train_pred      = np.concatenate(y_train_pred, axis=0)
        y_train_true      = np.concatenate(y_train_true, axis=0)
        y_train_pred_prob = y_train_pred.copy()
        y_train_pred      = (y_train_pred > 0.5).astype(int)

        train_subset_acc  = accuracy_score(y_train_true, y_train_pred)
        train_Acc         = accuracy_score(y_train_true.ravel(), y_train_pred.ravel())
        train_precision   = precision_score(y_train_true, y_train_pred, average="micro", zero_division=0)
        train_recall      = recall_score(y_train_true, y_train_pred, average="micro", zero_division=0)
        train_f1          = f1_score(y_train_true, y_train_pred, average="micro")
        train_hamming     = hamming_loss(y_train_true, y_train_pred)
        train_mAP         = average_precision_score(y_train_true, y_train_pred_prob, average="macro")
        tn, fp, fn, tp    = confusion_matrix(y_train_true.ravel(), y_train_pred.ravel(), labels=[0, 1]).ravel()
        train_specificity = tn / (tn + fp + 1e-7)

        train_label_precisions     = {}
        train_label_recalls        = {}
        train_label_hamming_losses = {}
        train_label_f1_scores      = {}
        train_label_specificities  = {}
        train_label_aucs           = {}
        train_label_maps           = {}

        for i in range(num_labels):
            # print(f'y_train_true[:, {i}], y_train_pred[:, {i}] = {y_train_true[:, i]}, {y_train_pred[:, i]}')
            train_label_acc = accuracy_score(y_train_true[:, i], y_train_pred[:, i])
            train_label_accuracies[f"label_{i}"].append(train_label_acc)

            train_label_prec = precision_score(y_train_true[:, i], y_train_pred[:, i], zero_division=0)
            train_label_rec  = recall_score(y_train_true[:, i], y_train_pred[:, i], zero_division=0)
            train_label_precisions[f"label_{i}"] = train_label_prec
            train_label_recalls[f"label_{i}"] = train_label_rec

            train_label_hamming = hamming_loss(y_train_true[:, i], y_train_pred[:, i])
            train_label_hamming_losses[f"label_{i}"] = train_label_hamming

            train_label_f1 = f1_score(y_train_true[:, i], y_train_pred[:, i], zero_division=0)
            train_label_f1_scores[f"label_{i}"] = train_label_f1

            try:
                tn, fp, fn, tp = confusion_matrix(y_train_true[:, i], y_train_pred[:, i], labels=[0, 1]).ravel()
                train_label_spec = tn / (tn + fp + 1e-7)
            except:
                train_label_spec = float('nan')
            train_label_specificities[f"label_{i}"] = train_label_spec

            try:
                train_label_auc = roc_auc_score(y_train_true[:, i], y_train_pred_prob[:, i])
            except ValueError:
                train_label_auc = float('nan')
            train_label_aucs[f"label_{i}"] = train_label_auc

            try:
                # print(f'y_train_true[:, i], y_train_pred[:, i] = {y_train_true[:, i]}, {y_train_pred[:, i]}')
                train_label_map = average_precision_score(y_train_true[:, i], y_train_pred_prob[:, i])
            except ValueError:
                train_label_map = float('nan')
            train_label_maps[f"label_{i}"] = train_label_map

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_Accs.append(train_Acc)

        print(f"Epoch [{epoch+1}/{opt.epochs}] - Train Loss: {train_loss:.4f}")

        y_val_pred = np.concatenate(y_val_pred, axis=0)
        y_val_true = np.concatenate(y_val_true, axis=0)
        y_val_pred_prob = y_val_pred.copy()
        y_val_pred = (y_val_pred > 0.5).astype(int)

        val_subset_acc = accuracy_score(y_val_true, y_val_pred)
        val_Acc = accuracy_score(y_val_true.ravel(), y_val_pred.ravel())
        val_precision = precision_score(y_val_true, y_val_pred, average="micro", zero_division=0)
        val_recall = recall_score(y_val_true, y_val_pred, average="micro", zero_division=0)
        val_f1 = f1_score(y_val_true, y_val_pred, average="micro")
        val_hamming = hamming_loss(y_val_true, y_val_pred)
        val_mAP = average_precision_score(y_val_true, y_val_pred_prob, average="macro")

        tn, fp, fn, tp = confusion_matrix(y_val_true.ravel(), y_val_pred.ravel(), labels=[0, 1]).ravel()
        val_specificity = tn / (tn + fp + 1e-7)

        val_label_precisions = {}
        val_label_recalls = {}
        val_label_hamming_losses = {}
        val_label_f1_scores = {}
        val_label_specificities = {}
        val_label_aucs = {}
        val_label_maps = {}

        for i in range(num_labels):
            val_label_acc = accuracy_score(y_val_true[:, i], y_val_pred[:, i])
            val_label_accuracies[f"label_{i}"].append(val_label_acc)

            val_label_prec = precision_score(y_val_true[:, i], y_val_pred[:, i], zero_division=0)
            val_label_rec = recall_score(y_val_true[:, i], y_val_pred[:, i], zero_division=0)
            val_label_precisions[f"label_{i}"] = val_label_prec
            val_label_recalls[f"label_{i}"] = val_label_rec

            val_label_f1 = f1_score(y_val_true[:, i], y_val_pred[:, i], zero_division=0)
            val_label_f1_scores[f"label_{i}"] = val_label_f1

            try:
                tn, fp, fn, tp = confusion_matrix(y_val_true[:, i], y_val_pred[:, i], labels=[0, 1]).ravel()
                val_label_spec = tn / (tn + fp + 1e-7)
            except:
                val_label_spec = float('nan')
            val_label_specificities[f"label_{i}"] = val_label_spec

            try:
                val_label_auc = roc_auc_score(y_val_true[:, i], y_val_pred_prob[:, i])
            except ValueError:
                val_label_auc = float('nan')
            val_label_aucs[f"label_{i}"] = val_label_auc

            try:
                val_label_map = average_precision_score(y_val_true[:, i], y_val_pred_prob[:, i])
            except ValueError:
                val_label_map = float('nan')
            val_label_maps[f"label_{i}"] = val_label_map

            val_label_hamming = hamming_loss(y_val_true[:, i], y_val_pred[:, i])
            val_label_hamming_losses[f"label_{i}"] = val_label_hamming

        val_loss /= len(valid_loader)
        val_losses.append(val_loss)
        valid_Accs.append(val_Acc)

        print(f"Epoch [{epoch+1}/{opt.epochs}], "
              f"Train Loss      : {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train subset acc: {train_subset_acc:.4f}, Val Acc: {val_subset_acc:.4f}, "
              f"Train Acc       : {train_Acc:.4f}, Val Acc: {val_Acc:.4f}, "
              f"Train mAP       : {train_mAP:.4f}, Val mAP: {val_mAP:.4f}, "
              f"Train F1        : {train_f1:.4f}, Val F1: {val_f1:.4f}, "
              f"Train Hamming   : {train_hamming:.4f}, Val Hamming: {val_hamming:.4f}, "
              f"Train spec      : {train_specificity:.4f}, Val spec: {val_specificity:.4f}, "
              f"Train recall    : {train_recall:.4f}, Val recall: {val_recall:.4f}, "
              f"Train precision : {train_precision:.4f}, Val precision: {val_precision:.4f}")
        print("========== Per-Label Training Metrics ==========")
        for i in range(num_labels):
            print(f"[Label {i}]")
            print(f"  Accuracy        : {train_label_accuracies[f'label_{i}'][-1]:.4f}")
            print(f"  Precision       : {train_label_precisions[f'label_{i}']:.4f}")
            print(f"  Recall (Sens)   : {train_label_recalls[f'label_{i}']:.4f}")
            print(f"  Specificity     : {train_label_specificities[f'label_{i}']:.4f}")
            print(f"  F1 Score        : {train_label_f1_scores[f'label_{i}']:.4f}")
            print(f"  AUC             : {train_label_aucs[f'label_{i}']:.4f}")
            print(f"  AP              : {train_label_maps[f'label_{i}']:.4f}")
            print(f"  Hamming Loss    : {train_label_hamming_losses[f'label_{i}']:.4f}")
            print("")

        print("========== Per-Label Validation Metrics ==========")
        for i in range(num_labels):
            print(f"[Label {i}]")
            print(f"  Accuracy        : {val_label_accuracies[f'label_{i}'][-1]:.4f}")
            print(f"  Precision       : {val_label_precisions[f'label_{i}']:.4f}")
            print(f"  Recall (Sens)   : {val_label_recalls[f'label_{i}']:.4f}")
            print(f"  Specificity     : {val_label_specificities[f'label_{i}']:.4f}")
            print(f"  F1 Score        : {val_label_f1_scores[f'label_{i}']:.4f}")
            print(f"  AUC             : {val_label_aucs[f'label_{i}']:.4f}")
            print(f"  AP              : {val_label_maps[f'label_{i}']:.4f}")
            print(f"  Hamming Loss    : {val_label_hamming_losses[f'label_{i}']:.4f}")
        """
        # ======= Early Stopping Check =======
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_epoch = epoch + 1
            best_train_overall, best_train_per_label = train_overall, train_per_label
            best_val_overall,   best_val_per_label   = val_overall,   val_per_label

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'adapter_state_dict': Adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'train_acc': train_Accs,
                'valid_loss': val_losses,
                'valid_acc': valid_Accs,
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
    print("\nTrain Metrics (Best Epoch)")
    print(utils.evaluation.format_metrics_table(cfg['top_labels'], best_train_overall, best_train_per_label))
    print("\nValidation Metrics (Best Epoch)")
    print(utils.evaluation.format_metrics_table(cfg['top_labels'], best_val_overall, best_val_per_label))

    utils.plot.plot_training_curves(train_losses, val_losses, train_Accs, valid_Accs,
                                      save_path=str(exp_dir / "training_curves.png"))

    print(f"\n===== Final Test Set Evaluation (Best Weights: {save_path}) =====")
    validate.run(cfg            = opt.cfg,
                 weights        = str(save_path),
                 clip_model     = opt.clip_model,
                 split          = "test",
                 batch_size     = opt.batch_size,
                 num_workers    = opt.num_workers,
                 context_length = opt.context_length,
                 seed           = opt.seed)

    # 9. Optional: Refit on Train+Valid combined for best_epoch epochs, no early stopping
    if opt.refit and best_epoch is None:
        print("Skipping refit: no epoch ever improved val_loss (best_epoch is None).")
    elif opt.refit:
        tqdm.write(f"\n===== Refitting on Train+Valid combined for {best_epoch} epochs =====")

        refit_df, refit_paths = utils.dataset.concat_splits(train_df_filtered, train_paths_filtered, valid_df_filtered, valid_paths_filtered)
        refit_dataset = utils.dataset.XrayDataset(image_paths=refit_paths, df=refit_df,
                                                   label_cols=cfg['top_labels'], preprocess=preprocess)
        refit_loader  = torch.utils.data.DataLoader(refit_dataset, batch_size=opt.batch_size,
                                                     shuffle=True, num_workers=opt.num_workers)
        print(f"Refit loader: {len(refit_dataset)} samples (train+valid combined).")

        refit_adapter   = utils.models.DualBranchAdapter().to(device)
        refit_optimizer = torch.optim.Adam(refit_adapter.parameters(), lr=opt.lr)
        model.eval()  # backbone stays frozen/eval, same as during the main training loop

        for refit_epoch in tqdm(range(best_epoch), desc="Refit"):
            refit_adapter.train()
            refit_loss = 0.0

            for images, labels in tqdm(refit_loader, desc=f"Refit Epoch {refit_epoch+1}/{best_epoch}", leave=False):
                images = images.to(device)
                labels = labels.to(device)

                image_features = model.encode_image(images)
                image_features = torch.nn.functional.normalize(image_features, dim=-1)

                predictions = refit_adapter(image_features, text_features)
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
            'adapter_state_dict': refit_adapter.state_dict(),
            'optimizer_state_dict': refit_optimizer.state_dict(),
        }, refit_save_path)
        print(f"Refit model saved to {refit_save_path}")

        print(f"\n===== Final Test Set Evaluation (Refit Weights: {refit_save_path}) =====")
        validate.run(cfg            = opt.cfg,
                     weights        = str(refit_save_path),
                     clip_model     = opt.clip_model,
                     split          = "test",
                     batch_size     = opt.batch_size,
                     num_workers    = opt.num_workers,
                     context_length = opt.context_length,
                     seed           = opt.seed)

def parse_opt():
    parser = argparse.ArgumentParser(description="CLIP-Based Chest X-Ray Multi-Label Classification")
    parser.add_argument("--cfg", type=str, default="data/cxr_dataset.yaml", help="Path to dataset YAML file")
    parser.add_argument("--clip_model", type=str, default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", help="Pre-trained CLIP model name")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1200, help="Total batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate for optimizer")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience epochs")
    parser.add_argument("--seed", type=int, default=42, help="Global training random seed")
    parser.add_argument("--save-path", type=str, default="muldiff.pth", help="File path to save the best model checkpoint")
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--context_length", type=int, default=77, help="the length of the prompt text.")
    parser.add_argument("--refit", action="store_true", help="After training, retrain a fresh Adapter on train+valid combined for best_epoch epochs (no early stopping) and re-evaluate on the test set")    
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)