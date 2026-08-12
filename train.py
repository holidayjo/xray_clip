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
    utils.utils.set_random_seeds(seed=opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(opt.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    exp_dir    = utils.utils.increment_path("runs/exp")
    save_path  = exp_dir / opt.save_path
    image_root = pathlib.Path(cfg['image_root'])
    
    
    # Save the run's configuration, YOLO-style, so a finished run is self-describing without
    # having to reconstruct which flags produced it. Written before training starts, so they
    # survive a crash or an interrupt.
    with open(exp_dir / "opt.yaml", "w") as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    with open(exp_dir / "cfg.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Saved run config to {exp_dir}/opt.yaml and {exp_dir}/cfg.yaml")
    
    
    
    # Load CLIP Model and Tokenizer using CLI model option
    model, preprocess, tokenizer, device = utils.models.load_clip_model(model_name=opt.clip_model, freeze_backbone=True, device=device)
    
    # Load dataset splits (image index built once and reused across all three splits)
    id_to_path = utils.dataset.build_image_index(image_root)
    if opt.split_source == "official":
        splits = utils.dataset.load_official_split(cfg['data_entry_csv'], cfg['train_val_list'],
                                                   cfg['test_list'], image_root, cfg['top_labels'],
                                                   id_to_path=id_to_path, val_fraction=0.2, seed=opt.seed, verbose=True)
        train_df, train_paths = splits['train']
        valid_df, valid_paths = splits['valid']
        test_df,  test_paths  = splits['test']
    else:
        train_df, train_paths, _ = utils.dataset.load_split(cfg['train_csv'], image_root, id_to_path=id_to_path, verbose=True)
        valid_df, valid_paths, _ = utils.dataset.load_split(cfg['valid_csv'], image_root, id_to_path=id_to_path)
        test_df,  test_paths,  _ = utils.dataset.load_split(cfg['test_csv'],  image_root, id_to_path=id_to_path)
    print(f"Split source: {opt.split_source}")
    
    
    # --filter-mode: filtering
    # "purity": keeps only images whose findings are entirely within top_labels (original behaviour); 
    # "any_positive": keeps any image with >=1 top_label regardless of what else co-occurs, 
    # matching train_resnet_baseline.py and the reference paper, so the two are comparable. 
    # val.py must be given the same mode or its test numbers won't line up.
    if opt.filter_mode == "any_positive":
        filter_fn = lambda df, paths: utils.dataset.filter_dataset_any_positive(df, paths, cfg['top_labels'])
    else:
        filter_fn = lambda df, paths: utils.dataset.filter_dataset(df, paths, cfg['top_labels'], cfg['all_labels'])
    print(f"Filter mode: {opt.filter_mode}")
    train_df_filtered, train_paths_filtered = filter_fn(train_df, train_paths)
    valid_df_filtered, valid_paths_filtered = filter_fn(valid_df, valid_paths)
    test_df_filtered,  test_paths_filtered  = filter_fn(test_df,  test_paths)
    # labels                                  = train_df_filtered[cfg['top_labels']].values.astype('float32')

    # Create DataLoaders using CLI batch-size option
    paths_dict = {'train': train_paths_filtered, 'valid': valid_paths_filtered, 'test' : test_paths_filtered}
    df_dict    = {'train': train_df_filtered,    'valid': valid_df_filtered,    'test' : test_df_filtered}
    train_augment = utils.dataset.build_train_augmentation() if opt.augment else None
    
    utils.dataset.summarize_splits({'train': train_df_filtered, 'valid': valid_df_filtered, 'test': test_df_filtered},
                                   cfg['top_labels'])
    
    train_loader, valid_loader, test_loader = utils.dataset.create_dataloaders(paths_dict  = paths_dict,
                                                                               df_dict     = df_dict,
                                                                               top_labels  = cfg['top_labels'],
                                                                               preprocess  = preprocess,
                                                                               batch_size  = opt.batch_size,
                                                                               num_workers = opt.num_workers,
                                                                               augment     = train_augment)

    if opt.cache_embeddings:
        # Encode ONLY the images that survive filtering -- any_positive discards every image
        # with none of the 9 labels (61% of this dataset), so encoding those would be waste.
        needed = set()
        for df in (train_df_filtered, valid_df_filtered, test_df_filtered):
            needed.update(df['id'].tolist())
        needed_paths = {i: id_to_path[i] for i in needed}
        print(f"Embedding cache: {len(needed_paths):,} of {len(id_to_path):,} images needed")

        cache = utils.dataset.build_embedding_cache(opt.embed_cache, model, preprocess, device,
                                                     needed_paths, opt.clip_model,
                                                     batch_size=opt.encode_batch_size,
                                                     num_workers=opt.num_workers,
                                                     force_rebuild=opt.rebuild_cache)
        srcs = {}
        for name, df in [('train', train_df_filtered), ('valid', valid_df_filtered)]:
            X = torch.from_numpy(utils.dataset.lookup_embeddings(cache, df['id'].values)).to(device)
            Y = torch.from_numpy(df[cfg['top_labels']].values.astype(np.float32)).to(device)
            srcs[name] = (X, Y)
        train_source, valid_source = srcs['train'], srcs['valid']
        n_train_batches = -(-train_source[0].shape[0] // opt.batch_size)
        n_valid_batches = -(-valid_source[0].shape[0] // opt.batch_size)
        print(f"Cached-embedding mode: train {tuple(train_source[0].shape)}, valid {tuple(valid_source[0].shape)}")
    else:
        train_source, valid_source       = train_loader, valid_loader
        n_train_batches, n_valid_batches = len(train_loader), len(valid_loader)


    # 5. Initialize Adapter, Optimizer, and Loss using CLI learning rate
    num_labels = len(cfg['top_labels'])
    Adapter    = utils.models.DualBranchAdapter_simple().to(device)
    optimizer  = torch.optim.Adam(Adapter.parameters(), lr=opt.lr)
    
    # AsymmetricLoss and pos_weight are two alternative answers to the same class-imbalance
    # problem, so exactly one is used -- never both.
    # if opt.pos_weight:
    pos_weight = utils.loss.compute_pos_weight(train_df_filtered, cfg['top_labels'], device=device)
    criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("Loss: BCEWithLogitsLoss with per-label pos_weight (computed from the train split)")
    for name, w in zip(cfg['top_labels'], pos_weight.tolist()):
        print(f"  {name:<15s} pos_weight = {w:6.2f}")
    # else:
    #     criterion  = utils.loss.AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05, disable_torch_grad_focal_loss=True)

    # 6. Pre-compute Text Features
    text_features = utils.models.encode_label_prompts(model, tokenizer, cfg['top_labels'],
                                                        opt.context_length, device, cfg['prompt_template'])

    # 7. Training Setup using CLI options
    best_val_map  = -float("inf")   # checkpoint / early-stopping criterion
    best_val_loss =  float("inf")   # still tracked, for reporting only
    counter       = 0
    early_stop    = False

    train_losses, val_losses, train_Accs, valid_Accs = [], [], [], []
    best_epoch                                       = None
    best_train_overall, best_train_per_label         = None, None
    best_val_overall,   best_val_per_label           = None, None

    # 8. Main Training Loop using CLI epochs option
    for epoch in tqdm(range(opt.epochs), desc="Training"):
        # model.train()
        Adapter.train()
        train_loss = 0.0

        y_train_true, y_train_pred = [], []
        train_label_accuracies     = {f"label_{i}": [] for i in range(num_labels)}   # {'label_0': [], 'label_1': [], 'label_2': []}
        
        # ---- Training Batch Loop ----
        batch_gen = utils.dataset.feature_batches(train_source, opt.batch_size, True, model, device,
                                                  torch.Generator().manual_seed(opt.seed + epoch))
        for image_features, labels in tqdm(batch_gen, total=n_train_batches, desc=f"Epoch {epoch+1}/{opt.epochs} [Train]", leave=False):
            # print(f"image_features.shape = {image_features.shape}, text_features.shape = {text_features.shape}") 
            # # image_features.shape = torch.Size([16, 512]), text_features.shape = torch.Size([3, 512])
            predictions = Adapter(image_features, text_features) # (C,B)
            # print(f"predictions = {predictions}") # (3, 16)
            loss        = criterion(predictions, labels)
            # print(f"loss = {loss.item()}") # loss = 0.14700116217136383
            
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
            batch_gen = utils.dataset.feature_batches(valid_source, opt.batch_size, False, model, device)
            for image_features, labels in tqdm(batch_gen, total=n_valid_batches,
                                               desc=f"Epoch {epoch+1}/{opt.epochs} [Valid]", leave=False):

                predictions = Adapter(image_features, text_features)
                loss        = criterion(predictions, labels)
                val_loss   += loss.item()
                predictions = torch.sigmoid(predictions)

                y_val_pred.append(predictions.detach().cpu().numpy())
                y_val_true.append(labels.cpu().numpy())

        y_train_pred_prob = np.concatenate(y_train_pred, axis=0)
        y_train_true      = np.concatenate(y_train_true, axis=0)
        y_val_pred_prob   = np.concatenate(y_val_pred, axis=0)
        y_val_true        = np.concatenate(y_val_true, axis=0)

        train_overall, train_per_label = utils.evaluation.compute_multilabel_metrics(y_train_true, y_train_pred_prob)
        val_overall,   val_per_label   = utils.evaluation.compute_multilabel_metrics(y_val_true,   y_val_pred_prob)

        train_loss /= len(train_loader)
        val_loss   /= len(valid_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_Accs.append(train_overall['accuracy'])
        valid_Accs.append(val_overall['accuracy'])

        val_per_class_str = " | ".join(f"{name}: Acc {row['accuracy']:.4f} AP {row['ap']:.4f}"
                                       for name, row in zip(cfg['top_labels'], val_per_label))
        tqdm.write(f"Epoch [{epoch+1}/{opt.epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"Val Acc: {val_overall['accuracy']:.4f} | Val mAP: {val_overall['mAP']:.4f} | {val_per_class_str}")

        utils.evaluation.log_epoch_to_csv(exp_dir / "results.csv", epoch + 1, train_loss, val_loss, train_overall, val_overall, val_per_label, cfg['top_labels'])
        
        # ======= Early Stopping Check =======
        # Selects on val mAP rather than val_loss: mAP measures ranking quality (how well
        # positives separate from negatives), whereas val_loss on imbalanced data can improve
        # just by growing better-calibrated toward the majority class. pos_weight also rescales
        # loss magnitude, so raw loss is not comparable across configs.
        if val_overall['mAP'] > best_val_map:
            best_val_map  = val_overall['mAP']
            best_val_loss = val_loss
            counter       = 0
            best_epoch    = epoch + 1
            best_train_overall, best_train_per_label = train_overall, train_per_label
            best_val_overall,   best_val_per_label   = val_overall,   val_per_label

            torch.save({'epoch': epoch + 1,
                        # 'model_state_dict': model.state_dict(),
                        'adapter_state_dict': Adapter.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_losses,
                        'train_acc': train_Accs,
                        'valid_loss': val_losses,
                        'valid_acc': valid_Accs,
                        # Recorded so val.py can reproduce this run's exact data preparation without
                        # the caller having to remember matching flags.
                        'filter_mode': opt.filter_mode,
                        'top_labels': cfg['top_labels'],
                        'split_source': opt.split_source,
                        'adapter_class': type(Adapter).__name__}, save_path)
                       

            tqdm.write(f"Validation mAP improved to {best_val_map:.4f}. Model saved.")
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

    print(f"\n===== Final Test Set Evaluation (Best Weights: {save_path}) =====")
    validate.run(cfg            = opt.cfg,
                 weights        = str(save_path),
                 clip_model     = opt.clip_model,
                 split          = "test",
                 batch_size     = opt.batch_size,
                 num_workers    = opt.num_workers,
                 context_length = opt.context_length,
                 seed           = opt.seed,
                 save_dir       = exp_dir,
                 filter_mode    = opt.filter_mode)

    # 9. Optional: Refit on Train+Valid combined for best_epoch epochs, no early stopping
    if opt.refit and best_epoch is None:
        print("Skipping refit: no epoch ever improved val_loss (best_epoch is None).")
    elif opt.refit:
        tqdm.write(f"\n===== Refitting on Train+Valid combined for {best_epoch} epochs =====")

        refit_df, refit_paths = utils.dataset.concat_splits(train_df_filtered, train_paths_filtered, valid_df_filtered, valid_paths_filtered)
        refit_dataset = utils.dataset.XrayDataset(image_paths=refit_paths, df=refit_df,
                                                   label_cols=cfg['top_labels'], preprocess=preprocess,
                                                   augment=train_augment)
        refit_loader  = torch.utils.data.DataLoader(refit_dataset, batch_size=opt.batch_size,
                                                     shuffle=True, num_workers=opt.num_workers)
        print(f"Refit loader: {len(refit_dataset)} samples (train+valid combined).")
        
        
        
        if opt.cache_embeddings:
            Xr = torch.cat([train_source[0], valid_source[0]], dim=0)
            Yr = torch.cat([train_source[1], valid_source[1]], dim=0)
            refit_source, n_refit_batches = (Xr, Yr), -(-Xr.shape[0] // opt.batch_size)
        else:
            refit_source, n_refit_batches = refit_loader, len(refit_loader)




        # refit_adapter   = utils.models.DualBranchAdapter().to(device)
        refit_adapter   = type(Adapter)().to(device)  # Why type(Adapter)(): it always matches whatever the main run used, so this can't drift again.
        refit_optimizer = torch.optim.Adam(refit_adapter.parameters(), lr=opt.lr)
        model.eval()  # backbone stays frozen/eval, same as during the main training loop

        for refit_epoch in tqdm(range(best_epoch), desc="Refit"):
            refit_adapter.train()
            refit_loss = 0.0

            batch_gen = utils.dataset.feature_batches(refit_source, opt.batch_size, True, model, device,
                                                      torch.Generator().manual_seed(opt.seed + refit_epoch))
            for image_features, labels in tqdm(batch_gen, total=n_refit_batches,
                                               desc=f"Refit Epoch {refit_epoch+1}/{best_epoch}", leave=False):

                predictions = refit_adapter(image_features, text_features)
                loss        = criterion(predictions, labels)

                refit_optimizer.zero_grad()
                loss.backward()
                refit_optimizer.step()

                refit_loss += loss.item()

            # refit_loss /= len(refit_loader)
            refit_loss /= n_refit_batches # Why: with --cache-embeddings, refit_source is a tensor tuple and len(refit_loader) would be the wrong divisor (it counts image batches, not embedding batches).
            tqdm.write(f"Refit Epoch [{refit_epoch+1}/{best_epoch}] Loss: {refit_loss:.4f}")

        refit_save_path = exp_dir / f"refit_{opt.save_path}"
        torch.save({'epoch'               : best_epoch,
                    # 'model_state_dict'    : model.state_dict(),
                    'adapter_state_dict'  : refit_adapter.state_dict(),
                    'optimizer_state_dict': refit_optimizer.state_dict(),
                    'filter_mode'         : opt.filter_mode,
                    'top_labels'          : cfg['top_labels'],
                    'split_source'        : opt.split_source,
                    'adapter_class'       : type(refit_adapter).__name__}, 
                   refit_save_path)
        print(f"Refit model saved to {refit_save_path}")

        print(f"\n===== Final Test Set Evaluation (Refit Weights: {refit_save_path}) =====")
        validate.run(cfg            = opt.cfg,
                     weights        = str(refit_save_path),
                     clip_model     = opt.clip_model,
                     split          = "test",
                     batch_size     = opt.batch_size,
                     num_workers    = min(opt.num_workers, 4),   # forking 20 workers off a live CUDA context deadlocks; a single eval pass does not need more than a few
                     context_length = opt.context_length,
                     seed           = opt.seed,
                     save_dir       = exp_dir,
                     filter_mode    = opt.filter_mode)

def parse_opt():
    parser = argparse.ArgumentParser(description="CLIP-Based Chest X-Ray Multi-Label Classification")
    parser.add_argument("--cfg", type=str, default="config/cxr_dataset_9class.yaml", help="Path to dataset YAML file")
    parser.add_argument("--clip_model", type=str, default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", help="Pre-trained CLIP model name")
    parser.add_argument("--epochs", type=int, default=200, help="Total number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1200, help="Total batch size")
    parser.add_argument("--lr", type=float, default=5e-2, help="Initial learning rate for optimizer")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience epochs")
    parser.add_argument("--seed", type=int, default=42, help="Global training random seed")
    parser.add_argument("--save-path", type=str, default="muldiff.pth", help="File path to save the best model checkpoint")
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--context_length", type=int, default=77, help="the length of the prompt text.")
    parser.add_argument("--refit", action="store_true", help="After training, retrain a fresh Adapter on train+valid combined for best_epoch epochs (no early stopping) and re-evaluate on the test set")
    parser.add_argument("--augment", action="store_true", help="Apply mild image augmentation (rotation, translation, brightness/contrast jitter) to the train split only")
    parser.add_argument("--pos-weight", action="store_true", help="Use BCEWithLogitsLoss with per-label pos_weight (num_neg/num_pos, computed from the train split) instead of AsymmetricLoss")
    parser.add_argument("--filter-mode", type=str, default="any_positive", choices=["purity", "any_positive"], help="purity: keep only images whose findings are entirely within top_labels (original). any_positive: keep any image with >=1 top_label, matching train_resnet_baseline.py and the reference paper")
    parser.add_argument("--split-source", type=str, default="official", choices=["prunecxr", "official"], help="prunecxr: the NIH-CXR-LT CSVs (test = 21081 imgs). official: NIH train_val_list/test_list, matching the reference paper's split exactly (test = 25596 imgs)")
    parser.add_argument("--cache-embeddings", action="store_true", help="Pre-encode the filtered images once and train on cached embeddings")
    parser.add_argument("--embed-cache", type=str, default="data/biomedclip_embeddings.npz")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--encode-batch-size", type=int, default=256)

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)