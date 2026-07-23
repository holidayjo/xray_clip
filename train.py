import yaml
import pathlib
import numpy as np
import torch
import torch.nn.functional
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, hamming_loss, confusion_matrix, average_precision_score

# Import your custom modules
import utils.dataset
import utils.models
import utils.utils

# TODO: Import or define AsymmetricLoss here
# from utils.losses import AsymmetricLoss 

def main():
    # 1. Initialize settings and load config
    utils.utils.set_random_seeds(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/cxr_dataset.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Load and filter dataset splits
    image_root = pathlib.Path(cfg['image_root'])
    train_df, train_paths, _ = utils.dataset.load_split(cfg['train_csv'], image_root)
    valid_df, valid_paths, _ = utils.dataset.load_split(cfg['valid_csv'], image_root)
    
    train_df_filtered, train_paths_filtered = utils.dataset.filter_dataset(train_df, train_paths, cfg['names'], cfg['all_labels'])
    valid_df_filtered, valid_paths_filtered = utils.dataset.filter_dataset(valid_df, valid_paths, cfg['names'], cfg['all_labels'])

    # 3. Load Model and Tokenizer
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, preprocess, tokenizer, device = utils.models.load_clip_model(model_name=model_name, freeze_backbone=True, device=device)

    # 4. Create DataLoaders
    train_loader, valid_loader, _ = utils.dataset.create_dataloaders(
        paths_dict={'train': train_paths_filtered, 'valid': valid_paths_filtered, 'test': []},
        df_dict={'train': train_df_filtered, 'valid': valid_df_filtered, 'test': []},
        top_labels=cfg['names'],
        preprocess=preprocess,
        batch_size=16
    )

    # 5. Initialize Adapter, Optimizer, and Loss
    num_labels = len(cfg['names'])
    Adapter = utils.models.DualBranchAdapter().to(device)
    optimizer = torch.optim.Adam(Adapter.parameters(), lr=3e-4)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05, disable_torch_grad_focal_loss=True)

    # 6. Pre-compute Text Features
    with torch.no_grad():
        label_texts = [f"A chest radiograph with {i}, characterized by specific radiological features in the pulmonary area, affecting the thoracic cavity." for i in cfg['names']]
        text_tokens = tokenizer(label_texts, context_length=77).to(device)
        text_features = model.encode_text(text_tokens)  
        text_features = F.normalize(text_features, dim=-1)

    # 7. Training Setup
    num_epochs = 1
    best_val_loss = float("inf")
    patience = 5
    counter = 0
    early_stop = False

    train_losses, val_losses, train_Accs, valid_Accs = [], [], [], []
    
    # 8. Main Training Loop
    for epoch in range(num_epochs):
        model.train()
        Adapter.train()
        train_loss = 0.0

        y_train_true, y_train_pred = [], []
        train_label_accuracies = {f"label_{i}": [] for i in range(num_labels)}
        
        # ... [Your exact training batch loop goes here unchanged] ...

        # Evaluate Validation
        model.eval()
        Adapter.eval()
        val_loss = 0.0
        y_val_true, y_val_pred = [], []

        with torch.no_grad():
            for images, labels in valid_loader:
                # ... [Your exact validation batch loop goes here unchanged] ...

        # ... [Your exact metric calculation and printing goes here unchanged] ...

        # ======= Early Stopping Check =======
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'adapter_state_dict': Adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'train_acc': train_Accs,
                'valid_loss': val_losses,
                'valid_acc': valid_Accs,
            }, "muldiff.pth")

            print(f"Validation loss improved. Model saved.")
        else:
            counter += 1
            print(f"No improvement for {counter}/{patience} epochs.")

            if counter >= patience:
                print("Early stopping triggered.")
                early_stop = True

        if early_stop:
            break

if __name__ == "__main__":
    main()