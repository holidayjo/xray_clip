import os
import yaml
import tarfile
import urllib.request
import pathlib
import pickle
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import PIL.Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

def download_dataset(cfg_path="data/cxr_dataset.yaml", output_dir="."):
    """Downloads and extracts the NIH Chest X-ray dataset, skipping completed steps."""
    root_dir  = pathlib.Path(output_dir)
    image_dir = root_dir / "nih_images"
    
    root_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    links = config.get("dataset_links", [])
    if not links:
        print(f"Error: No download links found in {cfg_path}. Aborting.")
        return

    print("--- Phase 1: Downloading ---")
    for idx, link in enumerate(links):
        fn_name = f'images_{idx+1:02d}.tar.gz'
        tar_path = root_dir / fn_name
        marker_path = image_dir / f'images_{idx+1:02d}.extracted'
        
        # Skip if already extracted OR already downloaded
        if marker_path.exists():
            print(f'{fn_name} is already extracted. Skipping download...')
            continue
        if tar_path.exists():
            print(f'{fn_name} already exists on disk. Skipping download...')
            continue
            
        print(f'Downloading {fn_name}...')
        urllib.request.urlretrieve(link, tar_path)

    print("\n--- Phase 2: Extracting ---")
    for i in range(1, len(links) + 1):
        fn_name = f'images_{i:02d}.tar.gz'
        tar_path = root_dir / fn_name
        marker_path = image_dir / f'images_{i:02d}.extracted'

        # Skip if already extracted
        if marker_path.exists():
            print(f"{fn_name} is already extracted. Skipping...")
            continue
        
        # Safety check if tar file is missing
        if not tar_path.exists():
            print(f"Warning: {tar_path} not found. Cannot extract.")
            continue

        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(image_dir)

        # Create a marker file so future runs know extraction is finished
        marker_path.touch()

        tar_path.unlink()  # Modern pathlib equivalent of os.remove
        print(f"Deleted {tar_path}")

    print("\nAll done. Please check the checksums and extracted files.")


def build_image_index(image_root, cache_path=None, force_rebuild=False):
    """Scans image_root once for every .png file and returns {filename: full_path}.
    Scanning a large image directory is slow and, since the dataset doesn't change
    between runs, the result is cached to disk (YOLO-style .cache file) so later
    runs load it instantly instead of re-scanning. Delete the cache file (or pass
    force_rebuild=True) if images are ever added/removed/moved."""
    image_root = pathlib.Path(image_root)
    if cache_path is None:
        cache_path = image_root.parent / f"{image_root.name}.index_cache.pkl"
    cache_path = pathlib.Path(cache_path)

    if cache_path.exists() and not force_rebuild:
        with open(cache_path, "rb") as f:
            id_to_path = pickle.load(f)
        print(f"[build_image_index] Loaded cached image index from {cache_path} ({len(id_to_path)} images).")
        return id_to_path

    print(f"[build_image_index] Scanning {image_root} for .png files (first run, result will be cached)...")
    id_to_path = {p.name: str(p) for p in image_root.rglob("*.png")}
    with open(cache_path, "wb") as f:
        pickle.dump(id_to_path, f)
    print(f"[build_image_index] Cached image index to {cache_path} ({len(id_to_path)} images).")
    return id_to_path


def load_split(csv_path, image_root, id_to_path=None, verbose=False):
    """
    To match CSV IDs with image paths
    """
    df = pd.read_csv(csv_path)
    if id_to_path is None:
        id_to_path = build_image_index(image_root)
    # print(id_to_path) # {'00005750_019.png': 'data/nih_images/images/00005750_019.png', ...}
    paths      = df['id'].map(id_to_path).values
    labels     = df.iloc[:, 1:-1].values

    if verbose:
        # Debug prints to check returned variables
        print(f"[load_split] Loaded {len(df)} rows from CSV. You may check how CSV file looks like.")
        print(f"[load_split] paths shape : {paths.shape}, sample: {paths[0] if len(paths) > 0 else 'None'}")
        print(f"[load_split] labels shape: {labels.shape}, sample row: {labels[0] if len(labels) > 0 else 'None'}")

    return df, paths, labels


def filter_dataset(df, paths, top_labels, label_cols):
    """Filters dataframe and paths to only include target labels with zero other findings."""
    has_top      = df[top_labels].sum(axis=1) > 0   # selecting top_labels column.
                                                    # summing in rows.
                                                    # make boolean.
    other_labels = [l for l in label_cols if l not in top_labels]
    no_other     = df[other_labels].sum(axis=1) == 0
    mask         = has_top & no_other
    return df[mask].reset_index(drop=True), paths[mask]


def filter_dataset_any_positive(df, paths, top_labels):
    """Keeps samples with at least one positive among top_labels, regardless of what
    other findings (target or non-target) are also present -- i.e. no purity
    restriction, unlike filter_dataset(). Used for reproducing baselines trained on
    the full multilabel reality rather than isolated single-condition cases."""
    mask = df[top_labels].sum(axis=1) > 0
    return df[mask].reset_index(drop=True), paths[mask]


def split_binary_label(df, paths, target_label):
    """Splits a (df, paths) split into a positive subset (target_label == 1, regardless of
    any co-occurring findings) and a negative subset (target_label == 0, i.e. every other
    image -- other diseases and "No Finding" alike). Unlike filter_dataset(), which only
    keeps top_labels-positive rows and so never yields a true negative sample, this gives a
    genuine binary positive/negative partition for single-label experiments."""
    is_pos = df[target_label] == 1
    pos_df, pos_paths = df[is_pos].reset_index(drop=True), paths[is_pos]
    neg_df, neg_paths = df[~is_pos].reset_index(drop=True), paths[~is_pos]
    return pos_df, pos_paths, neg_df, neg_paths


def sample_balanced_split(pos_df, pos_paths, neg_df, neg_paths, seed=None):
    """Draws a random subset of len(pos_df) rows (without replacement) from the negative
    pool and concatenates it with the (full) positive split, producing a 1:1 balanced
    split. Call this once per epoch with a different seed so the classifier sees a fresh
    random negative subset each epoch instead of one frozen draw."""
    n_pos      = len(pos_df)
    n_neg_pool = len(neg_df)
    if n_pos > n_neg_pool:
        raise ValueError(f"Not enough negative samples ({n_neg_pool}) to match positives ({n_pos}).")

    rng                = np.random.default_rng(seed)
    idx                = rng.choice(n_neg_pool, size=n_pos, replace=False)
    sampled_neg_df     = neg_df.iloc[idx].reset_index(drop=True)
    sampled_neg_paths  = neg_paths[idx]
    return concat_splits(pos_df, pos_paths, sampled_neg_df, sampled_neg_paths)


def concat_splits(df_a, paths_a, df_b, paths_b):
    """Concatenates two (df, paths) split pairs into one combined split, e.g. to
    refit on train+valid together after early stopping has picked a best_epoch."""
    combined_df    = pd.concat([df_a, df_b], ignore_index=True)
    combined_paths = np.concatenate([paths_a, paths_b])
    return combined_df, combined_paths


def build_train_augmentation():
    """Mild, clinically-conservative augmentation for chest X-rays, applied to the PIL
    image before the CLIP preprocess pipeline. Deliberately excludes horizontal flip
    (laterality can matter for some findings) and excludes cutout/aggressive cropping
    (could remove small findings like Nodule from a positive sample)."""
    return T.Compose([
        T.RandomRotation(degrees=7),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
    ])


class XrayDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, df, label_cols, preprocess, augment=None):
        self.image_paths = image_paths
        self.labels      = df[label_cols].values
        self.preprocess  = preprocess
        self.augment     = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.image_paths[idx]).convert("RGB")
        if self.augment is not None:
            image = self.augment(image)
        #image = self.preprocess(images=image, return_tensors="pt")["pixel_values"][0]
        image = self.preprocess(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label


def create_dataloaders(paths_dict, df_dict, top_labels, preprocess, batch_size=16, num_workers=2, seed=42, augment=None):
    """Creates PyTorch DataLoaders for train, valid, and test splits.
    augment (if given) is applied to the train split only -- valid/test always stay
    on the deterministic preprocess pipeline for comparable evaluation numbers."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders = {}
    for split in ['train', 'valid', 'test']:
        dataset = XrayDataset(image_paths = paths_dict[split],
                              df          = df_dict[split],
                              label_cols  = top_labels,
                              preprocess  = preprocess,
                              augment     = augment if split == 'train' else None)

        is_train       = (split == 'train') # Only shuffle the training dataset
        loaders[split] = torch.utils.data.DataLoader(dataset,
                                                     batch_size  = batch_size,
                                                     shuffle     = is_train,
                                                     num_workers = num_workers,
                                                     generator   = generator if is_train else None)
        print(f"Created {split} loader with {len(dataset)} samples.")
        
    return loaders['train'], loaders['valid'], loaders['test']



def inspect_dataloader(dataloader, split_name="DataLoader", class_names=['Infiltration', 'Effusion', 'Nodule'], num_images=5, ncols=5):
    """Pulls a single batch from the dataloader, prints stats, and visualizes multiple images with disease titles."""
    # 1. Pull one batch of images and labels
    images, labels = next(iter(dataloader))
    single_image   = images[0]

    # 2. Print the tensor statistics
    print(f"\n================ {split_name} Stats ================")
    print(f"Batch Shape        : {images.shape}")
    print(f"Single Image Shape : {single_image.shape}")
    print(f"Min Value          : {single_image.min().item():.4f}")
    print(f"Max Value          : {single_image.max().item():.4f}")
    print(f"Label Vector       : {labels[0].tolist()}")

    # 3. Prepare and display multiple images in a grid (e.g. 15 images, ncols=5 -> 3x5 grid)
    ncols     = min(ncols, num_images)
    nrows     = -(-num_images // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.8))
    axes      = np.atleast_1d(axes).ravel()

    for i in range(nrows * ncols):
        if i >= num_images:
            axes[i].axis("off")
            continue

        # Shift from [Channels, Height, Width] to [Height, Width, Channels]
        img_to_show = images[i].permute(1, 2, 0).numpy()

        # Scale the values between 0 and 1 so Matplotlib doesn't throw a clipping warning
        img_to_show = (img_to_show - img_to_show.min()) / (img_to_show.max() - img_to_show.min())

        # Translate the one-hot label vector into readable disease names
        pos_labels = [class_names[idx] for idx, val in enumerate(labels[i]) if val == 1]
        title      = ", ".join(pos_labels) if pos_labels else "No Finding"

        # Get the image path, clean file name, and numerical label values
        img_path   = dataloader.dataset.image_paths[i]
        file_name  = pathlib.Path(img_path).name
        label_vals = labels[i].tolist()

        # Print the full path to the console for easy debugging/copying
        print(f"[Sample {i+1}] File: {file_name} | Full Path: {img_path} | Vector: {label_vals}")

        # 4. Display the image in the subplot grid
        axes[i].imshow(img_to_show)
        axes[i].set_title(f"Sample {i+1}: {file_name}\n{title}\n{label_vals}", fontsize=10, fontweight='bold')
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()