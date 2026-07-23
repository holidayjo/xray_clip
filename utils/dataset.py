import os
import tarfile
import urllib.request
import pathlib
import pandas as pd
import torch
import torch.utils.data
import PIL.Image
import matplotlib.pyplot as plt

def download_dataset(output_dir="."):
    """Downloads and extracts the NIH Chest X-ray dataset, skipping completed steps."""
    root_dir  = pathlib.Path(output_dir)
    image_dir = root_dir / "nih_images"
    
    root_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # URLs for the zip/tar files
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]

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
    

def load_split(csv_path, image_root):
    """
    To match CSV IDs with image paths
    """
    df         = pd.read_csv(csv_path)
    id_to_path = {p.name: str(p) for p in image_root.rglob("*.png")}
    paths      = df['id'].map(id_to_path).values
    labels     = df.iloc[:, 1:-1].values
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


class XrayDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, df, label_cols, preprocess):
        self.image_paths = image_paths
        self.labels      = df[label_cols].values
        self.preprocess  = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.image_paths[idx]).convert("RGB")
        #image = self.preprocess(images=image, return_tensors="pt")["pixel_values"][0]
        image = self.preprocess(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label
    
    
def create_dataloaders(paths_dict, df_dict, top_labels, preprocess, batch_size=16, num_workers=2, seed=42):
    """Creates PyTorch DataLoaders for train, valid, and test splits."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    loaders = {}
    for split in ['train', 'valid', 'test']:
        dataset = XrayDataset(image_paths = paths_dict[split],
                              df          = df_dict[split],
                              label_cols  = top_labels,
                              preprocess  = preprocess)
        is_train       = (split == 'train') # Only shuffle the training dataset                
        loaders[split] = torch.utils.data.DataLoader(dataset,
                                                     batch_size  = batch_size,
                                                     shuffle     = is_train,
                                                     num_workers = num_workers,
                                                     generator   = generator if is_train else None)
        print(f"Created {split} loader with {len(dataset)} samples.")
        
    return loaders['train'], loaders['valid'], loaders['test']



def inspect_dataloader(dataloader):
    """Pulls a single batch from the dataloader, prints stats, and visualizes the first image."""
    # 1. Pull one batch of images and labels
    images, labels = next(iter(dataloader))
    single_image   = images[0]

    # 2. Print the tensor statistics
    print("--- Image Tensor Stats ---")
    print(f"Batch Shape        : {images.shape}")
    print(f"Single Image Shape : {single_image.shape}")
    print(f"Min Value          : {single_image.min().item():.4f}")
    print(f"Max Value          : {single_image.max().item():.4f}")
    print(f"Label Vector       : {labels[0].tolist()}")

    # 3. Prepare the tensor for Matplotlib
    # Shift from [Channels, Height, Width] to [Height, Width, Channels]
    img_to_show = single_image.permute(1, 2, 0).numpy()
    
    # Scale the values between 0 and 1 so Matplotlib doesn't throw a clipping warning
    img_to_show = (img_to_show - img_to_show.min()) / (img_to_show.max() - img_to_show.min())

    # 4. Display the image
    plt.figure(figsize=(4, 4))
    plt.imshow(img_to_show)
    plt.title(f"Label: {labels[0].tolist()}")
    plt.axis("off")
    plt.show()