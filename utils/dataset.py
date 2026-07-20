import os
import tarfile
import urllib.request
import pathlib

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