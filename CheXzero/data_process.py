import os

import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image
import h5py
import cv2
from typing import *
from pathlib import Path

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def load_data(filepath):
    dataframe = pd.read_csv(filepath)
    return dataframe

def get_cxr_paths_list(filepath): 
    dataframe = load_data(filepath)
    cxr_paths = dataframe['Path']
    return cxr_paths

'''
This function resizes and zero pads image 
'''
def preprocess(img, desired_size=320):
    old_size = img.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)  # ANTIALIAS was removed in Pillow 10; LANCZOS is the same filter
    # create a new image and paste the resized on it

    new_img = Image.new('L', (desired_size, desired_size))
    new_img.paste(img, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    return new_img

def _decode_one(args):
    """Decode and preprocess a single image. Module-level so it is picklable by Pool.

    Returns (array, None) on success and (None, exception) on failure, mirroring the
    per-image error tolerance of the original serial loop.
    """
    path, resolution = args
    try:
        # read image using cv2
        img = cv2.imread(str(path))
        # convert to PIL Image object
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        # preprocess
        img = preprocess(img_pil, desired_size=resolution)
        return np.asarray(img, dtype=np.uint8), None
    except Exception as e:
        return None, e


def img_to_hdf5(cxr_paths: List[Union[str, Path]], out_filepath: str, resolution=320,
                num_workers=None, chunksize=16):
    """
    Convert directory of images into a .h5 file given paths to all
    images.

    JPEG decode dominates the runtime, so it is spread over a process pool. Pool.imap
    preserves input order, and the parent still writes rows sequentially by index, so
    the output is byte-for-byte identical to the original serial implementation.
    """
    dset_size = len(cxr_paths)
    failed_images = []
    if num_workers is None:
        num_workers = max(1, min(16, (os.cpu_count() or 2) - 4))

    with h5py.File(out_filepath,'w') as h5f:
        # uint8 rather than h5py's float32 default: preprocess() returns an 8-bit 'L' image,
        # so the stored values are identical and the file is 4x smaller (38.6GB vs 154.5GB).
        img_dset = h5f.create_dataset('cxr', shape=(dset_size, resolution, resolution), dtype='uint8')
        work = ((path, resolution) for path in cxr_paths)
        with Pool(processes=num_workers) as pool:
            for idx, (arr, err) in enumerate(tqdm(pool.imap(_decode_one, work, chunksize=chunksize),
                                                  total=dset_size)):
                if err is None:
                    img_dset[idx] = arr
                else:
                    failed_images.append((cxr_paths[idx], err))
    print(f"{len(failed_images)} / {len(cxr_paths)} images failed to be added to h5.", failed_images)

def get_files(directory):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for file in filenames:
            if file.endswith(".jpg"):
                files.append(os.path.join(dirpath, file))
    return files

def get_cxr_path_csv(out_filepath, directory):
    files = get_files(directory)
    file_dict = {"Path": files}
    df = pd.DataFrame(file_dict)
    df.to_csv(out_filepath, index=False)

def section_start(lines, section=' IMPRESSION'):
    for idx, line in enumerate(lines):
        if line.startswith(section):
            return idx
    return -1

def section_end(lines, section_start):
    num_lines = len(lines)

def getIndexOfLast(l, element):
    """ Get index of last occurence of element
    @param l (list): list of elements
    @param element (string): element to search for
    @returns (int): index of last occurrence of element
    """
    i = max(loc for loc, val in enumerate(l) if val == element)
    return i 

def write_report_csv(cxr_paths, txt_folder, out_path):
    imps = {"filename": [], "impression": []}
    txt_reports = []
    for cxr_path in cxr_paths:
        tokens = cxr_path.split('/')
        study_num = tokens[-2]
        patient_num = tokens[-3]
        patient_group = tokens[-4]
        txt_report = txt_folder + patient_group + '/' + patient_num + '/' + study_num + '.txt'
        filename = study_num + '.txt'
        f = open(txt_report, 'r')
        s = f.read()
        s_split = s.split()
        if "IMPRESSION:" in s_split:
            begin = getIndexOfLast(s_split, "IMPRESSION:") + 1
            end = None
            end_cand1 = None
            end_cand2 = None
            # remove recommendation(s) and notification
            if "RECOMMENDATION(S):" in s_split:
                end_cand1 = s_split.index("RECOMMENDATION(S):")
            elif "RECOMMENDATION:" in s_split:
                end_cand1 = s_split.index("RECOMMENDATION:")
            elif "RECOMMENDATIONS:" in s_split:
                end_cand1 = s_split.index("RECOMMENDATIONS:")

            if "NOTIFICATION:" in s_split:
                end_cand2 = s_split.index("NOTIFICATION:")
            elif "NOTIFICATIONS:" in s_split:
                end_cand2 = s_split.index("NOTIFICATIONS:")

            if end_cand1 and end_cand2:
                end = min(end_cand1, end_cand2)
            elif end_cand1:
                end = end_cand1
            elif end_cand2:
                end = end_cand2            

            if end == None:
                imp = " ".join(s_split[begin:])
            else:
                imp = " ".join(s_split[begin:end])
        else:
            imp = 'NO IMPRESSION'
            
        imps["impression"].append(imp)
        imps["filename"].append(filename)
        
    df = pd.DataFrame(data=imps)
    df.to_csv(out_path, index=False)

