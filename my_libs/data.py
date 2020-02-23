import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from collections.abc import Sequence
from PIL import Image
import os
import pathlib
import shutil
import tqdm
import multiprocessing


class KfoldSplitter(Sequence):

    def __init__(self, data, n_splits=5, stratified=False, target_col=None, shuffle=False, random_state=123):
        self.n_splits = n_splits
        self.data = data
        self.shuffle = shuffle
        self.stratified = stratified
        self.target_col = target_col
        if self.stratified:
            assert self.target_col is not None, "if stratified splitting, target column should be provided"
            self.splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            self.indices = [(train_idx, test_idx) for train_idx, test_idx in self.splitter.split(self.data, self.data[self.target_col])]
        else:
            self.splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            self.indices = [(train_idx, test_idx) for train_idx, test_idx in self.splitter.split(self.data)]

        
    
    def __len__(self):
        return self.n_splits
    
    def __getitem__(self, i):
        train_idx, test_idx = self.indices[i]
        return self.data.loc[train_idx], self.data.loc[test_idx]


image_format = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG"
}

def image_conversion(fp, out_folder, to="jpg"):

    ext = pathlib.Path(fp).suffix.replace(".", "")
    if ext != to:
        fname = pathlib.Path(fp).stem
        out_path = os.path.join(out_folder, fname+"."+to)
        im = Image.open(fp)
        im = im.convert("RGB") if im.mode != "RGB" else im
        im.save(out_path, image_format[to])
    else:
        shutil.copy(fp, out_folder)


def image_conversion_parallel(list_fp, out_folder, to="jpg"):
    list_args = [(fp, out_folder, to) for fp in list_fp]

    with multiprocessing.Pool() as pool:
        pool.starmap(image_conversion, tqdm.tqdm(list_args))