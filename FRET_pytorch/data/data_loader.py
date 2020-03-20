import sys
import os
import os.path as osp
import glob
import numpy as np
from scipy import io
from abc import ABCMeta

from .utils import augment_trace
from .data_config import DATASET_CONFIG

max_intensity = 573.0


class DataManager(metaclass=ABCMeta):
    def __init__(self, ds_name, augment, div):
        if div not in ["train", "valid", "test"]:
            raise ValueError("Invalid name of ({}).".format(div))

        self.augment = augment

        # Find information of the dataset.
        if ds_name in DATASET_CONFIG.FRET.keys():
            ds_info = DATASET_CONFIG.FRET[ds_name]
            self.ds_dir = ds_info["ds_dir"]
        else:
            raise ValueError("Unexpected name of the dataset ({})".format(ds_name))

        raw_data = self.read_data(div)
        self.data = self.merge_class_as_binary(raw_data)

        # Weighting data to same ratio
        weighting = True if div == "train" else False
        self.weighted_idx = self.weighted_indexing(weighting)

    def __getitem__(self, index):
        name = self.weighted_idx[index]
        return self.load_data(name)

    def __len__(self):
        return len(self.weighted_idx)

    def size(self):
        return len(self.data)

    def read_data(self, div):
        class_list = os.listdir(osp.join(self.ds_dir, div))

        d = {}
        for idx, cls in enumerate(class_list):
            fnames = sorted(glob.glob(osp.join(self.ds_dir, div, cls, "*.mat")))
            fnames = [osp.basename(fpath) for fpath in fnames]
            for fname in fnames:
                mat_data = io.loadmat(osp.join(self.ds_dir, div, cls, fname))
                raw_trace = mat_data["output"][:, 1:]

                # tuning trace data
                r_trace = raw_trace[:, 0] / max_intensity  # normalize 0 to 1
                g_trace = raw_trace[:, 1] / max_intensity  # normalize 0 to 1
                e_trace = g_trace / (r_trace + g_trace)
                trace = np.stack([r_trace, g_trace, e_trace], axis=0)

                d.update(
                    {fname: {"trace": trace, "class": [idx], "num_channels": trace.shape[0], "length": trace.shape[1]}}
                )
        return d

    # merge class 0 and 1 as 0, and class 2 to 1
    def merge_class_as_binary(self, input_data):
        merged_data = {}
        for fname, data in input_data.items():
            if data["class"] == [0]:  # High
                merged_data.update({fname: {"trace": data["trace"], "class": [0]}})
            elif data["class"] == [1]:  # Low
                merged_data.update({fname: {"trace": data["trace"], "class": [0]}})
            elif data["class"] == [2]:  # Transition
                merged_data.update({fname: {"trace": data["trace"], "class": [1]}})
        return merged_data

    def weighted_indexing(self, weighting):
        weighted_list = []
        # TODO
        if weighting:
            weighted_list = list(self.data.keys())
        else:
            weighted_list = list(self.data.keys())

        return sorted(weighted_list)

    def load_data(self, name):
        trace = self.data[name]["trace"]
        r_trace = np.expand_dims(trace[0], axis=0)
        b_trace = np.expand_dims(trace[1], axis=0)
        c_trace = np.expand_dims(trace[2], axis=0)

        cls = self.data[name]["class"][0]

        if self.augment:
            trace = augment_trace(trace)

        return trace, cls, name
