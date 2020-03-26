import os.path as osp
import numpy as np
import random

from .utils import TransformerModule


class DataManager:
    def __init__(self, ds_name, transform, phase):
        if phase not in ["train", "val", "test"]:
            raise ValueError("Invalid name of phase ({}).".format(phase))
        self.num_data_per_epoch = 10000

        self.wl, self.cy3, self.cy5 = self.read_data()
        self.transform = TransformerModule(len(self.wl), transform, noise_scale=0.1)

    def __getitem__(self, index):
        return self.load_data()

    def __len__(self):
        return self.num_data_per_epoch

    @staticmethod
    def read_data():
        with open("../data/Cy3 em spectrum.txt", "r") as f_cy3:
            cy3 = f_cy3.read().split("\n")[:-1]
            cy3_wl = [int(line.split("\t")[0]) for line in cy3]
            cy3_spec = np.array([float(line.split("\t")[1]) for line in cy3], dtype=np.float32)

        with open("../data/Cy5 em spectrum.txt", "r") as f_cy5:
            cy5 = f_cy5.read().split("\n")[:-1]
            cy5_wl = [int(line.split("\t")[0]) for line in cy5]
            cy5_spec = np.array([float(line.split("\t")[1]) for line in cy5], dtype=np.float32)

        if cy3_wl != cy5_wl:
            raise ValueError("The wavelengths of cy3 and cy5 read from the file do not match")

        return cy3_wl, cy3_spec, cy5_spec

    def load_data(self):
        t = np.array(random.random(), dtype=np.float32)
        x = t * self.cy3 + (1 - t) * self.cy5

        # transform
        if self.transform.operate:
            x = self.transform.run(x)

        return np.expand_dims(x, axis=0), np.expand_dims(t, axis=0).astype(np.float32)
