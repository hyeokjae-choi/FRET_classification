import os
import os.path as osp
import random
from shutil import copyfile


data_dir = "D:\\hyeokjae\\data\\FRET_trace\\20191202"
raw_dir = osp.join(data_dir, "raw")
original_dir = osp.join(data_dir, "original")
if not osp.isdir(original_dir):
    os.makedirs(original_dir)

# read raw data
read_raw_data = {}
class_list = os.listdir(raw_dir)
for cls in class_list:
    read_raw_data[cls] = os.listdir(osp.join(raw_dir, cls, "mat files"))

# divide data into train, validation, test
divide_data = {"train": {}, "valid": {}, "test": {}}
for cls, raw_data in read_raw_data.items():
    shuffle_raw_data = random.sample(raw_data, len(raw_data))
    num_train_valid = int(len(shuffle_raw_data) * 0.8)
    num_train = int(num_train_valid * 0.8)

    divide_data["train"][cls] = shuffle_raw_data[:num_train]
    divide_data["valid"][cls] = shuffle_raw_data[num_train:num_train_valid]
    divide_data["test"][cls] = shuffle_raw_data[num_train_valid:]

# copy data from raw to original
for div in ["train", "valid", "test"]:
    if not osp.isdir(osp.join(original_dir, div)):
        os.makedirs(osp.join(original_dir, div))

    for cls, data_list in divide_data[div].items():
        if not osp.isdir(osp.join(original_dir, div, cls)):
            os.makedirs(osp.join(original_dir, div, cls))
        for data in data_list:
            src = osp.join(raw_dir, cls, "mat files", data)
            dst = osp.join(original_dir, div, cls, data)
            copyfile(src, dst)
