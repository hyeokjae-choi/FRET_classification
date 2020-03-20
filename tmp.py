import os.path as osp
import glob
from shutil import copyfile

root_dir = "./trained/raw_stn3d"

for i in range(1, 101):
    model_name = f"high_low_trans_{i}"
    log_file = glob.glob(osp.join(root_dir, model_name, "*.txt"))[0]

    read_list = []
    with open(log_file, "r") as fid:
        read_list += fid.read().split("\n")

    accuracy_list = []
    aver_loss_list = []
    for line in read_list:
        if "accuracy" in line:
            accuracy = float(line.split(":")[3][:-1])
            accuracy_list.append(accuracy)

        if "aver_loss" in line:
            aver_loss = float(line.split(":")[2][:-1])
            aver_loss_list.append(aver_loss)

    with open(osp.join(root_dir, f"{model_name}_accuracy.txt"), "w") as f:
        for item in accuracy_list:
            f.write("%s\n" % item)

    with open(osp.join(root_dir, f"{model_name}_loss.txt"), "w") as f:
        for item in aver_loss_list:
            f.write("%s\n" % item)
    a = 1
