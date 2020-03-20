import os.path as osp
import glob

trained_dir = "D:\\hyeokjae\\2020\\project\\trace_classification\\trained"

max_score, max_idx, mean_score = 0.0, 0, 0.0
for i in range(1, 101):
    model_dir = osp.join(trained_dir, "stn3d", f"trans_notrans_{i}")
    log = glob.glob(osp.join(model_dir, "*.txt"))[0]

    read_list = []
    with open(log, "r") as fid:
        read_list += fid.read().split("\n")

    test_score = float(read_list[-2].split()[-1])
    mean_score += test_score / 100
    if test_score > max_score:
        max_score = test_score
        max_idx = i

print(f"Max_score: {max_score} in {max_idx}")
print(f"Mean_score: {mean_score}")
