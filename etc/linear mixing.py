import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

from fret_pytorch.dataloader.fret_dataloader import DataManager


dm = DataManager("", "train")
l = len(dm.wl)
scale = 0.1

for t in np.arange(0, 1, 0.01):
    x = t * dm.cy3 + (1 - t) * dm.cy5
    n = np.random.rand(l)

    x = x + scale * n

    plt.plot(dm.wl, x)
    plt.ylim(0, 1)
    plt.savefig(osp.join("linear_mixing", f"{t:.4f}.png"))
    plt.close()
