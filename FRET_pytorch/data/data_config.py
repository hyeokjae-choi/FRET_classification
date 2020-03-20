import os.path as osp
from easydict import EasyDict as edict

from .config import DATASET_DIR


__C = edict()
__C.FRET = edict()
__C.FRET.transition = edict(ds_dir=osp.join(DATASET_DIR, osp.join("FRET_trace", "20191202", "original")),)

DATASET_CONFIG = __C
