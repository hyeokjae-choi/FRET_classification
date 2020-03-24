import os.path as osp
import numpy as np
import torch
from sklearn.metrics import mean_squared_error


class RMSEEvaluator:
    def __init__(self, th_val=0.5):
        self.th_val = th_val
        self.pred_list = []
        self.true_list = []

        self.best_score = 0.0

    def evaluation_step(self, pred, true):
        pred_batch = pred.cpu().numpy()
        true_batch = true.cpu().numpy()

        for batch_idx in range(0, true_batch.shape[0]):
            pred = pred_batch[batch_idx][0]
            true = true_batch[batch_idx][0]

            self.pred_list.append(pred)
            self.true_list.append(true)

    def evaluation_end(self):
        val_score = mean_squared_error(np.stack(self.true_list), np.stack(self.pred_list))
        self.pred_list = []
        self.true_list = []

        return val_score
