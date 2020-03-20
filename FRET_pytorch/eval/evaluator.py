import os
import numpy as np
from sklearn.metrics import accuracy_score


def evaluation(preds, num_classes):

    pred_list, true_list = [], []
    for n, d in preds.items():
        pred_list.append(np.argmax(d["pred"], axis=0))
        true_list.append(d["true"])
    score = accuracy_score(true_list, pred_list)

    return score
