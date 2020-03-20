import os
import shutil
from ..main_utils import save_checkpoint


class EvalSaveManager(object):
    def __init__(self):
        self.eval_score_max_epoch = -1
        self.eval_score_max = -1
        self.eval_out_max = None

    def getBestEpoch(self):
        return self.eval_score_max_epoch

    def getBestScore(self):
        return self.eval_score_max

    def saveEval(self, net, opt, cur_epoch, cur_score, output_dir):
        if cur_score > self.eval_score_max:
            self.eval_score_max = cur_score
            self.eval_score_max_epoch = cur_epoch

            # Save checkpoint.
            save_checkpoint(
                state={"net": net.state_dict(), "opt": opt.state_dict(), "epoch": cur_epoch},
                path=os.path.join(output_dir, "best_model-checkpoint.pt"),
            )

            print(f"** new max score epoch - epoch: {self.eval_score_max_epoch}, score: {self.eval_score_max} **")
