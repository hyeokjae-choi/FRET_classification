import os.path as osp
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .evaluators import RMSEEvaluator


class PLMain(pl.LightningModule):
    def __init__(self, network, dataloader, optimizer, train_log_interval):
        super(PLMain, self).__init__()
        self.network = network
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss = nn.MSELoss()
        self.train_log_interval = train_log_interval
        self.__get_curidx = lambda x: self.current_epoch * len(self.dataloader["train"]) + x
        self.train_avg_loss = 0.0
        self.train_avg_cnt = 0
        self.evaluator = RMSEEvaluator()
        self.val_best_score = 0.0

    def forward(self, batch):  # require
        x = batch[0]
        t = batch[1]
        pred = self.network(x)
        loss = self.loss(pred, t)
        return pred, loss

    def training_step(self, batch, batch_nb):  # require
        pred, loss = self.forward(batch)
        self.train_avg_loss += loss.mean()
        self.train_avg_cnt += 1

        if self.__get_curidx(batch_nb) % self.train_log_interval == 0:
            tensorboard_logs = {"avg_train_loss": self.train_avg_loss / self.train_avg_cnt}

            self.train_avg_loss = 0.0
            self.train_avg_cnt = 0

            return {
                "loss": loss,
                "progress_bar": {"train_loss": loss.item()},
                "log": tensorboard_logs,
            }

        return {"loss": loss, "progress_bar": {"train_loss": loss.item()}}

    def validation_step(self, batch, batch_nb):
        pred, loss = self.forward(batch)

        self.evaluator.evaluation_step(pred, batch[1])
        return {"val_loss": loss}

    def validation_end(self, outputs):
        val_eval = self.evaluator.evaluation_end()

        # model saver
        if val_eval > self.val_best_score:
            torch.save(
                {"net": self.network.state_dict(), "opt": self.optimizer.state_dict(), "epoch": self.current_epoch},
                osp.join(
                    self.logger.save_dir,
                    self.logger.name,
                    f"version_{self.logger.version}",
                    "checkpoints",
                    "best_val_score.model",
                ),
            )
            self.val_best_score = val_eval

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"avg_val_loss": avg_loss, "avg_val_eval": val_eval}

        return {"avg_val_loss": avg_loss, "log": logs, "progress_bar": logs}

    def test_step(self, batch, batch_nb):
        pred, loss = self.forward(batch)

        self.evaluator.evaluation_step(pred, batch[1])
        return {"test_loss": loss}

    def test_end(self, outputs):
        test_eval = self.evaluator.evaluation_end()
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()

        logs = {"avg_test_loss": avg_loss, "avg_test_eval": test_eval}
        return {"avg_test_loss": avg_loss, "log": logs, "progress_bar": logs}

    def configure_optimizers(self):  # require
        return self.optimizer

    @pl.data_loader
    def train_dataloader(self):
        return self.dataloader["train"]

    @pl.data_loader
    def val_dataloader(self):
        return self.dataloader["val"]

    @pl.data_loader
    def test_dataloader(self):
        return self.dataloader["test"]
