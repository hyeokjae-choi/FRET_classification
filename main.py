import argparse
import inspect

from torch.utils import data as torch_data
from torch import optim

from fret_pytorch import network as fret_network
from fret_pytorch import dataloader as fret_dl
from fret_pytorch import utils as fret_utils


def get_args():
    parser = argparse.ArgumentParser(description="FRET Pytorch from hyeokjae-choi")

    # dataloader
    # train
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--train_epoch", type=int, default=1000)

    # hw
    # -> Not PL upgraded... gpu_idx '0,1' -> use 0,1 multigpu. if '' use cpu
    parser.add_argument("--gpu_idx", type=str, default="0")
    parser.add_argument("--num_worker", type=int, default=0)

    # network
    parser.add_argument("--network", type=str, default="std3d")

    # data
    parser.add_argument("--ds_name", type=str)

    # optimizer settings - current isn't supported
    parser.add_argument("--loss", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_stepsize", type=int)
    parser.add_argument("--lr_gamma", type=float)
    parser.add_argument("--clip_grad", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    # logging
    parser.add_argument("--project_name", type=str, default="default")
    parser.add_argument("--train_log_freq", type=int, default=100)
    parser.add_argument("--val_log_freq_epoch", type=int, default=5)

    _args, network_option = parser.parse_known_args()

    return vars(_args), network_option


class MainLoader:
    def __init__(
        self,
        # train
        train_batch_size,
        test_batch_size,
        train_epoch,
        # hw
        gpu_idx,
        num_worker,  # gpu_idx '0,1' -> use 0,1 multi gpu. if '' use cpu
        # network
        network,
        # data
        ds_name,
        # optimizer settings
        loss,
        lr,
        lr_stepsize,
        lr_gamma,
        clip_grad,
        momentum,
        weight_decay,
        # logging,
        project_name,
        train_log_freq,
        val_log_freq_epoch,
        # network_option
        network_option,
    ):
        self.hw_option = self.__hw_intp(gpu_idx, num_worker)
        self.data_option = {"ds_name": ds_name}
        self.network_option = self.__network_intp(network, network_option)

        # use hw, data, network opt
        self.train_option = self.__train_intp(train_batch_size, train_epoch)
        self.val_option = self.__val_intp(test_batch_size)
        self.test_option = self.__test_intp(test_batch_size)
        self.opt_option = self.__opt_intp(loss, lr, lr_stepsize, lr_gamma, clip_grad, momentum, weight_decay)

        self.log_option = self.__log_intp(project_name, train_log_freq, val_log_freq_epoch)

    @staticmethod
    def __hw_intp(gpu_idx, num_workers):
        gpu_idx = [int(gpu) for gpu in gpu_idx.split(",") if gpu != ""]
        return {
            "gpu_idx": gpu_idx if len(gpu_idx) > 0 else None,
            "num_workers": num_workers,
            "gpu_on": True if len(gpu_idx) > 0 else False,
        }

    @staticmethod
    def __network_intp(network, network_option):
        def check_option(given_keys, target_keys):
            for given in given_keys:
                if given not in target_keys:
                    return False
            return True

        assert len(network_option) % 2 == 0, "wrong given network option"

        network_option_dict = {}

        for idx in range(0, len(network_option), 2):
            network_option_dict[network_option[idx]] = network_option[idx + 1]

        if len(network_option) > 0:
            assert check_option(
                list(network_option_dict.keys()), list(inspect.getfullargspec(fret_network.network_dict[network])[0]),
            )

        network = fret_network.get_network(network, network_option_dict)

        return {"network": network, "network_option": network_option_dict}

    def __data_intp(self, train_batch_size, phase):
        data = fret_dl.fret_dataloader.DataManager(ds_name=self.data_option["ds_name"], phase=phase,)
        dataloader = torch_data.DataLoader(
            data,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=self.hw_option["num_workers"],
            pin_memory=self.hw_option["gpu_on"],
        )
        return {"dataloader": dataloader}

    def __train_intp(self, batch_size, train_epoch):
        option = {"epoch": train_epoch}

        dl = self.__data_intp(batch_size, phase="train")
        option.update(dl)
        return option

    def __val_intp(self, batch_size):
        option = {}
        dl = self.__data_intp(batch_size, phase="val")
        option.update(dl)
        return option

    def __test_intp(self, batch_size):
        option = {}
        dl = self.__data_intp(batch_size, phase="test")
        option.update(dl)
        return option

    def __opt_intp(self, loss, lr, lr_stepsize, lr_gamma, clip_grad, momentum, weight_decay):
        if loss == "Adadelta" or loss == "Adam":
            opt = optim.__dict__[loss](self.network_option["network"].parameters(), lr=lr, weight_decay=weight_decay)
        else:
            opt = optim.__dict__[loss](
                self.network_option["network"].trainable_param, lr=lr, weight_decay=weight_decay, momentum=momentum
            )
        return {
            "loss": loss,
            "lr": lr,
            "lr_stepsize": lr_stepsize,
            "clip_grad": clip_grad,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "opt": opt,
        }

    @staticmethod
    def __log_intp(project_name, train_log_freq, val_log_freq_epoch):
        return {
            "project_name": project_name,
            "train_log_freq": train_log_freq,
            "val_log_freq_epoch": val_log_freq_epoch,
        }

    def run(self):
        from pytorch_lightning import Trainer
        from pytorch_lightning.logging import TensorBoardLogger

        network = self.network_option["network"]
        optimizer = self.opt_option["opt"]
        dataloader = {
            "train": self.train_option["dataloader"],
            "val": self.val_option["dataloader"],
            "test": self.test_option["dataloader"],
        }

        pl = fret_utils.main_pl.PLMain(
            network=network,
            dataloader=dataloader,
            optimizer=optimizer,
            train_log_interval=self.log_option["train_log_freq"],
        )

        trainer = Trainer(
            logger=TensorBoardLogger(save_dir="../trained", name=self.log_option["project_name"]),
            default_save_path=self.data_option["ds_name"],
            gpus=self.hw_option["gpu_idx"],
            check_val_every_n_epoch=self.log_option["val_log_freq_epoch"],
            max_epochs=self.train_option["epoch"],
            min_epochs=self.train_option["epoch"],
            log_save_interval=1,
            row_log_interval=1,
        )
        trainer.fit(pl)
        trainer.test()


if __name__ == "__main__":
    args, args_network = get_args()
    ml = MainLoader(**args, network_option=args_network)
    ml.run()
