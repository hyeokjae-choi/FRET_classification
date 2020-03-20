import os
import os.path as osp
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

# customs
import FRET_pytorch.model as OneDCNN
from FRET_pytorch.data.data_loader import DataManager
from FRET_pytorch.model.custom import Simple1DCNN, weights_init
from FRET_pytorch.main_utils import Logger, load_checkpoint
from FRET_pytorch.main_utils import trainer
from FRET_pytorch.eval.evaluator import evaluation
from FRET_pytorch.eval.utils import EvalSaveManager


def get_args():
    parser = argparse.ArgumentParser(description="FRET trace classification by Pytorch from HYEOKJAE")

    # 1. Actions.
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--test", action="store_true", help="Test the model.")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model.")
    # 2. DataLoader
    parser.add_argument("--ds_name", type=str, help="Name of the dataset.")
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--augment", action="store_true", help="Whether to augment data")
    # 3. Counts.
    parser.add_argument("--max_epoch", default=100, type=int, help="Total epochs.")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Training batch size.")
    # 4. Files and folders.
    parser.add_argument("--out_dir", default="./output", help="Output folder.")
    parser.add_argument("--eval_dir", type=str, help="Directory with evaluated results.")
    parser.add_argument("--checkpoint", default="", help="Resume the checkpoint.")
    # 5. Others.
    parser.add_argument("--cpu", action="store_true", help="Enable CPU mode.")
    parser.add_argument("--gpu_idx", default="0", type=str, help="Assign GPU index.")
    # 6. Network and optimizer settings.
    parser.add_argument("--network", choices=OneDCNN.supported_model, help="network type")
    parser.add_argument("--lr", default=1e-3, type=float, help="Initial learning rate.")
    parser.add_argument("--lr_stepsize", default=1e4, type=int, help="Learning rate step size.")
    parser.add_argument("--lr_gamma", default=1.0, type=float, help="Learning rate decay (gamma).")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum.")
    parser.add_argument("--weight_decay", default=2e-4, type=float, help="Weight decay.")

    args = parser.parse_args()
    return args


def _main(args):
    # Config
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    device = torch.device("cpu" if args.cpu else "cuda")

    ################################################
    # I. Miscellaneous.
    ################################################
    # Create the output directory.
    current_dir = osp.abspath(osp.dirname(__file__))
    output_dir = osp.join(current_dir, args.out_dir)
    if not osp.isdir(output_dir):
        os.makedirs(output_dir)

    # Set logger.
    now_str = datetime.now().strftime("%y%m%d-%H%M%S")
    log = Logger(osp.join(output_dir, "log-{}.txt".format(now_str)))
    sys.stdout = log  # Overwrite the standard output.
    # Print argument.
    for arg in vars(args):
        print(arg, getattr(args, arg))

    ################################################
    # II. Datasets.
    ################################################
    # Load dataset
    dataset_args = dict(ds_name=args.ds_name, augment=args.augment,)
    train_dataset = DataManager(div="train", **dataset_args)
    valid_dataset = DataManager(div="valid", **dataset_args)
    test_dataset = DataManager(div="test", **dataset_args)

    # Call dataloader
    train_loader = DataLoader(train_dataset, args.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, 1, shuffle=False)
    test_loader = DataLoader(test_dataset, 1, shuffle=False)

    eval_saver = EvalSaveManager()

    ################################################
    # III. Network and optimizer.
    ################################################
    # Create the network in GPU.
    net = OneDCNN.get_network(args.network, num_classes=args.num_classes, length=667)
    net.apply(weights_init)
    net.to(device)

    # Optimizer settings.
    parameter = filter(lambda p: p.requires_grad, net.parameters())

    # Create optimizer.
    opt = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler.
    lr_schd = lr_scheduler.StepLR(opt, step_size=args.lr_stepsize, gamma=args.lr_gamma)

    # Assign loss type
    loss = nn.CrossEntropyLoss()

    ################################################
    # IV. Pre-trained parameters.
    ################################################
    # Resume the checkpoint.
    if args.checkpoint:
        load_checkpoint(net, opt, args.checkpoint)

    ################################################
    # V. Training / testing.
    ################################################
    if args.train:
        start = time.time()
        # Train.
        for epoch in range(args.max_epoch):
            print("epoch: {}".format(epoch))
            # Epoch training and test.
            train_epoch_loss = trainer.train(train_loader, net, opt, loss, lr_schd, epoch, args=args, device=device)

            valid_pred = trainer.test(valid_loader, net)

            eval_score = evaluation(valid_pred, args.num_classes)

            print(f"Epoch:{epoch}: accuracy:{eval_score:.3f}.")

            eval_saver.saveEval(net, opt, epoch, eval_score, args.out_dir)

            # Write log.
            log.flush()

        # training finished : perform test
        best_epoch = eval_saver.getBestEpoch()
        best_score = eval_saver.getBestScore()

        print("\nTraining Finished")
        print("Training time :", time.time() - start)
        print(f"BEST EPOCH: {best_epoch}, SCORE: {best_score}")

        load_checkpoint(net, opt, osp.join(output_dir, "best_model-checkpoint.pt"))

        test_pred = trainer.test(test_loader, net)
        eval_score = evaluation(test_pred, args.num_classes)
        print(f"\nTEST SCORE: {eval_score}")

    if args.test:
        load_checkpoint(net, opt, osp.join(args.out_dir, "best_model-checkpoint.pt"))

        test_pred = trainer.test(test_loader, net)
        eval_score = evaluation(test_pred, args.num_classes)
        print(f"\nTEST SCORE: {eval_score}")


if __name__ == "__main__":
    args = get_args()

    _main(args)
