import torch
import torch.nn as nn

from . import AverageMeter


def train(train_loader, net, opt, loss, lr_schd, epoch, args, device="cuda"):
    """ Training procedure. """
    # Create the directory.
    # Switch to train mode and clear the gradient.
    net.train()
    opt.zero_grad()
    # Initialize meter and list.
    batch_loss_meter = AverageMeter()

    softmax = nn.Softmax(dim=1)

    batch_index = 0
    notfirst = False
    for step, (batch_x, batch_y, batch_n) in enumerate(train_loader):
        batch_x, batch_y = batch_x.numpy(), batch_y.numpy()

        # to avoid waring
        if notfirst:
            lr_schd.step()  # Step at the beginning of the iteration.
            notfirst = True

        batch_x_tensor = torch.from_numpy(batch_x).float().to(device)
        batch_y_tensor = torch.from_numpy(batch_y).float().to(device)

        preds = net(batch_x_tensor)

        batch_loss = loss(softmax(preds), batch_y_tensor.long())
        batch_index = batch_index + 1

        # Generate the gradient and accumulate (using equivalent average loss).
        batch_loss.backward()
        opt.step()
        opt.zero_grad()

        # Record loss.
        batch_loss_meter.update(batch_loss.item())

        # Log and save intermediate images.
        if batch_index % 30 == 30 - 1:
            # print log
            print(("Epoch:{}/{}, aver_loss:{:.4f}.").format(epoch, args.max_epoch, batch_loss_meter.avg))

    return batch_loss_meter.avg


def test(test_loader, net):
    """ Test procedure. """
    softmax = nn.Softmax(dim=1)

    # Switch to evaluation mode.
    net.eval()

    preds = {}
    for _, (batch_x, batch_y, batch_n) in enumerate(test_loader):
        batch_x, batch_y = batch_x.numpy(), batch_y.numpy()
        batch_x_tensor = torch.from_numpy(batch_x).float().cuda()

        pred = net(batch_x_tensor)
        pred = softmax(pred)
        pred = pred.cpu().detach().numpy()
        preds[batch_n[0]] = {"pred": pred[0], "true": batch_y[0]}

    return preds
