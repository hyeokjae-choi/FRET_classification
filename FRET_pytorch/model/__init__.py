import torch.nn as nn

from .custom import Simple1DCNN, weights_init
from .pointnet2 import STN3d, Feats_STN3d

supported_model = [
    "custom",
    "stn3d",
    "feats_stn3d",
]


def get_network(network_name, num_classes=2, length=667, device="cuda:0"):

    net = None
    if network_name == "custom":
        net = nn.DataParallel(Simple1DCNN(num_classes=num_classes))
        net.apply(weights_init)
        net.to(device)
    if network_name == "stn3d":
        net = nn.DataParallel(STN3d(num_classes=num_classes, length=length))
        net.apply(weights_init)
        net.to(device)
    if network_name == "feats_stn3d":
        net = nn.DataParallel(Feats_STN3d(num_classes=num_classes, length=length))
        net.apply(weights_init)
        net.to(device)

    return net
