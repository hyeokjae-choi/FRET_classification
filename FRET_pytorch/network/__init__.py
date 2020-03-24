from .network import STN3d, Feats_STN3d

network_dict = {
    "std3d": STN3d,
    "feats_stn": Feats_STN3d,
}


def get_network(network_name, network_opt):
    return network_dict[network_name](**network_opt)
