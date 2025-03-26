import geoopt
import torch
import torch.nn as nn
import manifolds
from networks.VNet_VAE import VNet, VNet_H


def net_factory(params, net_type="vnet", in_chns=1, class_num=4, mode = "train"):
    hidden_dim = params.hidden_dim

    if net_type == "vnet" and mode == "train":
        net = VNet(params, hidden_dim, n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()

    elif net_type == "vnet" and mode == "train_dp":
        net = VNet(params, hidden_dim, n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()

    elif net_type == "vnet" and mode == "test":
        net = VNet(params, hidden_dim, n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()

    return net


def net_factory_hyperbolic(params, net_type="vnet", in_chns=1, class_num=4, mode = "train"):
    hidden_dim = params.hidden_dim

    if net_type == "vnet" and mode == "train":
        net = VNet_H(params, hidden_dim, n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()

    elif net_type == "vnet" and mode == "train_dp":
        net = VNet_H(params, hidden_dim, n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()

    elif net_type == "vnet" and mode == "test":
        net = VNet_H(params, hidden_dim, n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()

    return net
