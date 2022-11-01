#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    #num_shards, num_imgs = 200, 250
    #For Cifar10
    num_shards, num_imgs = num_users*2, int(50000/(num_users*2))
    #for cifar100
    #num_shards,num_imgs = num_users, int(50000/num_users)
    shard_per_user = int(num_shards/num_users)

    idx_shard = [i for i in range(num_shards)]
    # dict_users = {i: np.array([]) for i in range(num_users)}
    dict_users = [np.array([]) for i in range(num_users)]

    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.targets.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    # To deal with data num not divisiable by number of users
    labels = labels[:idxs.shape[0]]

    #import pdb
    #pdb.set_trace()
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

