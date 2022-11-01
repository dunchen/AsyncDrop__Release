
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
mp.set_start_method('spawn',force=True)
import copy
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms

from asyncdrop_train import test,train_update_gradient_only

from asyncdrop_model import resnet34

from asyncdrop_utils import cifar_noniid
import os

import random
import numpy as np



# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model_type', type=str, default='resnet')
parser.add_argument('--dataset', type=str, default='CIFAR100',
                    help='datset')
parser.add_argument('--baseline', action='store_true', default=False,
                    help='baseline training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 65)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log_epoch', type=int, default=2, metavar='N',
                    help='number of epochs to log (default: 10)')

parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr_type', type=str, default='step_wise',
                    help='learning rate type (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')


parser.add_argument('--seed', type=int, default=3, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--non_iid', action='store_true', default=True,
                    help='non iid')

parser.add_argument('--random_mask', action='store_true', default=False,
                    help='random mask')

parser.add_argument('--num-processes', type=int, default=8,
                    help='how many training processes to use (default: 2)')
parser.add_argument('--num_total_users', type=int, default=104,
                    help='how many users to use (default: 2)')
parser.add_argument('--num-devices', type=int, default=4, 
                    help='how many device')
parser.add_argument('--hidden_dim_prob', type=float, default=0.25,
                    help='hidden_dim_prob')

parser.add_argument('--local_iterations', type=int, default=50,
                    help='local_iteration')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='alpha')
parser.add_argument('--smart_long_memory', action='store_true', default=True,
                    help='smart_long_memory')

parser.add_argument('--descending', action='store_true', default=True,
                    help='descending')


parser.add_argument('--delay', type=float, default=4,
                            help='alpha')

parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')

parser.add_argument('--cuda_id',type=str,default='0,1,2,3')

if __name__ == '__main__':


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_id
    assert args.log_epoch==2
    assert args.dataset=='CIFAR100'
    assert args.num_processes==8
    assert args.num_devices==4
    assert args.epochs==40
    assert args.batch_size==128
    assert args.non_iid
    assert args.alpha==0.5
    assert args.lr==0.01
    assert args.local_iterations==50
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    #use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cpu")
    
    kwargs = {'batch_size': args.batch_size,
        'shuffle': True}

    if args.dataset=="CIFAR100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        trainset = datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)


        testset = datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        output_class=100
    
    if args.non_iid:
        if args.dataset=="CIFAR100" or args.dataset=="CIFAR10":
            user_groups = cifar_noniid(trainset, args.num_processes)
        else:
            raise ValueError
        num_user_per_process=int(args.num_total_users/args.num_processes)
        user_groups_per_process=[]
        for i in range(args.num_processes):
            temp=[]
            for j in range(num_user_per_process):
                temp.append(np.random.choice(user_groups[i],size=int(len(user_groups[i])/2),replace=False))
            user_groups_per_process.append(temp)

    

    torch.manual_seed(args.seed)

    if args.model_type=='resnet':
        assert args.dataset=="CIFAR100" or args.dataset=="CIFAR10"
        model = resnet34(output_classes=output_class)
    else:
        raise ValueError

    start_epoch=0
    global_lr =torch.tensor([args.lr])
    global_lr.share_memory_()
    global_iter =torch.tensor([0])
    global_iter.share_memory_()
    model_bac=copy.deepcopy(model)
    model_bac.share_memory()
    model.share_memory() # gradients are allocated lazily, so they are not shared here



    processes = []


    for rank in range(args.num_processes):
        device=torch.device(rank%args.num_devices)

        if args.non_iid:
            p = mp.Process(target=train_update_gradient_only, args=(rank, args, model, model_bac, device, global_lr, global_iter,
                                               trainset,testset, start_epoch, kwargs, user_groups_per_process[rank]))
        else:
            p = mp.Process(target=train_update_gradient_only, args=(rank, args, model, model_bac, device, global_lr, global_iter,
                                               trainset,testset, start_epoch, kwargs))
        
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)



    for p in processes:
        p.join()


    # Once training is complete, we can test the model
    device=0
    test(args, model.to(device), device, testset, kwargs)
