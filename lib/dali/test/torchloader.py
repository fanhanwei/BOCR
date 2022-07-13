import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

import os
from timeit import default_timer as timer

def get_dataset(dset_name, batch_size, n_worker, data_root=image_dir):
    print('=> Preparing data..')
    if dset_name == 'imagenet':
        # get dir
        traindir = data_root

        # preprocessing
        input_size = 224
        imagenet_tran_train = [
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose(imagenet_tran_train)),
            batch_size=batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=True, sampler=None)
        val_loader = None
        n_class = 1000

    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class
testbatch=256
train_loader, val_loader, n_class = get_dataset('imagenet', testbatch, 6)

for batch_idx, (inputs, targets) in enumerate(train_loader):
    if batch_idx == 5:
        t_start = timer()
    print(batch_idx)
    inputs, targets = inputs.cuda(), targets.cuda()
    if batch_idx == 24:
        t = timer() - t_start
        print("Speed: {} imgs/s".format((20 * testbatch)/t))
        break