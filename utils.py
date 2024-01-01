import sys
import numpy

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms

def get_model(args, use_mps = True):

    if args.model == 'famonet':
        from models.famonet import famonet
        model = famonet()
    else:
        print('Error: can not find the model')
        sys.exit()
    
    if use_mps:
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
        model = model.to(mps_device)

    return model

def data_mean_std(fmst_data):

    data = numpy.dstack([fmst_data[i][0] for i in range(len(fmst_data))])
    mean = numpy.mean(data)
    std = numpy.std(data)
    
    return mean, std

def train_dataloader(mean, std, shuffle = True, num_workers = 2, batch_size = 64):
    """
        mean: the mean of FashionMNST training dataset
        std: the std of FashionMNST training dataset
        batch_size: the batchsize of dataset
        num_workers: the number of subprocesses to use for data loading
    """
     
    mean = (0.2861,)
    std = (0.3528,)

    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    fmst_train_data = torchvision.datasets.FashionMNIST(
        root = './dataset', 
        train = True, 
        transform = transform_train,
        download = True
    )
    fmst_train_loader = DataLoader(
        fmst_train_data, 
        shuffle = shuffle,
        num_workers = num_workers,
        batch_size = batch_size
    )

    return fmst_train_loader

def test_dataloader(mean, std, shuffle = False, num_workers = 2, batch_size = 64):

    mean = (0.2861,)
    std = (0.3528,)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    fmst_test__data = torchvision.datasets.FashionMNIST(
        root = './dataset', 
        train = False, 
        download = True, 
        transform = transform_test
    )
    fmst_test_loader = DataLoader(
        fmst_test__data, 
        shuffle = shuffle,
        num_workers = num_workers,
        batch_size = batch_size
    )

    return fmst_test_loader

class LRS(_LRScheduler):

    def __init__(self, optimizer, total_iters, last_epoch = -1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
    def get_lr(self):

        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]