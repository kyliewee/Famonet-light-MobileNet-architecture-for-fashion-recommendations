import numpy as np
import argparse
import random
from matplotlib import pyplot as plt

import torch
#from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

from config import configs
from utils import get_model, test_dataloader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mps', type = bool, default = True, help = 'use mps or not')
    parser.add_argument('-model', type = str, required = True, help = 'model')
    parser.add_argument('-worker', type = int, default = 2, help = 'the number of worker we use for dataloader')
    parser.add_argument('-batch', type = int, default = 64, help = 'batch size')
    parser.add_argument('-warmup', type = int, default = 1, help = 'learning rate warm up')
    parser.add_argument('-weightpath', type = str, required = True, help = 'path of weight file')
    args = parser.parse_args()

    model = get_model(args)

    fmst_test_loader = test_dataloader(
        configs.FMST_TRAIN_MEAN,
        configs.FMST_TRAIN_STD,
        shuffle = False,
        num_workers = args.worker,
        batch_size = args.batch
    )

    model.load_state_dict(torch.load(args.weightpath), args.mps)
    print(model)
    model.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    for number_iters, (image, label) in enumerate(fmst_test_loader):
        print("iteration: {}\ttotal {} iterations".format(number_iters + 1, len(fmst_test_loader)))
        
        mps_device = torch.device("mps")

        image = Variable(image).to(mps_device)
        label = Variable(label).to(mps_device)
        
        output = model(image)
        _, pred = output.topk(5, 1, sorted = True, largest = True)

        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()
        correct_1 += correct[:, :1].sum()
        correct_5 += correct[:, :5].sum()

        total += label.size(0)
        acc = 100. * correct / total

    print()
    print("Top 1 error: ", 1 - correct_1 / len(fmst_test_loader.dataset))
    print("Top 5 error: ", 1 - correct_5 / len(fmst_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))
    print(total)
    print(acc)