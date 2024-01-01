import os
import sys
import argparse

import numpy as np
import random

from config import configs
from utils import get_model, train_dataloader, test_dataloader, LRS

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter
import torch.profiler


seed = 430
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train(epoch):
    
    model.train()
    for index, (image, label) in enumerate(fmst_train_loader):
        if epoch <= args.warmup:
            warmup_scheduler.step()

        image = Variable(image)
        label = Variable(label)

        label = label.to(mps_device)
        image = image.to(mps_device)
        
        optimizer.zero_grad()

        output = model(image)
        loss = loss_function(output, label)
        loss.backward()

        optimizer.step()

        #profiler.step()

        number_iters = (epoch - 1) * len(fmst_train_loader) + index + 1
        last_layer = list(model.children())[-1]
        for name, param in last_layer.named_parameters():
            if 'weight' in name:
                if param.grad is not None:
                    writer.add_scalar('Gradients/weight_squared_norm', param.grad.norm(), number_iters)
            if 'bias' in name:
                if param.grad is not None:
                    writer.add_scalar('Gradients/bias_squared_norm', param.grad.norm(), number_iters)

        print('Epoch: {epoch}({trained_samples}/{total_samples})\tLoss: {:0.4f}\tLR: {:0.5f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch = epoch,
            trained_samples = len(image) + index * args.batch,
            total_samples = len(fmst_train_loader.dataset)
        ))

        # write loss per iteration
        writer.add_scalar('Train/Loss', loss.item(), number_iters)

    for name, param in model.named_parameters():
        layer, attribute = os.path.splitext(name)
        attribute = attribute[1:]
        writer.add_histogram("{}/{}".format(layer, attribute), param, epoch)

def eval(epoch):

    model.eval()

    loss_test = 0.0
    correct = 0.0

    for (image, label) in fmst_test_loader:
        image = Variable(image)
        label = Variable(label)
        
        image = image.to(mps_device)
        label = label.to(mps_device)

        output = model(image)
        loss = loss_function(output, label)
        loss_test += loss.item()

        _, pred = output.max(1)
        correct += pred.eq(label).sum()

    print('Test set: Average Loss: {:.4f}, Accuracy: {:.4f}'.format
          (loss_test / len(fmst_test_loader.dataset), correct.float() / len(fmst_test_loader.dataset)))

    # write on tensorboard
    writer.add_scalar('Test/Average Loss', loss_test / len(fmst_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(fmst_test_loader.dataset), epoch)

    result = correct.float() / len(fmst_test_loader.dataset)

    return result

if __name__ == '__main__':

    seed_everything(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('-mps', type = bool, default = True, help = 'use mps or not')
    parser.add_argument('-model', type = str, required = True, help = 'model')
    parser.add_argument('-worker', type = int, default = 2, help = 'the number of worker we use for dataloader')
    parser.add_argument('-batch', type = int, default = 64, help = 'batch size')
    parser.add_argument('-warmup', type = int, default = 1, help = 'learning rate warm up')
    args = parser.parse_args()

    model = get_model(args, use_mps=args.mps)

    # data preprocess
    fmst_train_loader = train_dataloader(
        configs.FMST_TRAIN_MEAN,
        configs.FMST_TRAIN_STD,
        shuffle = True,
        num_workers = args.worker,
        batch_size = args.batch
    )
    fmst_test_loader = test_dataloader(
        configs.FMST_TRAIN_MEAN,
        configs.FMST_TRAIN_STD,
        shuffle = False,
        num_workers = args.worker,
        batch_size = args.batch
    )
    
    # set hyperparameters
    loss_function = nn.CrossEntropyLoss()
    params = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = optim.SGD(params = params, lr = 0.1, momentum = 0.9, weight_decay = 5e-4)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer = optimizer, milestones = configs.MILESTONES, gamma = 0.2)
    iter_per_epoch = len(fmst_train_loader)
    warmup_scheduler = LRS(optimizer = optimizer, total_iters = iter_per_epoch * args.warmup)

    checkpoint_dir = os.path.join(configs.CHECKPOINT_DIR, args.model, configs.CURR_TIME)

    # set tensorboard
    if not os.path.exists(configs.LOG_DIR):
        os.mkdir(configs.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(configs.LOG_DIR, args.model, configs.CURR_TIME))
    mps_device = torch.device("mps")
    input_tensor = torch.Tensor(32, 1, 28, 28).to(mps_device)
    writer.add_graph(model, Variable(input_tensor, requires_grad=True))

    # create checkpoint
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_name = '{model}-{epoch}-{type}.pth'
    checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_name)

    accuracy_best = 0.0
    for epoch in range(1, configs.EPOCH):
        if epoch > args.warmup:
            lr_scheduler.step(epoch)
        train(epoch)
        accuracy = eval(epoch)

        # save weight
        if epoch > configs.MILESTONES[1] and accuracy_best < accuracy:
            torch.save(model.state_dict(), checkpoint_dir.format(model = args.model, epoch = epoch, type = 'best'))
            accuracy_best = accuracy
            continue
        if not epoch % configs.EPOCH_SAVE:
            torch.save(model.state_dict(), checkpoint_dir.format(model = args.model, epoch = epoch, type = 'regular'))

    writer.close()