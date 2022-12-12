import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import os
from collections import OrderedDict
import time
import copy

from PIL import Image

import argparse
import functions

parser = argparse.ArgumentParser(description = 'train.py')
parser.add_argument('data_dir', default="./flowers",type = str,action="store")
parser.add_argument('--arch', dest="arch", default="vgg19", type = str)
parser.add_argument('--learning_rate', dest="learning_rate", default=0.001,type=float)
parser.add_argument('--dropout', dest = "dropout", default = 0.5,type=float)
parser.add_argument('--hidden_units', type=int,dest="hidden_units",default=4096)
parser.add_argument('--epochs', dest="epochs", type=int, default=11)
parser.add_argument('--gpu', dest="gpu",action='store_true')
parser.add_argument('--save_dir',dest="save_dir",default="checkpoint.pth",type=str)
args = parser.parse_args()
data_path = args.data_dir
arch = args.arch
dropout = args.dropout
lr = args.learning_rate
hidden_layer = args.hidden_units
gpu = args.gpu
epochs = args.epochs
path = args.save_dir

print(gpu)
dataloaders,dataset_sizes,image_datasets= functions.load_data(data_path)
model,optimizer,criterion,scheduler = functions.create_model(arch,dropout,hidden_layer,lr)
from workspace_utils import active_session
with active_session():
    #training the model
    model = functions.train_model(model,criterion,optimizer,scheduler,epochs,dataloaders,dataset_sizes,gpu)

functions.save_checkpoint(image_datasets,model,dest=path,num_epochs =epochs,learn_rate=lr,hidden_layer = hidden_layer )
