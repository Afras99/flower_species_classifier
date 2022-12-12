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
import json
from PIL import Image


import argparse
import functions

parser = argparse.ArgumentParser(description = 'predict.py')
parser.add_argument('img',default = 'flowers/test/1/image_06743.jpg',type=str)
parser.add_argument('checkpoint',default = 'checkpoint.pth',type = str)
parser.add_argument('--top_k',default=5,dest='top_k',type=int)
parser.add_argument('--category_names',dest='category_names',default = 'cat_to_name.json',type=str)
parser.add_argument('--gpu', dest="gpu",action='store_true')

args = parser.parse_args()
image = args.img
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu
checkpoint_path = args.checkpoint

#dataloaders,_,_ = functions.load_data()

model = functions.load_model(checkpoint_path)

cat_to_name = functions.mapper(category_names)

probs, classes = functions.predict(image, model,gpu,top_k)
names = [cat_to_name[x] for x in classes]

for i in range(len(names)):
    print("{} has a probability of {:.4f}".format(names[i], probs[i]))