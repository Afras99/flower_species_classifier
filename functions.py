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




def load_data(data_path = "flowers"):
    
    data_dir = data_path
    data_transforms = { 'train': transforms.Compose(
                            [transforms.RandomRotation(45),
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],
                                                  [0.229,0.224,0.225])
                            ]),
                   'valid': transforms.Compose(
                            [transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],
                                                  [0.229,0.224,0.225])
                            ]),
                   'test': transforms.Compose(
                           [transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],
                                                  [0.229,0.224,0.225])
                            ]),
                  }
    # TODO: Load the datasets with ImageFolder
    image_datasets ={x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                     for x in ['train', 'valid', 'test']}


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders =  {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,shuffle=True,)
                   for x in ['train', 'valid', 'test']} 
    dataset_sizes = {x: len(image_datasets[x]) 
                              for x in ['train', 'valid', 'test']}
    return dataloaders,dataset_sizes,image_datasets

def create_model(arch='vgg19',dropout=0.5, hidden_layer = 4096,learn_rate = 0.001):
    if arch=='vgg19':
        model = torchvision.models.vgg19(pretrained=True)
    else:
        model = torchvision.models.vgg16(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                          ('fc1',nn.Linear(25088,hidden_layer)),
                          ('relu',nn.ReLU()),
                          ('dropout', nn.Dropout(dropout)),
                          ('fc2',nn.Linear(hidden_layer,102)),
                          ('output',nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    #hyperparameters
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    return model,optimizer,criterion,scheduler 


def train_model(model,criterion,optimizer,scheduler,epochs,dataloaders,dataset_sizes,gpu):
    #Check cuda
    if torch.cuda.is_available() and gpu==True:
        device= torch.device('cuda')
    else:
        device= torch.device('cpu')
    model = model.to(device)
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        for phase in ['train','valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                 # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward propogation only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_checkpoint(image_datasets,model,arch='vgg19',dest='checkpoint.pth',num_epochs = 10,learn_rate=0.001,hidden_layer = 4096):
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()
    checkpoint = {'arch': arch,
                 'epochs': num_epochs,
                 'hidden_layer':hidden_layer, 
                 'lr':learn_rate,
                 'model_state_dict': model.state_dict(), 
                  #'optimizer_state_dict':optimizer.state_dict(),
                 'class_to_idx': model.class_to_idx}
    torch.save(checkpoint,dest)
    
    
def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path)
    model,optimizer,criterion,scheduler = create_model(arch =chpt['arch'],dropout=0.5, 
                                          hidden_layer = chpt['hidden_layer'],learn_rate = chpt['lr'])
    model.class_to_idx = chpt['class_to_idx']
    #optimizer.load_state_dict(chpt['optimizer_state_dict'])
    epochs = chpt['epochs']
    model.load_state_dict(chpt['model_state_dict'])
    return model  

def mapper(category_names='cat_to_name.json'):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img.load()
    
    if img.size[0] > img.size[1]: #if width <height
        img.thumbnail((10000, 256)) #set height to 256
    else:
        img.thumbnail((256, 10000)) #set width to 256
        
    # cropping t0 224*224 size image
    size = img.size
    img = img.crop((size[0]//2 -(224/2),
                     size[1]//2 - (224/2),
                     size[0]//2 +(224/2),
                     size[1]//2 + (224/2) 
                    ))
    #Normalizing
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    img = (img - mean)/std
    
    img = img.transpose((2, 0, 1))
    return img

def predict(image_path, model,gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    if torch.cuda.is_available() and gpu==True:
        device= torch.device('cuda')
    else:
        device= torch.device('cpu')
    with torch.no_grad():
        img = process_image(image_path)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        img = img.unsqueeze(0)
        probs = model.forward(img)
        top_prob, top_labels = torch.topk(probs, topk)
        top_prob = top_prob.exp()
        top_prob_array = top_prob.data.numpy()[0]
    
         
        inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    
        top_labels_data = top_labels.data.numpy()
        top_labels_list = top_labels_data[0].tolist()  
    
        top_classes = [inv_class_to_idx[x] for x in top_labels_list]
    
        return top_prob_array, top_classes