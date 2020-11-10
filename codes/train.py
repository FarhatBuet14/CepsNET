# python train.py --inp mfcc_13
from __future__ import print_function, division, absolute_import

import torch
torch.cuda.empty_cache()

from BalancedDataGenerator import BalancedAudioDataGenerator
from dataLoader import reshape_folds
import os,time
import numpy as np
np.random.seed(1)
import math
from datetime import datetime
import argparse
import utils
from utils import Scoring
import h5py
import pandas as pd

class paramClass():
    def __init__(self):
        self.epochs = 2
        self.batch_size = 256
        self.inp = 'log-Filterbank-DCT-26-13'
        self.lr = 0.001
        self.base_lr = 2.455e-5
        self.max_lr = 1.767e-3
        self.step_size = 4
        self.log_dir = '../logs/'
param = paramClass()

# --- Parse the arguements

parser = argparse.ArgumentParser(description='Necessary variables')
parser.add_argument("--ep",type=int, help = "Number of epochs")
parser.add_argument("--inp",type=str, help = "Input Feature")
parser.add_argument("--batch",type=int, help = "Batch Size")
parser.add_argument("--base_lr",type=str, help = "Minimum Learning Rate")
parser.add_argument("--max_lr",type=str, help = "Maximum Learning Rate")
parser.add_argument("--step_size",type=str, help = "Step Size Cyclic Learning Rate")
arguments = parser.parse_args()

if(arguments.ep): param.epochs = arguments.ep
if(arguments.batch): param.batch_size = arguments.batch
if(arguments.lr): param.lr = arguments.lr
if(arguments.base_lr): param.base_lr = arguments.base_lr
if(arguments.max_lr): param.max_lr = arguments.max_lr
if(arguments.step_size): param.step_size = arguments.step_size

df = pd.read_excel (r'../logs/pretrained.xlsx')
features = df["feature"].iloc[:]
index = [i for i, feat in enumerate(features) if(feat==arguments.inp)][0]
param.inp = df["inp"][index]

# --- Import train-validation Data 

print("Loading data..")
data_file = "../data/feature/train_val_npzs/" + param.inp + "_train_val_data.npz"
data = np.load(data_file)

x_train = data['x_train_mfcc']
x_val = data['x_val_mfcc']
y_train = data['y_train']
y_val = data['y_val']
val_parts = data['val_parts']
val_wav_files = data['val_wav_files']
data = h5py.File('../data/feature/folds/fold_0.mat', 'r')
train_wav_files = [chr(stp) for stp in data['train_files'][0]]

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2], 1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2], 1)

#--- Reshape the folds
_, [y_train, y_val] = reshape_folds([], [y_train, y_val])

from keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes = 2)
y_val = to_categorical(y_val, num_classes = 2)

# Meta lables for both Class and Domain balance training
from collections import Counter
domains = list(Counter(train_wav_files).keys())
domainClass = [(cls,dfc) for cls in range(2) for dfc in domains]
meta_labels = [domainClass.index((cl,df)) for (cl,df) in zip(np.argmax(y_train,1),train_wav_files)]

# --- Import necessary libraries for training purpose

import torch.optim as optim
import torch
from torch.autograd import Variable
import time
import torch
import torch.nn as nn
from torchsummary import summary
from torch.backends import cudnn

# --- Assign necessary values

lr = param.lr

# --- Batch Size limiter

if(param.batch_size > max(y_train.shape)):
    print("Batch size if given greater than train files size. limiting batch size")
    param.batch_size = max(y_train.shape)

# --- Balance the training data

datagen = BalancedAudioDataGenerator(shift=.1)
flow = datagen.flow(x_train, y_train, meta_label = meta_labels,
                    batch_size = param.batch_size, shuffle = True, seed = 1)

# --- Load the model and optimizers and loss function

from model import resnet_extractor, abn_classifier
nodes = [16, 32, 64, 128]
num_layers = 2
model_fe = resnet_extractor(1, None, param.inp, nodes, num_layers).cuda()
model_abn = abn_classifier(model_fe.out_features, 2, dropout = None).cuda()
    
from torchsummary import summary
summary(model_fe.cuda(),(np.moveaxis(x_train[0], 2, 0).shape))
summary(model_abn.cuda(),(1,model_fe.out_features))

optimizer_fe = optim.SGD(model_fe.parameters(), lr, 0.9, weight_decay=1e-1, nesterov=True)
optimizer_abn = optim.SGD(model_abn.parameters(), lr, 0.9, weight_decay=1e-1, nesterov=True)

cudnn.benchmark = True

scheduler_fe = optim.lr_scheduler.CyclicLR(optimizer_fe, base_lr = param.base_lr, 
            max_lr = param.max_lr, step_size_up= param.step_size * flow.steps_per_epoch,
            mode='triangular', gamma=0.99980)
scheduler_abn = optim.lr_scheduler.CyclicLR(optimizer_abn, base_lr = param.base_lr, 
        max_lr = param.max_lr, step_size_up= param.step_size * flow.steps_per_epoch,
        mode='triangular', gamma=0.99980)

loss_fn = nn.CrossEntropyLoss()

# --- Save the log

log_dir = os.path.join(param.log_dir, param.inp)
if(not os.path.isdir(log_dir)):os.mkdir(log_dir)
log_dir = os.path.join(log_dir, str(datetime.now())[:-10].replace(":", "_"))
if(not os.path.isdir(log_dir)):os.mkdir(log_dir)

import CSVLogger as log
logger = log.CSVLogger(os.path.join(log_dir,'training.csv')) 
logger_iter = log.CSVLogger(os.path.join(log_dir,'training_iters.csv'))

log_dir = os.path.join(log_dir, "checkpoints")
if(not os.path.isdir(log_dir)):os.mkdir(log_dir)

# --- Training begins..

ep_st = 0
num_iter = 0

logger.on_train_begin()
logger_iter.on_train_begin()
for e in range(ep_st, param.epochs):
    
    ep_lr = (scheduler_fe.get_last_lr()[0])

    print(f"EPOCH ---- ( {e+1} )---- Learning Rate ---- ({ep_lr}) ---- {datetime.now()}")
    
    model_fe.train()
    model_abn.train()
    epoch_loss = 0

    for i in range(flow.steps_per_epoch + 1):
        
        optimizer_fe.zero_grad()
        optimizer_abn.zero_grad()

        x,y= flow.next()

        x,y= torch.from_numpy(x),torch.from_numpy(y)
        x,y= Variable(x),Variable(y)
        x = x.type(torch.FloatTensor).cuda()
        x = x.reshape(x.shape[0],1,x.shape[1],x.shape[2])
        y = torch.tensor(np.array(y).reshape(-1, 2)).cuda().float()

        features = model_fe(x)
        cls = model_abn(features)
        
        loss = loss_fn(cls, torch.argmax(y,dim=1))
        epoch_loss = epoch_loss + loss

        loss.backward()

        batch_lr = (scheduler_fe.get_last_lr()[0])
        logs = {'epoch' : num_iter + 1,
                'loss' : loss.item(),
                'Lr' : batch_lr}
        logger_iter.on_epoch_end(num_iter, logs)
        num_iter += 1

        optimizer_fe.step()
        optimizer_abn.step()
        
        scheduler_fe.step()
        scheduler_abn.step()

    epoch_loss = epoch_loss/int(flow.steps_per_epoch)
    print("Training loss - ", str(epoch_loss.item()))
    
    # --- Validation starts..

    model_fe.eval()
    model_abn.eval()

    with torch.no_grad():
        cls_pred = None
        cls_val = None
        epoch_val_loss = 0
        s = 0
        for part in val_parts:
            x,y = torch.from_numpy(x_val[s:s+part]),torch.from_numpy(y_val[s:s+part])
            s = s + part
            
            if(len(x) == 0): # If no bits are found
                continue
            
            x,y = Variable(x),Variable(y)
            x = x.type(torch.FloatTensor).cuda()
            
            x = x.reshape(x.shape[0],1,x.shape[1],x.shape[2])
            y = torch.tensor(np.array(y).reshape(-1, 2)).cuda().float()
            
            features = model_fe(x)
            cls = model_abn(features)
            
            val_loss = loss_fn(cls, torch.argmax(y,dim=1))

            if(cls_pred is None):
                cls_pred = cls
                cls_val = y
            else:
                cls_pred = torch.cat((cls_pred,cls))
                cls_val = torch.cat((cls_val,y))
            
            epoch_val_loss = epoch_val_loss + val_loss

        epoch_val_loss = epoch_val_loss/len(val_parts)

        print("Validation loss - ", str(epoch_val_loss.item()))

        score_log = Scoring(e + 1, log_dir)
        
        Macc,F1,sensitivity,specificity, precision, \
            accuracy, threshold, scores, _, _, _ = score_log.log_score(cls_pred, cls_val, val_parts)

    logs = {'epoch' : e + 1,
            'loss' : epoch_loss.item(),
            'val_loss' : epoch_val_loss.item(),
            'val_macc' : Macc,
            'val_precision' : precision,
            'val_sensitivity' : sensitivity,
            'val_specificity' : specificity,
            'val_F1' : F1,
            'accuracy' : accuracy,
            'threshold' : threshold,
            'Lr' : ep_lr,
            'TN' : scores[0], 
            'FP' : scores[1], 
            'FN' : scores[2], 
            'TP' : scores[3]}

    model_dir_fe = os.path.join(log_dir, str(e+1) + \
                                "_saved_fe_model.pth.tar")
    model_dir_abn = os.path.join(log_dir, str(e+1) + \
                                "_saved_cls_model.pth.tar")

    torch.save({'epoch': e + 1, 'state_dict': model_fe.state_dict(), 
            'optimizer' : optimizer_fe.state_dict(),
            'scheduler' : [scheduler_fe.state_dict()],
            'loss' : epoch_loss.item(),
            'val_loss' : epoch_val_loss.item()}, model_dir_fe)
    torch.save({'epoch': e + 1, 'state_dict': model_abn.state_dict(),
            'optimizer' : optimizer_abn.state_dict(),
            'scheduler' : [scheduler_abn.state_dict()],
            'loss' : epoch_loss.item(),
            'val_loss' : epoch_val_loss.item()}, model_dir_abn)

    flow.reset()

    logger.on_epoch_end(e, logs)

logger.on_train_end()
logger_iter.on_train_end()
