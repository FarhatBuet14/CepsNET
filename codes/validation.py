# python validation.py fbank
from __future__ import print_function, division, absolute_import

import torch
torch.cuda.empty_cache()

import os,time
import numpy as np
np.random.seed(1)
import math
import utils
from utils import Scoring
import matplotlib.pyplot as plt
from dataLoader import reshape_folds
from collections import Counter
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Necessary variables')
parser.add_argument("--feat",type=str, help = "Input Feature")
arguments = parser.parse_args()

df = pd.read_excel (r'../logs/pretrained.xlsx')
features = df["feature"].iloc[:]
index = [i for i, feat in enumerate(features) if(feat==arguments.feat)][0]

class paramClass():
    def __init__(self):
        self.epoch = df["epoch"][index]
        self.batch_size = df["batch_size"][index]
        self.inp = df["inp"][index]
        self.base_lr = df["base_lr"][index]
        self.max_lr = df["max_lr"][index]
        self.step_size = df["step_size"][index]
        self.checkpoint = '../logs/' + self.inp + "/checkpoints/"
param = paramClass()

# --- Import train-validation Data 

print("Loading data..")
data_file = "../data/feature/train_val_npzs/" + param.inp + "_train_val_data.npz"
data = np.load(data_file)

x_val = data['x_val_mfcc']
y_val = data['y_val']
val_parts = data['val_parts']
val_wav_files = data['val_wav_files']

x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2], 1)

#--- Reshape the folds
_, [y_val] = reshape_folds([], [y_val])

from keras.utils import to_categorical
y_val = to_categorical(y_val, num_classes = 2)

# --- Import necessary libraries for training purpose

import torch.optim as optim
import torch
from torch.autograd import Variable
import time
import torch
import torch.nn as nn
from torchsummary import summary
from torch.backends import cudnn

# --- Load the model and optimizers and loss function

from model import resnet_extractor, abn_classifier
nodes = [16, 32, 64, 128]
num_layers = 2
model_fe = resnet_extractor(1, None, param.inp, nodes, num_layers).cuda()
model_abn = abn_classifier(model_fe.out_features, 2, dropout = None).cuda()

e = param.epoch - 1
print("EPOCH ---- ( " + str(e+1) + " )----")
checkpoint_fe = param.checkpoint + str(e+1) + "_saved_fe_model.pth.tar"
model_fe.load_state_dict(torch.load(checkpoint_fe)["state_dict"])
checkpoint_abn = param.checkpoint + str(e+1) + "_saved_cls_model.pth.tar"
model_abn.load_state_dict(torch.load(checkpoint_abn)["state_dict"])

epoch_loss = torch.load(checkpoint_fe)["loss"]
val_loss_load = torch.load(checkpoint_fe)["val_loss"]

loss_fn = nn.CrossEntropyLoss()

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

domain = np.asarray(val_wav_files).reshape((-1, 1))
score_log = Scoring(e + 1)
score, true, pred, subset = score_log.log_score(cls_pred, cls_val, val_parts, 
        y_domain = domain, list_out=True)

tpn = true == pred
avg = 0
accu = []
log = []
log.append(score)
for dataset in np.unique(subset):
    mask = subset == dataset
    true_sub = true[mask].reshape((-1, 1))
    pred_sub = pred[mask].reshape((-1, 1))
    sub = subset[mask].reshape((-1, 1))
    score = score_log.log_score(pred_sub, true_sub, y_domain = dataset, list_out=True)
    log.append(score)

avg = []
std = []
avg.append("average")
std.append("Std")
for l in range(1, len(log[0])-5):
    s = []
    for lo in log:
        if(lo[0] == "Overall"): continue
        s.append(lo[l])
    avg.append(np.array(s).mean())
    std.append(np.array(s).std())

import csv
with open('../logs/' + param.inp + '/validation_fold_wise_' + str(param.epoch) + '.csv', 'w', newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["domain", "Macc", "precision", "sensitivity", "specificity", "F1", "accuracy", 
                    "threshold", "TN", "FP", "FN", "TP"])
    for row in log:
        writer.writerow(row)
    writer.writerow(avg)
    writer.writerow(std)

