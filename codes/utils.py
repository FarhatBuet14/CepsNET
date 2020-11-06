from __future__ import print_function, division, absolute_import
import numpy as np
np.random.seed(1)
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import types
from torch._six import inf
from collections import Counter
from functools import partial
import torch
    
def anno_merge(y):
    temp = np.zeros((len(y), 5))
    temp[:, 0] = y[:, 0]
    temp[:, 1] = np.logical_or([bool(x) for x in (y[:, 1])], [bool(x) for x in (y[:, 2])])
    temp[:, 2] = np.logical_or([bool(x) for x in (y[:, 3])], [bool(x) for x in (y[:, 4])])
    temp[:, 3] = np.logical_or([bool(x) for x in (y[:, 5])], [bool(x) for x in (y[:, 6])])
    temp[:, 4] = np.logical_or([bool(x) for x in (y[:, 7])], [bool(x) for x in (y[:, 8])])
    return temp


class Datagen_k():
    def __init__(self, x, y=None):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Scoring():
    def __init__(self, epoch, log_dir = None, last_func = "softmax"):
        self.epoch = epoch
        self.log_dir = log_dir
        self.last_func = last_func

    def log_score(self, y_pred, y_val, val_parts = None, y_domain = None, roc = False, list_out = False):

        self.y_pred = y_pred.cpu().detach().numpy() if(type(y_pred).__name__ == "Tensor") else y_pred
        self.y_val = y_val.cpu().detach().numpy() if(type(y_val).__name__ == "Tensor") else y_val
        self.val_parts = val_parts
        self.y_domain = y_domain
 
        pred = np.argmax(self.y_pred, axis=-1).reshape((-1, 1)) if(self.y_pred.shape[1] != 1) else self.y_pred
        true = np.argmax(self.y_val, axis=-1).reshape((-1, 1)) if(self.y_val.shape[1] != 1) else self.y_val
        
        if(self.val_parts is not None): true, pred, dom_list = self.get_full_wave(true, pred)
        
        if(self.last_func == "sigmoid" and roc):
            
            fpr, tpr, thresholds = roc_curve(true, pred)
            roc_auc = auc(fpr, tpr)
            gmeans = np.sqrt(tpr * (1-fpr))
            ix = np.argmax(gmeans)
            threshold = thresholds[ix]
            pred = (pred > threshold) * 1

            #--- Print ROC
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Thresholding from ROC curve')
            plt.legend(loc="lower right")
            plt.savefig(self.log_dir + "/" + str(self.epoch) + '_roc.png')
            # plt.show()
        
        else: threshold = 0.5
        
        TN, FP, FN, TP = confusion_matrix(true, pred, labels=[0,1]).ravel()
        scores = (TN, FP, FN, TP)
        eps = 0.0000001
        accuracy = ((TP + TN)/len(pred))
        sensitivity = TP / (TP + FN + eps)
        specificity = TN / (TN + FP + eps)
        precision = TP / (TP + FP + eps)
        F1 = 2 * (precision * sensitivity) / (precision + sensitivity + eps)
        Macc = (sensitivity + specificity) / 2
        
        if(val_parts is not None):
            print("TN:",TN,"FP:",FP,"FN:",FN,"TP:",TP)
            print("Sensitivity:","%.3f"%sensitivity,"Specificity:","%.3f"%specificity,"Precision:","%.3f"%precision,end=' ')
            print("F1:", "%.3f"%F1,"MACC", "%.3f"%Macc, "Accuracy", "%.3f"%accuracy)
        
        if(list_out and (val_parts is None)):
            return [y_domain, Macc, precision, sensitivity, specificity, F1, accuracy, 
                    threshold, scores[0], scores[1], scores[2], scores[3]]
        else:
            if(list_out):
                return ["Overall", Macc, precision, sensitivity, specificity, F1, accuracy, 
                threshold, scores[0], scores[1], scores[2], scores[3]], \
                    np.array(true), np.array(pred), np.array(dom_list)
            else:
                return Macc,F1,sensitivity,specificity,precision, accuracy, threshold, \
                        scores, np.array(true), np.array(pred), np.array(dom_list)

    def get_full_wave(self, y_val, y_pred):
        
        true = []
        pred = []
        dom_list = []
        start_idx = 0
        eps = 0.0000001
        
        threshold = 0.5
        y_pred_bin = (y_pred > threshold) * 1

        for _,s in enumerate(self.val_parts):

            if not s:  ## for e00032 in validation0 there was no cardiac cycle
                continue
            
            if(self.y_domain is not None):
                dic = Counter([d[0] for d in self.y_domain[start_idx:start_idx + int(s)]])
                if(len(dic) == 1):
                    dom_list.append([k for k, v in dic.items()][0])
                else: print("Wrong Val_parts given")
            
            temp_ = y_val[start_idx:start_idx + int(s)]
            temp_bin = y_pred_bin[start_idx:start_idx + int(s)]
            temp = y_pred[start_idx:start_idx + int(s)]

            if(self.last_func == "sigmoid"):
                if (sum(temp_bin == 0) > sum(temp_bin == 1)):
                    zero = np.sum(temp[temp<threshold])/(len(temp[temp<threshold])+eps)
                    pred.append(zero)
                else:
                    one = np.sum(temp[temp>=threshold])/(len(temp[temp>=threshold])+eps)
                    pred.append(one)
            else:
                if (sum(temp == 0) > sum(temp == 1)):
                    pred.append(0)
                else:
                    pred.append(1)

            if (sum(temp_ == 0) > sum(temp_ == 1)):
                true.append(0)
            else:
                true.append(1)

            start_idx = start_idx + int(s)
 
        return np.array(true), np.array(pred), np.array(dom_list)


def featureGen(x_, inp = None):
    import python_speech_features as psf
    if(inp == "log-Filterbank-DCT-26-13"):
        fet = np.array([(psf.base.mfcc(x, samplerate=1000, winlen=0.05, winstep=0.01)) for x in x_])
    elif(inp == "log-Filterbank-DCT-26"):
        fet = np.array([(psf.base.mfcc(x, samplerate=1000, winlen=0.05, winstep=0.01, numcep=26)) for x in x_])
    elif(inp == "log-Filterbank-26"):
        fet = np.array([(psf.base.logfbank(x, samplerate=1000, winlen=0.05, winstep=0.01)) for x in x_])
    elif(inp == "Filterbank-26"):
        fet = np.array([(psf.base.fbank(x, samplerate=1000, winlen=0.05, winstep=0.01)[0]) for x in x_])
    elif(inp == "13-merged-2"):
        fet_1 = np.array([(psf.base.mfcc(x, samplerate=1000, winlen=0.05, winstep=0.01)) for x in x_])
        fet_2 = np.array([(psf.base.delta(x, 2)) for x in fet_1])
        fet = np.stack([fet_1, fet_2])
    elif(inp == "13-merged-3"):
        fet_1 = np.array([(psf.base.mfcc(x, samplerate=1000, winlen=0.05, winstep=0.01)) for x in x_])
        fet_2 = np.array([(psf.base.delta(x, 2)) for x in fet_1])
        fet_3 = np.array([(psf.base.delta(x, 2)) for x in fet_2])
        fet = np.stack([fet_1, fet_2, fet_3])
    elif(inp == "26-merged-2d"):
        fet_1 = np.array([(psf.base.mfcc(x, samplerate=1000, winlen=0.05, winstep=0.01, numcep=26)) for x in x_])
        fet_2 = np.array([(psf.base.delta(x, 2)) for x in fet_1])
        fet = np.stack([fet_1, fet_2])
    elif(inp == "26-merged-3d"):
        fet_1 = np.array([(psf.base.mfcc(x, samplerate=1000, winlen=0.05, winstep=0.01, numcep=26)) for x in x_])
        fet_2 = np.array([(psf.base.delta(x, 2)) for x in fet_1])
        fet_3 = np.array([(psf.base.delta(x, 2)) for x in fet_2])
        fet = np.stack([fet_1, fet_2, fet_3])
    elif(inp == "26-merged-2"):
        fbank = np.array([(psf.base.fbank(x, samplerate=1000, winlen=0.05, winstep=0.01)[0]) for x in x_])
        log_fbank = np.array([(psf.base.logfbank(x, samplerate=1000, winlen=0.05, winstep=0.01)) for x in x_])
        fet = np.stack([fbank, log_fbank])
    
    if("merged" not in inp):
        fet = np.array([((x-np.mean(x))/np.std(x)) for x in fet])
    else:
        fet = np.array([np.array([((x-np.mean(x))/np.std(x)) for x in chan]) for chan in fet])
        fet = np.moveaxis(fet, 0, 3)
    
    return fet

