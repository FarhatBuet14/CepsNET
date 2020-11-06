# python prepare_cepstralFeature.py 
from __future__ import print_function, division, absolute_import

from dataLoader import reshape_folds
import numpy as np
np.random.seed(1)
import h5py
import argparse
import python_speech_features as psf
import pandas as pd

parser = argparse.ArgumentParser(description='Necessary variables')
parser.add_argument("--inp",type=str, help = "Input Feature")
arguments = parser.parse_args()

#--- Load Data

data = h5py.File("../data/feature/folds/fold_0.mat", 'r')
x_train = np.array(data['trainX'][:]).astype('float32')
y_train = data['trainY'][:][0].astype('int32')
y_train[y_train<0] = 0
train_parts = data['train_parts'][0].astype('int32')
x_val = np.array(data['valX'][:]).astype('float32')
y_val = data['valY'][:][0].astype('int32')
y_val[y_val<0] = 0
val_parts = data['val_parts'][0].astype('int32')
val_wav_files = [chr(stp) for stp in data['val_files'][0]]

df = pd.read_excel (r'../logs/pretrained.xlsx')
features = df["feature"].iloc[:]
index = [i for i, feat in enumerate(features) if(feat==arguments.inp)][0]
inp = df["inp"][index]
print(f'Converting to - {inp}')

# Functions for Feature Extraction

def get_fb(train, val):
    out = np.array([(psf.base.fbank(x, samplerate=1000, winlen=0.05, winstep=0.01)[0]) for x in train.transpose()])
    train = np.array([(x-np.mean(x))/np.std(x) for x in out])
    out = np.array([(psf.base.fbank(x, samplerate=1000, winlen=0.05, winstep=0.01)[0]) for x in val.transpose()])
    val = np.array([(x-np.mean(x))/np.std(x) for x in out])
    return train, val

def get_logfb(train, val):
    out = np.array([(psf.base.logfbank(x, samplerate=1000, winlen=0.05, winstep=0.01)) for x in train.transpose()])
    train = np.array([(x-np.mean(x))/np.std(x) for x in out])
    out = np.array([(psf.base.logfbank(x, samplerate=1000, winlen=0.05, winstep=0.01)) for x in val.transpose()])
    val = np.array([(x-np.mean(x))/np.std(x) for x in out])
    return train, val

def get_mfcc26(train, val):
    out = np.array([(psf.base.mfcc(x, samplerate=1000, winlen=0.05, winstep=0.01, numcep=26)) for x in train.transpose()])
    train = np.array([(x-np.mean(x))/np.std(x) for x in out])
    out = np.array([(psf.base.mfcc(x, samplerate=1000, winlen=0.05, winstep=0.01, numcep=26)) for x in val.transpose()])
    val = np.array([(x-np.mean(x))/np.std(x) for x in out])
    return train, val

def get_mfcc13(train, val):
    out = np.array([(psf.base.mfcc(x, samplerate=1000, winlen=0.05, winstep=0.01, numcep=13)) for x in train.transpose()])
    train = np.array([(x-np.mean(x))/np.std(x) for x in out])
    out = np.array([(psf.base.mfcc(x, samplerate=1000, winlen=0.05, winstep=0.01, numcep=13)) for x in val.transpose()])
    val = np.array([(x-np.mean(x))/np.std(x) for x in out])
    return train, val

def get_mfcc13d(train, val):
    out = np.array([(psf.base.mfcc(x, samplerate=1000, winlen=0.05, winstep=0.01, numcep=13)) for x in train.transpose()])
    out = np.array([(psf.base.delta(x, 2)) for x in out])
    train = np.array([(x-np.mean(x))/np.std(x) for x in out])
    out = np.array([(psf.base.mfcc(x, samplerate=1000, winlen=0.05, winstep=0.01, numcep=13)) for x in val.transpose()])
    out = np.array([(psf.base.delta(x, 2)) for x in out])
    val = np.array([(x-np.mean(x))/np.std(x) for x in out])
    return train, val

def get_mfcc13dd(train, val):
    out = np.array([(psf.base.mfcc(x, samplerate=1000, winlen=0.05, winstep=0.01, numcep=13)) for x in train.transpose()])
    out = np.array([(psf.base.delta(x, 2)) for x in out])
    out = np.array([(psf.base.delta(x, 2)) for x in out])
    train = np.array([(x-np.mean(x))/np.std(x) for x in out])
    out = np.array([(psf.base.mfcc(x, samplerate=1000, winlen=0.05, winstep=0.01, numcep=13)) for x in val.transpose()])
    out = np.array([(psf.base.delta(x, 2)) for x in out])
    out = np.array([(psf.base.delta(x, 2)) for x in out])
    val = np.array([(x-np.mean(x))/np.std(x) for x in out])
    return train, val


#--- Cepstral Feature Extraction

if(inp == "Filterbank-26"): train, val = get_fb(x_train, x_val)
if(inp == "log-Filterbank-26"): train, val = get_logfb(x_train, x_val)
if(inp == "log-Filterbank-DCT-26"): train, val = get_mfcc26(x_train, x_val)
if(inp == "log-Filterbank-DCT-26-13"): train, val = get_mfcc13(x_train, x_val)

if(inp == "13-MF2613D-merged_flat-2"):
    feat1_t, feat1_v = get_mfcc13(x_train, x_val)
    feat2_t, feat2_v = get_mfcc13d(x_train, x_val)
    train = np.concatenate([feat1_t, feat2_t], axis=-1)
    val = np.concatenate([feat1_v, feat2_v], axis=-1)
if(inp == "13-MF2613DD-merged_flat-3"):
    feat1_t, feat1_v = get_mfcc13(x_train, x_val)
    feat2_t, feat2_v = get_mfcc13dd(x_train, x_val)
    train = np.concatenate([feat1_t, feat2_t], axis=-1)
    val = np.concatenate([feat1_v, feat2_v], axis=-1)

if(inp == "26-F26LF26-merged_flat-2"):
    feat1_t, feat1_v = get_fb(x_train, x_val)
    feat2_t, feat2_v = get_logfb(x_train, x_val)
    train = np.concatenate([feat1_t, feat2_t], axis=-1)
    val = np.concatenate([feat1_v, feat2_v], axis=-1)
if(inp == "13-F26MF2613-merged_flat-3"):
    feat1_t, feat1_v = get_fb(x_train, x_val)
    feat2_t, feat2_v = get_mfcc13(x_train, x_val)
    train = np.concatenate([feat1_t, feat2_t], axis=-1)
    val = np.concatenate([feat1_v, feat2_v], axis=-1)
if(inp == "13-LF26MF2613-merged_flat-3"):
    feat1_t, feat1_v = get_logfb(x_train, x_val)
    feat2_t, feat2_v = get_mfcc13(x_train, x_val)
    train = np.concatenate([feat1_t, feat2_t], axis=-1)
    val = np.concatenate([feat1_v, feat2_v], axis=-1)
if(inp == "13-F26LF26MF2613-merged_flat-5"):
    feat1_t, feat1_v = get_fb(x_train, x_val)
    feat2_t, feat2_v = get_logfb(x_train, x_val)
    feat3_t, feat3_v = get_mfcc13(x_train, x_val)
    train = np.concatenate([feat1_t, feat2_t, feat3_t], axis=-1)
    val = np.concatenate([feat1_t, feat2_t, feat3_t], axis=-1)

np.savez("../data/feature/train_val_npzs/" + inp + "_train_val_data.npz", x_train_mfcc = train, x_val_mfcc = val, 
                                                                y_train = y_train, y_val = y_val, 
                                                                val_parts = val_parts, 
                                                                val_wav_files = val_wav_files)
