#!/usr/bin/env python
# coding: utf-8

# # Chord

# In[ ]:


# IMPORTS

import os
import time

import numpy as np

import madmom

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as Dataset
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score

# configurations
import scripts.chord_config as cc

# feature, target, annotation initializer
from scripts.chord_feat import init_data

from scripts.chord_util import parse_annotations


# In[ ]:


# GLOBAL VARIABLES

# random seed
SEED = cc.SEED

# cuda configuration
USE_CUDA = cc.USE_CUDA
DEVICE = cc.DEVICE
print("CURRENT DEVICE:", DEVICE)

# paths
MODEL_NAME = cc.MODEL_NAME
MODEL_PATH = cc.MODEL_PATH
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)  
    
FPS = cc.FPS


# In[ ]:


# TRAINING PARAMETERS

num_epochs = cc.NUM_EPOCHS
lr = cc.LR
batch_size = cc.BATCH_SIZE
patience = cc.PATIENCE


# In[ ]:


# COMMAND LINE SUPPORT

# TODO:

TRAIN = cc.TRAIN
PREDICT = cc.PREDICT
VERBOSE = cc.VERBOSE

if VERBOSE:
    print('\n---- EXECUTION STARTED ----\n')
    # print('Command line arguments:\n\n', args, '\n')


# In[ ]:


# LOAD FEATURES AND ANNOTATIONS, COMPUTE TARGETS
train_f, train_t, train_anno, valid_f, valid_t, valid_anno, test_f, test_t, test_anno = init_data()

