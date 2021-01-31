#!/usr/bin/env python
# coding: utf-8

# # Beat SOTA

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

