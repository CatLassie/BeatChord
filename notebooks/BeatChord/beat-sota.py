#!/usr/bin/env python
# coding: utf-8

# # Beat SOTA

# In[ ]:


# IMPORTS

import os
import time

import pandas as pd
import numpy as np

import madmom
from madmom.features.onsets import OnsetPeakPickingProcessor
# from madmom.features.beats import BeatTrackingProcessor
from madmom.evaluation.beats import BeatEvaluation
from madmom.evaluation.beats import BeatMeanEvaluation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as Dataset
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score

# configurations
import scripts.beat_sota_config as bsc

# feature, target, annotation initializer
from scripts.beat_sota_feat import init_data


# In[ ]:


# GLOBAL VARIABLES

# random seed
SEED = bsc.SEED

# cuda configuration
USE_CUDA = bsc.USE_CUDA
DEVICE = bsc.DEVICE
print("CURRENT DEVICE:", DEVICE)

# paths
MODEL_NAME = bsc.MODEL_NAME
MODEL_PATH = bsc.MODEL_PATH
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)  
    
FPS = bsc.FPS

# peak picker params
THRESHOLD = bsc.THRESHOLD
PRE_AVG = bsc.PRE_AVG
POST_AVG = bsc.POST_AVG
PRE_MAX = bsc.PRE_MAX
POST_MAX = bsc.POST_MAX


# In[ ]:


# TRAINING PARAMETERS

num_epochs = bsc.NUM_EPOCHS

lr = bsc.LR

feature_context = bsc.FEATURE_CONTEXT
traininig_hop_size = bsc.TRAINING_HOP_SIZE

batch_size = bsc.BATCH_SIZE
patience = bsc.PATIENCE


# In[ ]:


# COMMAND LINE SUPPORT

# TODO:

TRAIN = bsc.TRAIN
TRAIN_EXISTING = bsc.TRAIN_EXISTING
PREDICT = bsc.PREDICT
VERBOSE = bsc.VERBOSE

if VERBOSE:
    print('\n---- EXECUTION STARTED ----\n')
    print('Train:', TRAIN)
    print('Train existing model:', TRAIN_EXISTING)
    print('Predict', PREDICT)
    # print('Command line arguments:\n\n', args, '\n')


# In[ ]:


# LOAD FEATURES AND ANNOTATIONS, COMPUTE TARGETS
train_f, train_t, train_anno, valid_f, valid_t, valid_anno, test_f, test_t, test_anno = init_data()


# In[ ]:


# NETWORK PARAMETERS

# CNN

LAST_CNN_KERNEL_FREQUENCY_SIZE = bsc.LAST_CNN_KERNEL_FREQUENCY_SIZE

# filters
cnn_in_size = 1
cnn_h1_size = 32
cnn_h2_size = 32
cnn_h3_size = 64

# kernels
cnn_k_1_size = 3
cnn_k_2_size = (1, 2)
cnn_padding = (1,0)
cnn_max_pool_k_size = (1,3)

cnn_dropout_rate = 0.1

# TCN

tcn_layer_num = 11 #11

# filters
tcn_h_size = 64

# kernels
tcn_k_size = 5
tcn_dilations = [2**x for x in range(0, tcn_layer_num)]
tcn_paddings = [2*x for x in tcn_dilations]

tcn_dropout_rate = 0.1

# FULLY CONNECTED (by using a 1d convolutional. layer)

# filters
fc_h_size = 64
fc_out_size = 1

# kernels
fc_k_size = 1

# alternatively a dense layer with in size 8193*16 and out size 1 ?


# In[ ]:


# BEAT NETWORK CLASS and DATA SET CLASS for DATA LOADER

class BeatNet(nn.Module):
    def __init__(self):
        super(BeatNet, self).__init__()
        
        self.l1 = nn.Sequential(
            nn.Conv2d(cnn_in_size, cnn_h1_size, cnn_k_1_size, padding=cnn_padding),
            nn.BatchNorm2d(cnn_h1_size),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = cnn_max_pool_k_size),
            nn.Dropout2d(p = cnn_dropout_rate)
        )
        
        self.l2 = nn.Sequential(
            nn.Conv2d(cnn_h1_size, cnn_h2_size, cnn_k_1_size, padding=cnn_padding),
            nn.BatchNorm2d(cnn_h2_size),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = cnn_max_pool_k_size),
            nn.Dropout2d(p = cnn_dropout_rate)
        )
        
        self.l2b = nn.Sequential(
            nn.Conv2d(cnn_h2_size, cnn_h3_size, cnn_k_1_size, padding=cnn_padding),
            nn.BatchNorm2d(cnn_h3_size),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = cnn_max_pool_k_size),
            nn.Dropout2d(p = cnn_dropout_rate)
        )
        
        self.l3 = nn.Sequential(
            nn.Conv2d(cnn_h3_size, cnn_h3_size, cnn_k_2_size),
            # nn.BatchNorm2d(cnn_h_size), # cant use because spec is reduced to 1x1
            # NOTE: if needed try Instance normalization (InstanceNorm2d)
            nn.ELU(),
            nn.Dropout2d(p = cnn_dropout_rate)
        )
        
        self.ld1 = nn.Sequential(
                nn.Conv1d(tcn_h_size, tcn_h_size, tcn_k_size, padding=tcn_paddings[0], dilation=tcn_dilations[0]),
                nn.BatchNorm1d(tcn_h_size),
                nn.ELU(),
                nn.Dropout(p = tcn_dropout_rate)
        )
        
        self.ld2 = nn.Sequential(
                nn.Conv1d(tcn_h_size, tcn_h_size, tcn_k_size, padding=tcn_paddings[1], dilation=tcn_dilations[1]),
                nn.BatchNorm1d(tcn_h_size),
                nn.ELU(),
                nn.Dropout(p = tcn_dropout_rate)
        )
        
        self.ld3 = nn.Sequential(
                nn.Conv1d(tcn_h_size, tcn_h_size, tcn_k_size, padding=tcn_paddings[2], dilation=tcn_dilations[2]),
                nn.BatchNorm1d(tcn_h_size),
                nn.ELU(),
                nn.Dropout(p = tcn_dropout_rate)
        )
        
        self.ld4 = nn.Sequential(
                nn.Conv1d(tcn_h_size, tcn_h_size, tcn_k_size, padding=tcn_paddings[3], dilation=tcn_dilations[3]),
                nn.BatchNorm1d(tcn_h_size),
                nn.ELU(),
                nn.Dropout(p = tcn_dropout_rate)
        )
        
        self.ld5 = nn.Sequential(
                nn.Conv1d(tcn_h_size, tcn_h_size, tcn_k_size, padding=tcn_paddings[4], dilation=tcn_dilations[4]),
                nn.BatchNorm1d(tcn_h_size),
                nn.ELU(),
                nn.Dropout(p = tcn_dropout_rate)
        )
        
        self.ld6 = nn.Sequential(
                nn.Conv1d(tcn_h_size, tcn_h_size, tcn_k_size, padding=tcn_paddings[5], dilation=tcn_dilations[5]),
                nn.BatchNorm1d(tcn_h_size),
                nn.ELU(),
                nn.Dropout(p = tcn_dropout_rate)
        )
        
        self.ld7 = nn.Sequential(
                nn.Conv1d(tcn_h_size, tcn_h_size, tcn_k_size, padding=tcn_paddings[6], dilation=tcn_dilations[6]),
                nn.BatchNorm1d(tcn_h_size),
                nn.ELU(),
                nn.Dropout(p = tcn_dropout_rate)
        )
        
        self.ld8 = nn.Sequential(
                nn.Conv1d(tcn_h_size, tcn_h_size, tcn_k_size, padding=tcn_paddings[7], dilation=tcn_dilations[7]),
                nn.BatchNorm1d(tcn_h_size),
                nn.ELU(),
                nn.Dropout(p = tcn_dropout_rate)
        )

        self.ld9 = nn.Sequential(
                nn.Conv1d(tcn_h_size, tcn_h_size, tcn_k_size, padding=tcn_paddings[8], dilation=tcn_dilations[8]),
                nn.BatchNorm1d(tcn_h_size),
                nn.ELU(),
                nn.Dropout(p = tcn_dropout_rate)
        )
        
        self.ld10 = nn.Sequential(
                nn.Conv1d(tcn_h_size, tcn_h_size, tcn_k_size, padding=tcn_paddings[9], dilation=tcn_dilations[9]),
                nn.BatchNorm1d(tcn_h_size),
                nn.ELU(),
                nn.Dropout(p = tcn_dropout_rate)
        )
        
        self.ld11 = nn.Sequential(
                nn.Conv1d(tcn_h_size, tcn_h_size, tcn_k_size, padding=tcn_paddings[10], dilation=tcn_dilations[10]),
                nn.BatchNorm1d(tcn_h_size),
                nn.ELU(),
                nn.Dropout(p = tcn_dropout_rate)
        )
        
        self.lfc = nn.Sequential(
            nn.Conv1d(fc_h_size, fc_out_size, fc_k_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):

        # print(x.shape)

        out = self.l1(x)
        # print(out.shape)

        out = self.l2(out)
        # print(out.shape)
        out = self.l2b(out)

        out = self.l3(out)
        # print(out.shape)
        
        out.squeeze_(-1)
        # print(out.shape)
        
        out = self.ld1(out)
        out = self.ld2(out)
        out = self.ld3(out)
        out = self.ld4(out)
        out = self.ld5(out)
        out = self.ld6(out)
        out = self.ld7(out)
        out = self.ld8(out)
        out = self.ld9(out)
        out = self.ld10(out)
        out = self.ld11(out)
        # print(out.shape)
        
        out = self.lfc(out)
        # print(out.shape)
        
        out = out.squeeze(1)
        # print(out.shape)

        return out
    


# Dataset for DataLoader (items are pairs of Context x 81 (time x freq.) spectrogram snippets and 0-1 (0.5) target values)
class BeatSet(Dataset):
    def __init__(self, feat_list, targ_list, context, hop_size):
        self.features = feat_list
        self.targets = targ_list
        self.context = context
        self.hop_size = hop_size
 
        # list with snippet count per track
        self.snip_cnt = []
        # overall snippet count
        total_snip_cnt = 0
        # calculate overall number of snippets we can get from our data
        for feat in feat_list:
            if feat.shape[0]- self.context >= 0: # !!! WARNING: was > previously !!!
                cur_len = int(np.floor((feat.shape[0] - self.context)/hop_size) + 1)
                self.snip_cnt.append(cur_len)
                total_snip_cnt += cur_len
            else:
                cur_len = 1
                self.snip_cnt.append(cur_len)
                total_snip_cnt += cur_len 
                #print("warning: skipped 1 example, shape", feat.shape[0])

        self.length = int(total_snip_cnt)
        super(BeatSet, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # find track which contains snippet with index [index]
        overal_pos = 0
        for idx, cnt in enumerate(self.snip_cnt):
            # check if this track contains the snippet with index [index]
            if index < overal_pos+cnt:
                break
            else:
                # if not, add the current tracks snippet count to the overall snippet count already visited
                overal_pos += cnt

        # calculate the position of the snippet within the track nr. [idx]
        position = index-overal_pos
        position *= self.hop_size

        # get snippet and target
        
        sample = None #self.features[idx][position : position+self.context] if self.features[idx].shape[0] - self.context >= 0 else self.features[idx]
        target = None #self.targets[idx][position : position+self.context] if self.targets[idx].shape[0] - self.context >= 0 else self.targets[idx]

        if self.features[idx].shape[0] - self.context >= 0:
            sample = self.features[idx][position : position+self.context]
        else:
            sample = np.zeros((self.context, self.features[idx].shape[1]), np.float32)
            start = 0
            increment = self.features[idx].shape[0]
            remainder = self.context
            while remainder >= increment:
                sample[start:start+increment] = self.features[idx]
                start += increment
                remainder -= increment
            sample[start:] = self.features[idx][:remainder]

        if self.targets[idx].shape[0] - self.context >= 0:
            target = self.targets[idx][position : position+self.context]
        else:
            target = np.zeros(self.context, np.float32)
            start = 0
            increment = self.targets[idx].shape[0]
            remainder = self.context
            while remainder >= increment:
                target[start:start+increment] = self.targets[idx]
                start += increment
                remainder -= increment
            target[start:] = self.targets[idx][:remainder]

        # convert to PyTorch tensor and return (unsqueeze is for extra dimension, asarray is cause target is scalar)
        return torch.from_numpy(sample).unsqueeze_(0), torch.from_numpy(np.asarray(target))



# helper class for arguments
class Args:
    pass


# In[ ]:


# TRAIN / TEST / PREDICT

def train_one_epoch(args, model, device, train_loader, optimizer, epoch):
    """
    Training for one epoch.
    """
    # set model to training mode (activate dropout / batch normalization).
    model.train()
    t = time.time()
    # iterate through all data using the loader
    for batch_idx, (data, target) in enumerate(train_loader):
        # move data to device
        data, target = data.to(device), target.to(device)
        
        # reset optimizer (clear previous gradients)
        optimizer.zero_grad()
        # forward pass (calculate output of network for input)
        output = model(data.float())
        # calculate loss        
        loss = F.binary_cross_entropy(output, target)
        # do a backward pass (calculate gradients using automatic differentiation and backpropagation)
        loss.backward()
        # udpate parameters of network using calculated gradients
        optimizer.step()
        
        # print logs
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, took {:.2f}s'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), time.time()-t))
            t = time.time()

        # WORK IN PROGRESS: skip rest of loop
        # print('train batch index:', batch_idx)
        # break
       
def calculate_unseen_loss(model, device, unseen_loader):
    """
    Calculate loss for unseen data (validation or testing)
    :return: cumulative loss
    """
    # set model to inference mode (deactivate dropout / batch normalization).
    model.eval()
    # init cumulative loss
    unseen_loss = 0
    # no gradient calculation    
    with torch.no_grad():
        # iterate over test data
        for data, target in unseen_loader:
            # move data to device
            data, target = data.to(device), target.to(device)
            # forward pass (calculate output of network for input)
            output = model(data.float())
            
            # WORK IN PROGRESS: skip rest of loop
            # continue
            
            # claculate loss and add it to our cumulative loss
            unseen_loss += F.binary_cross_entropy(output, target, reduction='sum').item() # sum up batch loss

    # output results of test run
    unseen_loss /= len(unseen_loader.dataset)
    print('Average loss: {:.4f}\n'.format(
        unseen_loss, len(unseen_loader.dataset)))

    return unseen_loss
  
    
    
def predict(model, device, data, context):
    """
    Predict beat
    :return: prediction
    """
    # set model to inference mode (deactivate dropout / batch normalization).
    model.eval()
    output = None
    # move data to device
    data = torch.from_numpy(data[None, None, :, :])
    data = data.to(device)
    # no gradient calculation
    with torch.no_grad():
        output = model(data.float())
    return output


# In[ ]:


def run_training():
    print('Training network...')

    # parameters for NN training
    args = Args()
    args.batch_size = batch_size #1 #64
    args.max_epochs = num_epochs #25 #1000
    args.patience = patience #4
    args.lr = lr # 0.001, 0.0001
    args.momentum = 0.5 #UNUSED
    args.log_interval = 100 #100
    args.context = feature_context #5
    args.hop_size = traininig_hop_size

    # setup pytorch
    torch.manual_seed(SEED)
    
    # create model and optimizer
    model = BeatNet().to(DEVICE)
    if TRAIN_EXISTING:
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + '.model')))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # setup our datasets for training, evaluation and testing
    kwargs = {'num_workers': 4, 'pin_memory': True} if USE_CUDA else {'num_workers': 4}
    train_loader = torch.utils.data.DataLoader(BeatSet(train_f, train_t, args.context, args.hop_size),
                                               batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(BeatSet(valid_f, valid_t, args.context, args.hop_size),
                                               batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(BeatSet(test_f, test_t, args.context, args.hop_size),
                                              batch_size=args.batch_size, shuffle=False, **kwargs)

    # main training loop
    best_validation_loss = 9999
    cur_patience = args.patience
    for epoch in range(1, args.max_epochs + 1):
        # run one epoch of NN training
        train_one_epoch(args, model, DEVICE, train_loader, optimizer, epoch)
        
        # WORK IN PROGRESS: 
        # return
        
        # validate on validation set
        print('\nValidation Set:')
        validation_loss = calculate_unseen_loss(model, DEVICE, valid_loader)
        # check for early stopping
        if validation_loss < best_validation_loss:
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME + '.model'))
            best_validation_loss = validation_loss
            cur_patience = args.patience
        else:
            # if performance does not improve, we do not stop immediately but wait for 4 iterations (patience)
            if cur_patience <= 0:
                print('Early stopping, no improvement for %d epochs...' % args.patience)
                break
            else:
                print('No improvement, patience: %d' % cur_patience)
                cur_patience -= 1

    # testing on test data
    print('Evaluate network...')
    print('Test Set:')
    # calculate loss for test set
    calculate_unseen_loss(model, DEVICE, test_loader)


# In[ ]:


if TRAIN or TRAIN_EXISTING:
    run_training()


# In[ ]:


def run_prediction(test_features): 
    args = Args()
    args.context = feature_context #5
    
    torch.manual_seed(SEED)
    
    # load model
    model = BeatNet().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + '.model')))
    print('model loaded...')
    
    # calculate actual output for the test data
    results_cnn = [None for _ in range(len(test_features))]
    # iterate over test tracks
    for test_idx, cur_test_feat in enumerate(test_features):
        if test_idx % 100 == 0:
            completion = int((test_idx / len(test_features))*100)
            print(str(completion)+'% complete...')
        
        # run the inference method
        result = predict(model, DEVICE, cur_test_feat, args.context)
        results_cnn[test_idx] = result.cpu().numpy()

    return results_cnn


# In[ ]:


predicted = None
picked_beats = []

if PREDICT:
    # beat_picker = BeatTrackingProcessor(fps=FPS) # TODO: replace with OnsetPeakPickingProcessor(fps=FPS)
    beat_picker = OnsetPeakPickingProcessor(fps=FPS, threshold=THRESHOLD, pre_avg=PRE_AVG, post_avg=POST_AVG, pre_max=PRE_MAX, post_max=POST_MAX) # TODO: replace with OnsetPeakPickingProcessor(fps=FPS)
    
    # predict beats
    if VERBOSE:
        print('predicting...')
    predicted = run_prediction(test_f) #[test_t[0], test_t[1]]
    
    # pick peaks
    if VERBOSE:
        print('picking beats...')
        
    for i, pred in enumerate(predicted):
        picked = beat_picker(pred.squeeze(0)) # squeeze cause the dimensions are (1, frame_num, cause of the batch)!!!
        picked_beats.append(picked)
        
    if VERBOSE:
        print('\n')
    
    # evaluate results
    if VERBOSE:
        print('evaluating results...')
        
    evals = []
    for i, beat in enumerate(picked_beats):
        e = madmom.evaluation.beats.BeatEvaluation(beat, test_anno[i])
        evals.append(e)
        
    if VERBOSE:
        print('\n')
    
    mean_eval = madmom.evaluation.beats.BeatMeanEvaluation(evals)
    print(mean_eval)
    
    # print('F-Measure:', mean_eval.fmeasure)
    # print('CMLc:', mean_eval.cmlc, 'CMLt:', mean_eval.cmlt, 'AMLc:', mean_eval.amlc, 'AMLt:', mean_eval.amlt)
    # print('Information Gain:', mean_eval.information_gain)

