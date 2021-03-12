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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

feature_context = cc.FEATURE_CONTEXT
traininig_hop_size = cc.TRAINING_HOP_SIZE


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


# In[ ]:


# NETWORK PARAMETERS

# CNN

# filters
cnn_in_size = 1
cnn_h1_size = 16 #32
cnn_h2_size = 32 #64

# kernels
cnn_k_size = 3
cnn_padding = (1,1)
cnn_max_pool_k_size = (2,2)

cnn_dropout_rate = 0.1

# FULLY CONNECTED (by using a 1d convolutional. layer)

# filters
fc_h1_size = 156 #52 # neurons in FC layers
fc_out_size = 13 # 13 outputs for 13 classes

# kernels
fc_k1_size = (6,22) #22 # something big that would correspond to an FC layer (capture all data into 1 input)
fc_k2_size = 1 # second FC layer gets input from first one, filter size is 1

# loss function
loss_func = nn.CrossEntropyLoss()
unseen_loss_func = nn.CrossEntropyLoss(reduction="sum")


# In[ ]:


# CHORD NETWORK CLASS and DATA SET CLASS for DATA LOADER

class ChordNet(nn.Module):
    def __init__(self):
        super(ChordNet, self).__init__()
        
        self.l1 = nn.Sequential(
            nn.Conv2d(cnn_in_size, cnn_h1_size, cnn_k_size, padding=cnn_padding),
            nn.BatchNorm2d(cnn_h1_size),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = cnn_max_pool_k_size),
            nn.Dropout2d(p = cnn_dropout_rate)
        )
        
        self.l2 = nn.Sequential(
            nn.Conv2d(cnn_h1_size, cnn_h2_size, cnn_k_size, padding=cnn_padding),
            nn.BatchNorm2d(cnn_h2_size),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = cnn_max_pool_k_size),
            nn.Dropout2d(p = cnn_dropout_rate)
        )
        
        self.lfc1 = nn.Sequential(
            nn.Conv2d(cnn_h2_size, fc_h1_size, fc_k1_size), # nn.Conv1d(cnn_h2_size, fc_h1_size, fc_k1_size),
            # nn.BatchNorm1d(fc_h1_size),
            nn.ELU(),
            nn.Dropout(p = cnn_dropout_rate)
        )
        
        self.lfcout = nn.Sequential(
            nn.Conv1d(fc_h1_size, fc_out_size, fc_k2_size),
            #nn.Softmax(1),
        )
        
    def forward(self, x):

        #print(x.shape)

        out = self.l1(x)
        #print(out.shape)

        out = self.l2(out)
        #print(out.shape)
        
        #out.squeeze_(2)
        #print(out.shape)

        out = self.lfc1(out)
        #print(out.shape)
                
        out.squeeze_(2)
        #print(out.shape)
            
        out = self.lfcout(out)
        #print(out.shape)
        
        out.squeeze_(-1)
        #print(out.shape)
                        
        #raise Exception("UNDER CONSTRUCTION!")

        return out
    


# Dataset for DataLoader (items are pairs of Context x 81 (time x freq.) spectrogram snippets and 0-1 (0.5) target values)
class ChordSet(Dataset):
    def __init__(self, feat_list, targ_list, context, hop_size):
        self.features = feat_list
        self.targets = targ_list
        self.context = context
        self.side = int((context-1)/2)
        self.hop_size = hop_size
        # list with snippet count per track
        self.snip_cnt = []
        # overall snippet count
        total_snip_cnt = 0
        # calculate overall number of snippets we can get from our data
        for feat in feat_list:
            cur_len = int(np.ceil((feat.shape[0] - self.side*2) / self.hop_size))
            self.snip_cnt.append(cur_len)
            total_snip_cnt += cur_len
        self.length = int(total_snip_cnt)
        super(ChordSet, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # find track which contains snipped with index [index]
        overal_pos = 0
        for idx, cnt in enumerate(self.snip_cnt):
            # check if this track contains the snipped with index [index]
            if index < overal_pos+cnt:
                break
            else:
                # if not, add the current tracks snipped count to the overall snipped count already visited
                overal_pos += cnt

        # calculate the position of the snipped within the track nr. [idx]
        position = index-overal_pos
        position += self.side

        # get snipped and target
        sample = self.features[idx][(position-self.side):(position+self.side+1), :]        
        target = self.targets[idx][position] # self.targets[idx][position, :]        
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
        loss = loss_func(output, target)
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
            unseen_loss += unseen_loss_func(output, target).item() # sum up batch loss

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
    
    in_shape = data.shape
    side = int((context-1)/2)
    outlen = in_shape[0] - 2*side
    output = np.empty(outlen)
    # move data to device
    data = torch.from_numpy(data[None, None, :, :])
    data = data.to(device)
    # no gradient calculation
    with torch.no_grad():
        # iterate over input data
        for idx in range(outlen):
            # calculate output for input data
            out_vector = model(data[:, :, idx:(idx+context), :])[0]
            _, out_val = torch.max(out_vector.data, 0)            
            output[idx] = out_val

    # compensate for convolutional context and return output
    output = np.append([12 for _ in range(side)], output)
    output = np.append(output,[12 for _ in range(side)])
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
    model = ChordNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # setup our datasets for training, evaluation and testing
    kwargs = {'num_workers': 4, 'pin_memory': True} if USE_CUDA else {'num_workers': 4}
    train_loader = torch.utils.data.DataLoader(ChordSet(train_f, train_t, args.context, args.hop_size),
                                               batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(ChordSet(valid_f, valid_t, args.context, args.hop_size),
                                               batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(ChordSet(test_f, test_t, args.context, args.hop_size),
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


if TRAIN:
    run_training()


# In[ ]:


def run_prediction(test_features): 
    args = Args()
    args.context = feature_context #5
    
    torch.manual_seed(SEED)
    
    # load model
    model = ChordNet().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + '.model'))) #, map_location=torch.device('cpu')
    print('model loaded...')
    
    # calculate actual output for the test data
    results_cnn = [None for _ in range(len(test_features))]
    # iterate over test tracks
    for test_idx, cur_test_feat in enumerate(test_features):
        if test_idx % 100 == 0:
            completion = int((test_idx / len(test_features))*100)
            print(str(completion)+'% complete...')
        print('file number:', test_idx+1)
        
        # run the inference method
        result = predict(model, DEVICE, cur_test_feat, args.context)
        results_cnn[test_idx] = result

    return results_cnn


# In[ ]:


predicted = None

if PREDICT:    
    # predict chords
    if VERBOSE:
        print('predicting...')
    predicted = run_prediction(test_f)
                
    # evaluate results
    if VERBOSE:
        print('evaluating results...')
    
    p_scores = []
    r_scores = []
    f1_scores = []
    for i, pred_chord in enumerate(predicted):        
        p_scores.append(precision_score(test_t[i], pred_chord, average='micro'))
        r_scores.append(recall_score(test_t[i], pred_chord, average='micro'))
        f1_scores.append(f1_score(test_t[i], pred_chord, average='micro'))
    
    print('Precision:', np.mean(p_scores))
    print('Recall:', np.mean(r_scores))
    print('F-measure:', np.mean(f1_scores))

