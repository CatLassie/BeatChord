#!/usr/bin/env python
# coding: utf-8

# # MTL Chord Scale

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
import scripts.mtl_chord_scale_config as csc

# feature, target, annotation initializer
from scripts.mtl_chord_scale_feat import init_data

from scripts.chord_util import labels_to_notataion_and_intervals

import mir_eval


# In[ ]:


# GLOBAL VARIABLES

# random seed
SEED = csc.SEED

# cuda configuration
USE_CUDA = csc.USE_CUDA
DEVICE = csc.DEVICE
print("CURRENT DEVICE:", DEVICE)

# paths
MODEL_NAME = csc.MODEL_NAME
MODEL_PATH = csc.MODEL_PATH
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)  
    
FPS = csc.FPS


# In[ ]:


# TRAINING PARAMETERS

num_epochs = csc.NUM_EPOCHS
lr = csc.LR
batch_size = csc.BATCH_SIZE
patience = csc.PATIENCE

feature_context = csc.FEATURE_CONTEXT
traininig_hop_size = csc.TRAINING_HOP_SIZE


# In[ ]:


# COMMAND LINE SUPPORT

# TODO:

TRAIN = csc.TRAIN
TRAIN_EXISTING = csc.TRAIN_EXISTING
PREDICT = csc.PREDICT
TRAIN_ON_CHORD = csc.TRAIN_ON_CHORD
TRAIN_ON_SCALE = csc.TRAIN_ON_SCALE
VERBOSE = csc.VERBOSE

if VERBOSE:
    print('\n---- EXECUTION STARTED ----\n')
    print('Train:', TRAIN)
    print('Train existing model:', TRAIN_EXISTING)
    print('Predict', PREDICT)
    print('Training on chord data:', TRAIN_ON_CHORD, ', training on scale data:', TRAIN_ON_SCALE)
    # print('Command line arguments:\n\n', args, '\n')


# In[ ]:


# LOAD FEATURES AND ANNOTATIONS, COMPUTE TARGETS
train_f, train_c_t, train_c_anno, train_s_t, train_s_anno, valid_f, valid_c_t, valid_c_anno, valid_s_t, valid_s_anno, test_f, test_c_t, test_c_anno, test_s_t, test_s_anno = init_data()


# In[ ]:


# NETWORK PARAMETERS

# CNN

# filters
cnn_in_size = 1
cnn_h1_size = 16 #32
cnn_h2_size = 32 #64
cnn_h3_size = 64

# kernels
cnn_k_size = 3
cnn_padding = (1,1)
cnn_max_pool_k_size = (2,2)

cnn_dropout_rate = 0.1

# FULLY CONNECTED (by using a 1d convolutional. layer)

# filters
fc_chord_h1_size = 104 #52 #156 # neurons in FC layers
fc_chord_out_size = 13 # 13 outputs for 13 classes

fc_scale_h1_size = 96 #52 #156 # neurons in FC layers
fc_scale_h2_size = 24
fc_scale_out_size = 3 # 13 outputs for 13 classes

# kernels
fc_k1_size = (18,11) #(37,22) #(6,22) #22 # something big that would correspond to an FC layer (capture all data into 1 input)
fc_k2_size = 1 # second FC layer gets input from first one, filter size is 1

# loss function
chord_loss_func = nn.CrossEntropyLoss()
chord_unseen_loss_func = nn.CrossEntropyLoss(reduction="sum")

scale_loss_func = nn.CrossEntropyLoss()
scale_unseen_loss_func = nn.CrossEntropyLoss(reduction="sum")


# In[ ]:


# CHORD NETWORK CLASS and DATA SET CLASS for DATA LOADER

class ChordScaleNet(nn.Module):
    def __init__(self):
        super(ChordScaleNet, self).__init__()
        
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
        
        self.l3 = nn.Sequential(
            nn.Conv2d(cnn_h2_size, cnn_h3_size, cnn_k_size, padding=cnn_padding),
            nn.BatchNorm2d(cnn_h3_size),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = cnn_max_pool_k_size),
            nn.Dropout2d(p = cnn_dropout_rate)
        )
        
        ######## CHORD SPECIFIC LAYERS ########
        
        self.lfc1_chord = nn.Sequential(
            nn.Conv2d(cnn_h3_size, fc_chord_h1_size, fc_k1_size), # nn.Conv1d(cnn_h2_size, fc_h1_size, fc_k1_size),
            nn.BatchNorm2d(fc_chord_h1_size),
            nn.ELU(),
            nn.Dropout(p = cnn_dropout_rate)
        )
        
        self.lfcout_chord = nn.Sequential(
            nn.Conv1d(fc_chord_h1_size, fc_chord_out_size, fc_k2_size),
            #nn.Softmax(1),
        )
        
        ######## SCALE SPECIFIC LAYERS ########
        
        self.lfc1_scale = nn.Sequential(
            nn.Conv2d(cnn_h3_size, fc_scale_h1_size, fc_k1_size), # nn.Conv1d(cnn_h2_size, fc_h1_size, fc_k1_size),
            nn.BatchNorm2d(fc_scale_h1_size),
            nn.ELU(),
            nn.Dropout(p = cnn_dropout_rate)
        )
        
        self.lfc2_scale = nn.Sequential(
            nn.Conv1d(fc_scale_h1_size, fc_scale_h2_size, fc_k2_size),
            nn.BatchNorm1d(fc_scale_h2_size),
            nn.ELU(),
            nn.Dropout(p = cnn_dropout_rate)
        )
        
        self.lfcout_scale = nn.Sequential(
            nn.Conv1d(fc_scale_h2_size, fc_scale_out_size, fc_k2_size),
            #nn.Softmax(1),
        )
        
    def forward(self, x):

        #print(x.shape)

        out = self.l1(x)
        #print(out.shape)

        out = self.l2(out)
        #print(out.shape)
        
        out = self.l3(out)
        #print(out.shape)
        
        #out.squeeze_(2)
        #print(out.shape)
        
        ######## CHORD SPECIFIC LAYERS ########

        out_chord = self.lfc1_chord(out)
        #print(out.shape)
                
        out_chord.squeeze_(2)
        #print(out.shape)
            
        out_chord = self.lfcout_chord(out_chord)
        #print(out.shape)
        
        out_chord.squeeze_(-1)
        #print(out.shape)
        
        ######## SCALE SPECIFIC LAYERS ########
        
        out_scale = self.lfc1_scale(out)
        #print(out.shape)
                
        out_scale.squeeze_(2)
        #print(out.shape)
            
        out_scale = self.lfc2_scale(out_scale)
            
        out_scale = self.lfcout_scale(out_scale)
        #print(out.shape)
        
        out_scale.squeeze_(-1)
        #print(out.shape)
                        
        #raise Exception("UNDER CONSTRUCTION!")

        return out_chord, out_scale #NOTE: MAYBE THIS IS A LIST?
    


# Dataset for DataLoader (items are pairs of Context x 81 (time x freq.) spectrogram snippets and 0-1 (0.5) target values)
class ChordScaleSet(Dataset):
    def __init__(self, feat_list, targ_c_list, targ_s_list, context, hop_size):
        self.features = feat_list
        self.chord_targets = targ_c_list
        self.scale_targets = targ_s_list
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
        super(ChordScaleSet, self).__init__()

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
        chord_target = self.chord_targets[idx][position] # self.targets[idx][position, :]
        scale_target = self.scale_targets[idx][position]
        # convert to PyTorch tensor and return (unsqueeze is for extra dimension, asarray is cause target is scalar)
        return torch.from_numpy(sample).unsqueeze_(0), torch.from_numpy(np.asarray(chord_target)), torch.from_numpy(np.asarray(scale_target))



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
    for batch_idx, (data, chord_target, scale_target) in enumerate(train_loader):
        # move data to device
        data, chord_target, scale_target = data.to(device), chord_target.to(device), scale_target.to(device)
        
        # reset optimizer (clear previous gradients)
        optimizer.zero_grad()
        # forward pass (calculate output of network for input)
        chord_output, scale_output = model(data.float())
        # calculate loss        
        chord_loss = chord_loss_func(chord_output, chord_target)
        scale_loss = scale_loss_func(scale_output, scale_target)
        loss = chord_loss + scale_loss
        
        #print('chord loss:', chord_loss, scale_loss, loss)
        #raise Exception("TESTING")
        
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
        for data, chord_target, scale_target in unseen_loader:
            # move data to device
            data, chord_target, scale_target = data.to(device), chord_target.to(device), scale_target.to(device)
            # forward pass (calculate output of network for input)
            chord_output, scale_output = model(data.float())
            
            # WORK IN PROGRESS: skip rest of loop
            # continue
            
            # claculate loss and add it to our cumulative loss
            chord_unseen_loss = chord_unseen_loss_func(chord_output, chord_target)
            scale_unseen_loss = scale_unseen_loss_func(scale_output, scale_target)
            sum_unseen_loss = chord_unseen_loss + scale_unseen_loss
            unseen_loss += sum_unseen_loss.item() # sum up batch loss

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
    chord_output = np.empty(outlen)
    scale_output = np.empty(outlen)
    # move data to device
    data = torch.from_numpy(data[None, None, :, :])
    data = data.to(device)
    # no gradient calculation
    with torch.no_grad():
        # iterate over input data
        for idx in range(outlen):
            # calculate output for input data
            chord_prediction, scale_prediction = model(data[:, :, idx:(idx+context), :])
            chord_out_vector = chord_prediction[0]
            scale_out_vector = scale_prediction[0]
            
            _, chord_out_val = torch.max(chord_out_vector.data, 0)
            _, scale_out_val = torch.max(scale_out_vector.data, 0)
            chord_output[idx] = chord_out_val
            scale_output[idx] = scale_out_val

    # compensate for convolutional context and return output
    chord_output = np.append([12 for _ in range(side)], chord_output)
    chord_output = np.append(chord_output, [12 for _ in range(side)])
    
    scale_output = np.append([2 for _ in range(side)], scale_output)
    scale_output = np.append(scale_output, [2 for _ in range(side)])
    
    return chord_output, scale_output


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
    model = ChordScaleNet().to(DEVICE)
    if TRAIN_EXISTING:
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + '.model')))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # setup our datasets for training, evaluation and testing
    kwargs = {'num_workers': 4, 'pin_memory': True} if USE_CUDA else {'num_workers': 4}
    train_loader = torch.utils.data.DataLoader(ChordScaleSet(train_f, train_c_t, train_s_t, args.context, args.hop_size),
                                               batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(ChordScaleSet(valid_f, valid_c_t, valid_s_t, args.context, args.hop_size),
                                               batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(ChordScaleSet(test_f, test_c_t, test_s_t, args.context, args.hop_size),
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
    model = ChordScaleNet().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + '.model'))) #, map_location=torch.device('cpu')
    print('model loaded...')
    
    # calculate actual output for the test data
    chord_results_cnn = [None for _ in range(len(test_features))]
    scale_results_cnn = [None for _ in range(len(test_features))]
    # iterate over test tracks
    for test_idx, cur_test_feat in enumerate(test_features):
        if test_idx % 100 == 0:
            completion = int((test_idx / len(test_features))*100)
            print(str(completion)+'% complete...')
        if VERBOSE:
            print('file number:', test_idx+1)
        
        # run the inference method
        chord_result, scale_result = predict(model, DEVICE, cur_test_feat, args.context)
        chord_results_cnn[test_idx] = chord_result
        scale_results_cnn[test_idx] = scale_result

    return chord_results_cnn, scale_results_cnn


# In[ ]:


predicted = None

if PREDICT:    
    # predict chords
    if VERBOSE:
        print('predicting...')
    predicted_chords, predicted_scales = run_prediction(test_f)
                
    # evaluate results
    if VERBOSE:
        print('evaluating results...')
    
    chord_p_scores_mic = []
    chord_r_scores_mic = []
    chord_f1_scores_mic = []
    chord_p_scores_w = []
    chord_r_scores_w = []
    chord_f1_scores_w = []
    
    weighted_accuracies = []
    
    for i, pred_chord in enumerate(predicted_chords):        
        chord_p_scores_mic.append(precision_score(test_c_t[i], pred_chord, average='micro'))
        chord_r_scores_mic.append(recall_score(test_c_t[i], pred_chord, average='micro'))
        chord_f1_scores_mic.append(f1_score(test_c_t[i], pred_chord, average='micro'))

        chord_p_scores_w.append(precision_score(test_c_t[i], pred_chord, average='weighted'))
        chord_r_scores_w.append(recall_score(test_c_t[i], pred_chord, average='weighted'))
        chord_f1_scores_w.append(f1_score(test_c_t[i], pred_chord, average='weighted'))
        
        # mir_eval score (weighted accuracy)

        ref_labels, ref_intervals = labels_to_notataion_and_intervals(test_c_t[i])
        est_labels, est_intervals = labels_to_notataion_and_intervals(pred_chord)

        est_intervals, est_labels = mir_eval.util.adjust_intervals(
            est_intervals, est_labels, ref_intervals.min(),
            ref_intervals.max(), mir_eval.chord.NO_CHORD,
            mir_eval.chord.NO_CHORD)

        # print('label length before merge', len(ref_labels), len(est_labels))
        # print('interval length before merge', len(ref_intervals), len(est_intervals))
        merged_intervals, ref_labels, est_labels = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels, est_intervals, est_labels)
        # print('label length after merge', len(ref_labels), len(est_labels))
        # print('interval length after merge', len(merged_intervals))

        durations = mir_eval.util.intervals_to_durations(merged_intervals)
        comparison = mir_eval.chord.root(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparison, durations)

        weighted_accuracies.append(score)
    
    print('CHORD EVALUATION:')
    print('Precision (micro):', np.mean(chord_p_scores_mic))
    print('Recall (mico):', np.mean(chord_r_scores_mic))
    print('F-measure (micro):', np.mean(chord_f1_scores_mic))
    
    print('Precision (weighted):', np.mean(chord_p_scores_w))
    print('Recall (weighted):', np.mean(chord_r_scores_w))
    print('F-measure (weighted):', np.mean(chord_f1_scores_w))
    
    print('Weighted accuracies (mir_eval):', np.mean(weighted_accuracies))
    
    
    
    scale_p_scores_mic = []
    scale_r_scores_mic = []
    scale_f1_scores_mic = []
    scale_p_scores_w = []
    scale_r_scores_w = []
    scale_f1_scores_w = []    
    
    for i, pred_scale in enumerate(predicted_scales):        
        scale_p_scores_mic.append(precision_score(test_s_t[i], pred_scale, average='micro'))
        scale_r_scores_mic.append(recall_score(test_s_t[i], pred_scale, average='micro'))
        scale_f1_scores_mic.append(f1_score(test_s_t[i], pred_scale, average='micro'))

        scale_p_scores_w.append(precision_score(test_s_t[i], pred_scale, average='weighted'))
        scale_r_scores_w.append(recall_score(test_s_t[i], pred_scale, average='weighted'))
        scale_f1_scores_w.append(f1_score(test_s_t[i], pred_scale, average='weighted'))
             
    print('SCALE EVALUATION:')
    print('Precision (micro):', np.mean(scale_p_scores_mic))
    print('Recall (mico):', np.mean(scale_r_scores_mic))
    print('F-measure (micro):', np.mean(scale_f1_scores_mic))
    
    print('Precision (weighted):', np.mean(scale_p_scores_w))
    print('Recall (weighted):', np.mean(scale_r_scores_w))
    print('F-measure (weighted):', np.mean(scale_f1_scores_w))

