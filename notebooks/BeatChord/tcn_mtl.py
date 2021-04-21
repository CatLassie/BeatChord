#!/usr/bin/env python
# coding: utf-8

# # TCN MTL

# In[ ]:


# IMPORTS

import os
import time

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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# configurations
import scripts.tcn_mtl_config as tmc

# feature, target, annotation initializer
from scripts.tcn_mtl_feat import init_data

from scripts.chord_util import labels_to_notataion_and_intervals

import mir_eval


# In[ ]:


# GLOBAL VARIABLES

# random seed
SEED = tmc.SEED

# cuda configuration
USE_CUDA = tmc.USE_CUDA
DEVICE = tmc.DEVICE
print("CURRENT DEVICE:", DEVICE)

# paths
MODEL_NAME = tmc.MODEL_NAME
MODEL_PATH = tmc.MODEL_PATH
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)  
    
FPS = tmc.FPS

# peak picker params
THRESHOLD = tmc.THRESHOLD
PRE_AVG = tmc.PRE_AVG
POST_AVG = tmc.POST_AVG
PRE_MAX = tmc.PRE_MAX
POST_MAX = tmc.POST_MAX


# In[ ]:


# TRAINING PARAMETERS

num_epochs = tmc.NUM_EPOCHS
lr = tmc.LR

feature_context = tmc.FEATURE_CONTEXT
traininig_hop_size = tmc.TRAINING_HOP_SIZE

batch_size = tmc.BATCH_SIZE
patience = tmc.PATIENCE


# In[ ]:


# COMMAND LINE SUPPORT

# TODO:

TRAIN = tmc.TRAIN
TRAIN_EXISTING = tmc.TRAIN_EXISTING
PREDICT = tmc.PREDICT
TRAIN_ON_BEAT = tmc.TRAIN_ON_BEAT
TRAIN_ON_CHORD = tmc.TRAIN_ON_CHORD
VERBOSE = tmc.VERBOSE

if VERBOSE:
    print('\n---- EXECUTION STARTED ----\n')
    print('Train:', TRAIN)
    print('Train existing model:', TRAIN_EXISTING)
    print('Predict', PREDICT)
    print('Training on beat data:', TRAIN_ON_BEAT, ', training on chord data:', TRAIN_ON_CHORD)
    # print('Command line arguments:\n\n', args, '\n')


# In[ ]:


# LOAD FEATURES AND ANNOTATIONS, COMPUTE TARGETS
train_f, train_b_t, train_b_anno, train_c_t, train_c_anno, valid_f, valid_b_t, valid_b_anno, valid_c_t, valid_c_anno, test_f, test_b_t, test_b_anno, test_c_t, test_c_anno = init_data()


# In[ ]:


# NETWORK PARAMETERS

# CNN

LAST_CNN_KERNEL_FREQUENCY_SIZE = tmc.LAST_CNN_KERNEL_FREQUENCY_SIZE

# filters
cnn_in_size = 1
cnn_h_size = 16

# kernels
cnn_k_1_size = 3
cnn_k_2_size = (1, LAST_CNN_KERNEL_FREQUENCY_SIZE)
cnn_padding = (1,0)
cnn_max_pool_k_size = (1,3)

cnn_dropout_rate = 0.1

# TCN

tcn_layer_num = 8 #11

# filters
tcn_h_size = 16

# kernels
tcn_k_size = 5
tcn_dilations = [2**x for x in range(0, tcn_layer_num)]
tcn_paddings = [2*x for x in tcn_dilations]

tcn_dropout_rate = 0.1

# FULLY CONNECTED (by using a 1d convolutional. layer)

# filters
fc_h_size = 16
fc_out_size = 14

# kernels
fc_k_size = 1

# loss function
chord_loss_func = nn.CrossEntropyLoss()
chord_unseen_loss_func = nn.CrossEntropyLoss(reduction="sum")


# In[ ]:


# BEAT NETWORK CLASS and DATA SET CLASS for DATA LOADER

class TCNMTLNet(nn.Module):
    def __init__(self):
        super(TCNMTLNet, self).__init__()
        
        self.l1 = nn.Sequential(
            nn.Conv2d(cnn_in_size, cnn_h_size, cnn_k_1_size, padding=cnn_padding),
            nn.BatchNorm2d(cnn_h_size),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = cnn_max_pool_k_size),
            nn.Dropout2d(p = cnn_dropout_rate)
        )
        
        self.l2 = nn.Sequential(
            nn.Conv2d(cnn_h_size, cnn_h_size, cnn_k_1_size, padding=cnn_padding),
            nn.BatchNorm2d(cnn_h_size),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = cnn_max_pool_k_size),
            nn.Dropout2d(p = cnn_dropout_rate)
        )
        
        self.l3 = nn.Sequential(
            nn.Conv2d(cnn_h_size, cnn_h_size, cnn_k_2_size),
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
        
        self.lfc = nn.Sequential(
            nn.Conv1d(fc_h_size, fc_out_size, fc_k_size),
            # nn.Sigmoid()
        )
        
        self.activationSigmoid = nn.Sequential(
            nn.Sigmoid()
        )
        
    def forward(self, x):

        # print(x.shape)

        out = self.l1(x)
        # print(out.shape)

        out = self.l2(out)
        # print(out.shape)

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
        # print(out.shape)
        
        out = self.lfc(out)
        # print(out.shape)
                
        out_beat = out[:, 13:, :]
        out_chord = out[:, :13, :]
        
        out_beat = self.activationSigmoid(out_beat)
        
        #print(out_beat.shape)
        #print(out_chord.shape)
        
        out_beat = out_beat.squeeze(1)
        #out_chord = out_chord.squeeze(1)
        
        #print(out_beat.shape)
        #print(out_chord.shape)
        
        return out_beat, out_chord
    


# Dataset for DataLoader (items are pairs of Context x 81 (time x freq.) spectrogram snippets and 0-1 (0.5) target values)
class TCNMTLSet(Dataset):
    def __init__(self, feat_list, targ_b_list, targ_c_list, context, hop_size):
        self.features = feat_list
        self.b_targets = targ_b_list
        self.c_targets = targ_c_list
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
                cur_len = 0
                self.snip_cnt.append(cur_len)
                total_snip_cnt += cur_len 
                print("warning: skipped 1 example, shape", feat.shape[0])

        self.length = int(total_snip_cnt)
        super(TCNMTLSet, self).__init__()

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
        
        sample = self.features[idx][position : position+self.context]
        b_target = self.b_targets[idx][position : position+self.context]
        c_target = self.c_targets[idx][position : position+self.context]
        # convert to PyTorch tensor and return (unsqueeze is for extra dimension, asarray is cause target is scalar)
        return torch.from_numpy(sample).unsqueeze_(0), torch.from_numpy(np.asarray(b_target)), torch.from_numpy(np.asarray(c_target))



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
    for batch_idx, (data, b_target, c_target) in enumerate(train_loader):
        # move data to device
        data, b_target, c_target = data.to(device), b_target.to(device), c_target.to(device)
        
        # reset optimizer (clear previous gradients)
        optimizer.zero_grad()
        # forward pass (calculate output of network for input)
        b_output, c_output = model(data.float())
        # calculate loss        
        loss = 0
        if TRAIN_ON_BEAT:
            b_loss = F.binary_cross_entropy(b_output, b_target)
            loss = loss + b_loss
        if TRAIN_ON_CHORD:
            c_loss = chord_loss_func(c_output, c_target)
            loss = loss + c_loss
        
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
        for data, b_target, c_target in unseen_loader:
            # move data to device
            data, b_target, c_target = data.to(device), b_target.to(device), c_target.to(device)
            # forward pass (calculate output of network for input)
            b_output, c_output = model(data.float())
            
            # WORK IN PROGRESS: skip rest of loop
            # continue
            
            # claculate loss and add it to our cumulative loss
            sum_unseen_loss = 0
            if TRAIN_ON_BEAT:
                b_unseen_loss = F.binary_cross_entropy(b_output, b_target, reduction='sum')
                sum_unseen_loss = sum_unseen_loss + b_unseen_loss
            if TRAIN_ON_CHORD:
                c_unseen_loss = chord_unseen_loss_func(c_output, c_target)
                sum_unseen_loss = sum_unseen_loss + c_unseen_loss
                
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
    output = None
    # move data to device
    data = torch.from_numpy(data[None, None, :, :])
    data = data.to(device)
    # no gradient calculation
    with torch.no_grad():
        output_beat, output_chord = model(data.float())
        _, out_chord_val = torch.max(output_chord.data, 1) # 0 -> batch, 1 -> 13 output neurons, 2 -> data size
    return output_beat, out_chord_val


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
    model = TCNMTLNet().to(DEVICE)
    if TRAIN_EXISTING:
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + '.model')))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # setup our datasets for training, evaluation and testing
    kwargs = {'num_workers': 4, 'pin_memory': True} if USE_CUDA else {'num_workers': 4}
    train_loader = torch.utils.data.DataLoader(TCNMTLSet(train_f, train_b_t, train_c_t, args.context, args.hop_size),
                                               batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(TCNMTLSet(valid_f, valid_b_t, valid_c_t, args.context, args.hop_size),
                                               batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(TCNMTLSet(test_f, test_b_t, test_c_t, args.context, args.hop_size),
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
    model = TCNMTLNet().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + '.model')))
    print('model loaded...')
    
    # calculate actual output for the test data
    b_results_cnn = [None for _ in range(len(test_features))]
    c_results_cnn = [None for _ in range(len(test_features))]
    # iterate over test tracks
    for test_idx, cur_test_feat in enumerate(test_features):
        if test_idx % 100 == 0:
            completion = int((test_idx / len(test_features))*100)
            print(str(completion)+'% complete...')
        if VERBOSE:
            print('file number:', test_idx+1)
        
        # run the inference method
        b_result, c_result = predict(model, DEVICE, cur_test_feat, args.context)
        b_results_cnn[test_idx] = b_result.cpu().numpy()
        c_results_cnn[test_idx] = c_result.cpu().numpy()

    return b_results_cnn, c_results_cnn


# In[ ]:


if PREDICT:
    # predict beats and chords
    if VERBOSE:
        print('predicting...')
    predicted_beats, predicted_chords = run_prediction(test_f) #[test_t[0], test_t[1]]
                    
    # evaluate results
    if VERBOSE:
        print('evaluating results...')
        
    #### CHORDS ################
        
    chord_p_scores_mic = []
    chord_r_scores_mic = []
    chord_f1_scores_mic = []
    chord_p_scores_w = []
    chord_r_scores_w = []
    chord_f1_scores_w = []
    
    chord_weighted_accuracies = []
    
    for i, pred_chord in enumerate(predicted_chords):        
        
        pred_chord = pred_chord.squeeze(0) # squeeze cause the dimensions are (1, frame_num, cause of the batch)!!!
        
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

        chord_weighted_accuracies.append(score)
    
    print('\nCHORD EVALUATION:')
    
    print('Precision (micro):', np.mean(chord_p_scores_mic))
    print('Recall (mico):', np.mean(chord_r_scores_mic))
    print('F-measure (micro):', np.mean(chord_f1_scores_mic))
    
    print('Precision (weighted):', np.mean(chord_p_scores_w))
    print('Recall (weighted):', np.mean(chord_r_scores_w))
    print('F-measure (weighted):', np.mean(chord_f1_scores_w))
    
    print('Weighted accuracies (mir_eval):', np.mean(chord_weighted_accuracies))
    
    #### BEATS ################
    
    print('\nBEAT EVALUATION:')
    
    picked_beats = []
    
    # beat_picker = BeatTrackingProcessor(fps=FPS) # TODO: replace with OnsetPeakPickingProcessor(fps=FPS)
    beat_picker = OnsetPeakPickingProcessor(fps=FPS, threshold=THRESHOLD, pre_avg=PRE_AVG, post_avg=POST_AVG, pre_max=PRE_MAX, post_max=POST_MAX) # TODO: replace with OnsetPeakPickingProcessor(fps=FPS)
    
    # pick peaks
    if VERBOSE:
        print('\npicking beats...')
        
    for i, pred_beat in enumerate(predicted_beats):
        picked = beat_picker(pred_beat.squeeze(0)) # squeeze cause the dimensions are (1, frame_num, cause of the batch)!!!
        picked_beats.append(picked)
        
    if VERBOSE:
        print('\n')
        
    evals = []
    for i, beat in enumerate(picked_beats):
        e = madmom.evaluation.beats.BeatEvaluation(beat, test_b_anno[i])
        evals.append(e)
        
    mean_eval = madmom.evaluation.beats.BeatMeanEvaluation(evals)
    print(mean_eval)

