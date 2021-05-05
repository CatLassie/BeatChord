#!/usr/bin/env python
# coding: utf-8

# # TCN MTL BCE

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
import scripts.mtl_2xbce_config as tmc

# feature, target, annotation initializer
from scripts.mtl_2xbce_feat import init_data, init_data_for_evaluation_only

from scripts.chord_util import labels_to_notataion_and_intervals
from scripts.chord_util import targets_to_one_hot

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

beat_loss_weight = tmc.BEAT_BCE_LOSS_WEIGHT
chord_loss_weight = tmc.CHORD_BCE_LOSS_WEIGHT


# In[ ]:


# COMMAND LINE SUPPORT

# TODO:

TRAIN = tmc.TRAIN
TRAIN_EXISTING = tmc.TRAIN_EXISTING
PREDICT = tmc.PREDICT
PREDICT_PER_DATASET = tmc.PREDICT_PER_DATASET
PREDICT_UNSEEN = tmc.PREDICT_UNSEEN
TRAIN_ON_BEAT = tmc.TRAIN_ON_BEAT
TRAIN_ON_CHORD = tmc.TRAIN_ON_CHORD
VERBOSE = tmc.VERBOSE

if VERBOSE:
    print('\n---- EXECUTION STARTED ----\n')
    print('Train:', TRAIN)
    print('Train existing model:', TRAIN_EXISTING)
    print('Predict', PREDICT)
    print('Predict per dataset', PREDICT_PER_DATASET)
    print('Predict whole unseen datasets', PREDICT_UNSEEN)
    print('Training on beat data:', TRAIN_ON_BEAT, ', training on chord data:', TRAIN_ON_CHORD)
    print('\nSelected model:', MODEL_NAME)
    # print('Command line arguments:\n\n', args, '\n')


# In[ ]:


# LOAD FEATURES AND ANNOTATIONS, COMPUTE TARGETS
train_f, train_b_t, train_b_anno, train_c_t, train_c_anno, valid_f, valid_b_t, valid_b_anno, valid_c_t, valid_c_anno, test_f, test_b_t, test_b_anno, test_c_t, test_c_anno, test_per_dataset = init_data()


# In[ ]:


train_c_t_1hot = targets_to_one_hot(train_c_t)
valid_c_t_1hot = targets_to_one_hot(valid_c_t)
test_c_t_1hot = targets_to_one_hot(test_c_t)

if VERBOSE and len(train_c_t_1hot) > 0:
    print('example of 1-hot-encoded target shape:', train_c_t_1hot[0].shape)

evaluate_only_datasets = []
if PREDICT_UNSEEN:
    evaluation_only_datasets = init_data_for_evaluation_only()


# In[ ]:


# base approach

'''
total_labels = 0
beat_occurences = 0
#class_occurences = np.zeros(14, np.float32)
for i, target in enumerate(train_c_t_1hot):
    for j, frame in enumerate(target):
        total_labels = total_labels + 1
        beat_occurences = beat_occurences + train_b_t[i][j]
        #for k, label in enumerate(frame):
        #    class_occurences[k] = class_occurences[k] + label
'''


# In[ ]:


# 2nd approach
'''
chord_occurences = 0
for i, c in enumerate(class_occurences):
    if i < 13:
        chord_occurences = chord_occurences + c

weight_values = np.full(14, chord_occurences, np.float32)
weight_values[13] = class_occurences[13]
weight_values = (total_labels - weight_values) / weight_values
'''

# 1st approach
#class_occurences[13] = beat_occurences
#weight_values = (total_labels - class_occurences) / class_occurences
#weight_values[13] = weight_values[13]*13 # to balance off 13vs1 neurons

#print(class_occurences)
#print(chord_occurences)
#print('loss weights:', weight_values)

#print(beat_occurences)
#print(total_labels)
'''
beat_weight = (total_labels - beat_occurences) / beat_occurences
weight_values = [1,1,1,1,1,1,1,1,1,1,1,1,1,beat_weight]
print('beat loss weight:', beat_weight)
#print('loss weights:', weight_values)

#calculate weights
pos_weight = torch.from_numpy(np.array(weight_values))
pos_weight = pos_weight.to(DEVICE)
pos_weight = pos_weight.unsqueeze(1)
'''


# In[ ]:


# NETWORK PARAMETERS

# CNN

LAST_CNN_KERNEL_FREQUENCY_SIZE = tmc.LAST_CNN_KERNEL_FREQUENCY_SIZE

# filters
cnn_in_size = 1
cnn_h1_size = 32
cnn_h2_size = 32
cnn_h3_size = 64
cnn_h4_size = 64

# kernels
cnn_k_1_size = 3
cnn_k_2_size = (1, LAST_CNN_KERNEL_FREQUENCY_SIZE)
cnn_padding = (1,0)
cnn_max_pool_k_size = (1,3)

cnn_dropout_rate = 0.1

# TCN

tcn_layer_num = 8 #11

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
fc_out_size = 14

# kernels
fc_k_size = 1

# loss functions
class_weight = np.full(14, chord_loss_weight, np.float32)
class_weight[13] = beat_loss_weight

print('loss weights:', class_weight, '\n')

class_weight = torch.from_numpy(class_weight)
class_weight = class_weight.to(DEVICE)
class_weight = class_weight.unsqueeze(1)

beat_loss_func = nn.BCEWithLogitsLoss(weight=class_weight[13:].squeeze(1))
unseen_beat_loss_func = nn.BCEWithLogitsLoss(weight=class_weight[13:].squeeze(1), reduction="sum")
chord_loss_func = nn.BCEWithLogitsLoss(weight=class_weight[:13])
unseen_chord_loss_func = nn.BCEWithLogitsLoss(weight=class_weight[:13], reduction="sum")


# In[ ]:


# BEAT NETWORK CLASS and DATA SET CLASS for DATA LOADER

class TCNMTLNet(nn.Module):
    def __init__(self):
        super(TCNMTLNet, self).__init__()
        
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
        
        self.l3 = nn.Sequential(
            nn.Conv2d(cnn_h2_size, cnn_h3_size, cnn_k_1_size, padding=cnn_padding),
            nn.BatchNorm2d(cnn_h3_size),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = cnn_max_pool_k_size),
            nn.Dropout2d(p = cnn_dropout_rate)
        )
        
        self.l4 = nn.Sequential(
            nn.Conv2d(cnn_h3_size, cnn_h4_size, cnn_k_2_size),
            #nn.BatchNorm2d(cnn_h3_size),
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
            #nn.Sigmoid()
        )
        
    def forward(self, x):

        # print(x.shape)

        out = self.l1(x)
        # print(out.shape)

        out = self.l2(out)
        # print(out.shape)
        out = self.l3(out)

        out = self.l4(out)
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
                
        #out_beat = out[:, 13:, :]
        #out_chord = out[:, :13, :]
        
        #print(out_beat.shape)
        #print(out_chord.shape)
        
        #out_beat = out_beat.squeeze(1)
        #out_chord = out_chord.squeeze(1)
        
        #print(out_beat.shape)
        #print(out_chord.shape)
        
        return out
    


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
        
        # probably will need to be removed when beat 2nd neuron is used
        # also b_target will need to be transposed like chord?
        #b_target_2d = np.expand_dims(b_target, axis=0)

        transposed_c_target = np.transpose(np.asarray(c_target))

        #joint_target = np.concatenate((transposed_c_target, b_target_2d))

        # convert to PyTorch tensor and return (unsqueeze is for extra dimension, asarray is cause target is scalar)
        return torch.from_numpy(sample).unsqueeze_(0), torch.from_numpy(np.asarray(b_target)), torch.from_numpy(transposed_c_target)



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
        output = model(data.float())
        b_output, c_output = output[:, 13:].squeeze(1), output[:, :13]
        # calculate loss
        loss = 0
        if TRAIN_ON_BEAT:
            #b_loss = beat_loss_func(b_output, b_target)
            #loss = loss + b_loss
            # b_target[2] = torch.from_numpy(np.full(1025, -1, np.float32)) # for testing mask
            mask = [(t[0].item() >= 0) for t in b_target]
            b_output = b_output[mask]
            b_target = b_target[mask]
            #b_mask = b_target != -1
            #b_output = b_output[b_mask].reshape(-1, args.context)
            #b_target = b_target[b_mask].reshape(-1, args.context)

            b_zip = zip(b_output, b_target)
            b_list = list(b_zip)
            b_loss_arr = [beat_loss_func(el[0], el[1]) for el in b_list]
            b_loss_mean = sum(b_loss_arr) / batch_size
            loss = loss + b_loss_mean

        if TRAIN_ON_CHORD:
            #c_loss = chord_loss_func(c_output, c_target)
            #loss = loss + c_loss            
            # c_target[2][0] = torch.from_numpy(np.full(1025, -1, np.float32)) # for testing mask
            mask = [(t[0][0].item() >= 0) for t in c_target]
            c_output = c_output[mask]
            c_target = c_target[mask]
            #c_mask = c_target != -1
            #c_output = c_output[c_mask].reshape(-1, 13, args.context)
            #c_target = c_target[c_mask].reshape(-1, 13, args.context)

            c_zip = zip(c_output, c_target)
            c_list = list(c_zip)
            c_loss_arr = [chord_loss_func(el[0], el[1]) for el in c_list]
            c_loss_mean = sum(c_loss_arr) / batch_size
            loss = loss + c_loss_mean
               
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
            output = model(data.float())
            b_output, c_output = output[:, 13:].squeeze(1), output[:, :13]
            # claculate loss and add it to our cumulative loss
            sum_unseen_loss = 0
            if TRAIN_ON_BEAT:
                #b_unseen_loss = unseen_beat_loss_func(b_output, b_target)
                #sum_unseen_loss = sum_unseen_loss + b_unseen_loss
                mask = [(t[0].item() >= 0) for t in b_target]
                b_output = b_output[mask]
                b_target = b_target[mask]

                b_zip = zip(b_output, b_target)
                b_list = list(b_zip)
                b_loss_arr = [beat_loss_func(el[0], el[1]) for el in b_list]
                b_loss_sum = sum(b_loss_arr)
                sum_unseen_loss = sum_unseen_loss + b_loss_sum

            if TRAIN_ON_CHORD:
                #c_unseen_loss = unseen_chord_loss_func(c_output, c_target)
                #sum_unseen_loss = sum_unseen_loss + c_unseen_loss
                mask = [(t[0][0].item() >= 0) for t in c_target]
                c_output = c_output[mask]
                c_target = c_target[mask]

                c_zip = zip(c_output, c_target)
                c_list = list(c_zip)
                c_loss_arr = [chord_loss_func(el[0], el[1]) for el in c_list]
                c_loss_sum = sum(c_loss_arr)
                sum_unseen_loss = sum_unseen_loss + c_loss_sum
                
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
        output = model(data.float())

        sgm = nn.Sigmoid()
        #smx = nn.Softmax(dim=1)
        output = sgm(output)

        output_beat = output[:, 13:]
        output_chord = output[:, :13]
        
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
    train_loader = torch.utils.data.DataLoader(TCNMTLSet(train_f, train_b_t, train_c_t_1hot, args.context, args.hop_size),
                                               batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(TCNMTLSet(valid_f, valid_b_t, valid_c_t_1hot, args.context, args.hop_size),
                                               batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(TCNMTLSet(test_f, test_b_t, test_c_t_1hot, args.context, args.hop_size),
                                              batch_size=args.batch_size, shuffle=False, **kwargs)

    # main training loop
    best_validation_loss = 9999999
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
    #print('model loaded...')
    
    # calculate actual output for the test data
    b_results_cnn = [None for _ in range(len(test_features))]
    c_results_cnn = [None for _ in range(len(test_features))]
    # iterate over test tracks
    for test_idx, cur_test_feat in enumerate(test_features):
        if test_idx % 100 == 0:
            completion = int((test_idx / len(test_features))*100)
            #print(str(completion)+'% complete...')
        if VERBOSE:
            #print('file number:', test_idx+1)
            pass
        
        # run the inference method
        b_result, c_result = predict(model, DEVICE, cur_test_feat, args.context)
        b_results_cnn[test_idx] = b_result.cpu().numpy()
        c_results_cnn[test_idx] = c_result.cpu().numpy()

    return b_results_cnn, c_results_cnn


# In[ ]:


def evaluate(feats, c_targs, b_annos):
    # predict beats and chords
    if VERBOSE:
        #print('predicting...')
        pass
    predicted_beats, predicted_chords = run_prediction(feats) #[test_t[0], test_t[1]]
                    
    # evaluate results
    if VERBOSE:
        #print('evaluating results...')
        pass
        
    #### CHORDS ################
        
    chord_p_scores_mic = []
    chord_r_scores_mic = []
    chord_f1_scores_mic = []
    chord_p_scores_w = []
    chord_r_scores_w = []
    chord_f1_scores_w = []
    
    chord_weighted_accuracies = []
    
    for i, pred_chord in enumerate(predicted_chords):        
        
        if c_targs[i][0] != -1:

            pred_chord = pred_chord.squeeze(0) # squeeze cause the dimensions are (1, frame_num, cause of the batch)!!!
            
            chord_p_scores_mic.append(precision_score(c_targs[i], pred_chord, average='micro'))
            chord_r_scores_mic.append(recall_score(c_targs[i], pred_chord, average='micro'))
            chord_f1_scores_mic.append(f1_score(c_targs[i], pred_chord, average='micro'))

            chord_p_scores_w.append(precision_score(c_targs[i], pred_chord, average='weighted'))
            chord_r_scores_w.append(recall_score(c_targs[i], pred_chord, average='weighted'))
            chord_f1_scores_w.append(f1_score(c_targs[i], pred_chord, average='weighted'))
            
            # mir_eval score (weighted accuracy)

            ref_labels, ref_intervals = labels_to_notataion_and_intervals(c_targs[i])
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
    
    #### BEATS ################    
    
    picked_beats = []
    
    # beat_picker = BeatTrackingProcessor(fps=FPS) # TODO: replace with OnsetPeakPickingProcessor(fps=FPS)
    beat_picker = OnsetPeakPickingProcessor(fps=FPS, threshold=THRESHOLD, pre_avg=PRE_AVG, post_avg=POST_AVG, pre_max=PRE_MAX, post_max=POST_MAX) # TODO: replace with OnsetPeakPickingProcessor(fps=FPS)
            
    for i, pred_beat in enumerate(predicted_beats):

        pred_beat = pred_beat.squeeze(0).squeeze(0)

        picked = beat_picker(pred_beat) # squeeze cause the dimensions are (1, frame_num, cause of the batch)!!!
        picked_beats.append(picked)
                
    evals = []
    for i, beat in enumerate(picked_beats):
        if b_annos[i] is not None:
            e = madmom.evaluation.beats.BeatEvaluation(beat, b_annos[i])
            evals.append(e)
            
    return evals, chord_p_scores_mic, chord_r_scores_mic, chord_f1_scores_mic, chord_p_scores_w, chord_r_scores_w, chord_f1_scores_w, chord_weighted_accuracies

def display_results(beat_eval, p_m, r_m, f_m, p_w, r_w, f_w, mireval_acc):
    print('\nCHORD EVALUATION:')
    
    #print('Precision (micro):', np.mean(p_m) if len(p_m) > 0 else 'no annotations provided!')
    #print('Recall (mico):', np.mean(r_m) if len(r_m) > 0 else 'no annotations provided!')
    print('F-measure (micro):', np.mean(f_m) if len(f_m) > 0 else 'no annotations provided!')
    
    #print('Precision (weighted):', np.mean(p_w) if len(p_w) > 0 else 'no annotations provided!')
    #print('Recall (weighted):', np.mean(r_w) if len(r_w) > 0 else 'no annotations provided!')
    print('F-measure (weighted):', np.mean(f_w) if len(f_w) > 0 else 'no annotations provided!')
    
    print('Weighted accuracies (mir_eval):', np.mean(mireval_acc) if len(mireval_acc) > 0 else 'no annotations provided!')

    print('\nBEAT EVALUATION:')

    mean_beat_eval = madmom.evaluation.beats.BeatMeanEvaluation(beat_eval) if len(beat_eval) > 0 else 'no annotations provided!'
    print(mean_beat_eval)

if PREDICT:
    beat_eval, p_m, r_m, f_m, p_w, r_w, f_w, mireval_acc = evaluate(test_f, test_c_t, test_b_anno)
    print('\nOverall results:')
    display_results(beat_eval, p_m, r_m, f_m, p_w, r_w, f_w, mireval_acc)

if PREDICT_PER_DATASET:
    print('\nResults by dataset:')
    for i, s in enumerate(test_per_dataset):
        print('\nDATASET:', s['path'])
        beat_eval, p_m, r_m, f_m, p_w, r_w, f_w, mireval_acc = evaluate(s['feat'], s['c_targ'], s['b_anno'])
        display_results(beat_eval, p_m, r_m, f_m, p_w, r_w, f_w, mireval_acc)

if PREDICT_UNSEEN:
    print('\nResults for evaluation only datasets:')
    for i, s in enumerate(evaluation_only_datasets):
        print('\nDATASET:', s['path'])
        beat_eval, p_m, r_m, f_m, p_w, r_w, f_w, mireval_acc = evaluate(s['feat'], s['c_targ'], s['b_anno'])
        display_results(beat_eval, p_m, r_m, f_m, p_w, r_w, f_w, mireval_acc)