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
from madmom.features.beats import DBNBeatTrackingProcessor # USE THIS!!!!!
from madmom.evaluation.beats import BeatEvaluation
from madmom.evaluation.beats import BeatMeanEvaluation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as Dataset
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# configurations
import scripts.mtl_8fold_config as tmc

# feature, target, annotation initializer
from scripts.mtl_8fold_feat import init_data, datasets_to_splits, init_data_for_evaluation_only

from scripts.chord_util import labels_to_notataion_and_intervals, annos_to_labels_and_intervals
from scripts.chord_util import labels_to_qualities_and_intervals, labels_to_majmin_and_intervals, annos_to_qualities_and_intervals
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

#datasets
FEATURE_PATH = tmc.FEATURE_PATH
EVAL_FEATURE_PATH = tmc.EVAL_FEATURE_PATH
DATASET_NUM = tmc.DATASET_NUM
EVAL_DATASET_NUM = tmc.EVAL_DATASET_NUM
BEAT_ANNOTATION_PATH = tmc.BEAT_ANNOTATION_PATH
EVAL_BEAT_ANNOTATION_PATH = tmc.EVAL_BEAT_ANNOTATION_PATH
CHORD_ANNOTATION_PATH = tmc.CHORD_ANNOTATION_PATH
EVAL_CHORD_ANNOTATION_PATH = tmc.EVAL_CHORD_ANNOTATION_PATH

# paths
MODEL_NAME = tmc.MODEL_NAME
MODEL_PATH = tmc.MODEL_PATH
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH) 

RESULTS_PATH = tmc.RESULTS_PATH
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

RESULTS_FILE_PATH = tmc.RESULTS_FILE_PATH
WRITE_TO_FILE = tmc.WRITE_TO_FILE

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
quality_loss_weight = tmc.QUALITY_BCE_LOSS_WEIGHT


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
ROOT_OUT_NUM = tmc.ROOT_OUT_NUM
QUALITY_OUT_NUM = tmc.QUALITY_OUT_NUM
CHORD_OUT_NUM = tmc.CHORD_OUT_NUM
FOLD_RANGE = tmc.FOLD_RANGE
DISPLAY_INTERMEDIATE_RESULTS = tmc.DISPLAY_INTERMEDIATE_RESULTS
USE_DBN_BEAT_TRACKER = tmc.USE_DBN_BEAT_TRACKER
VERBOSE = tmc.VERBOSE

if VERBOSE:
    print('\n---- EXECUTION STARTED ----\n')
    print('Train:', TRAIN)
    print('Train existing model:', TRAIN_EXISTING)
    print('Predict', PREDICT)
    print('Predict per dataset', PREDICT_PER_DATASET)
    print('Predict whole unseen datasets', PREDICT_UNSEEN)
    print('Training on beat data:', TRAIN_ON_BEAT, ', training on chord data:', TRAIN_ON_CHORD)
    print('Cross-validation range is:', FOLD_RANGE)
    print('\nSelected model:', MODEL_NAME)
    if USE_DBN_BEAT_TRACKER:
        print('\nWARNING: using DBN Beat Tracker!')
    # print('Command line arguments:\n\n', args, '\n')


# In[ ]:


# LOAD FEATURES AND ANNOTATIONS, COMPUTE TARGETS
evaluate_only_datasets = []
if PREDICT_UNSEEN:
    evaluation_only_datasets = init_data_for_evaluation_only()

datasets, test_per_dataset = init_data()


# In[ ]:


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
fc_out_size = CHORD_OUT_NUM + 1

# kernels
fc_k_size = 1

# loss functions
class_weight = np.full(CHORD_OUT_NUM + 1, chord_loss_weight, np.float32)
class_weight[CHORD_OUT_NUM] = beat_loss_weight
for i in range(ROOT_OUT_NUM, CHORD_OUT_NUM):
    class_weight[i] = quality_loss_weight

print('loss weights:', class_weight, '\n')

class_weight = torch.from_numpy(class_weight)
class_weight = class_weight.to(DEVICE)
class_weight = class_weight.unsqueeze(1)


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
    def __init__(self, feat_list, targ_b_list, targ_r_list, targ_q_list, context, hop_size):
        self.features = feat_list
        self.b_targets = targ_b_list
        self.r_targets = targ_r_list
        self.q_targets = targ_q_list
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
        r_target = self.r_targets[idx][position : position+self.context]
        q_target = self.q_targets[idx][position : position+self.context]
        
        # probably will need to be removed when beat 2nd neuron is used
        # also b_target will need to be transposed like chord?
        #b_target_2d = np.expand_dims(b_target, axis=0)

        transposed_r_target = np.transpose(np.asarray(r_target))
        transposed_q_target = np.transpose(np.asarray(q_target))

        #joint_target = np.concatenate((transposed_c_target, b_target_2d))

        # convert to PyTorch tensor and return (unsqueeze is for extra dimension, asarray is cause target is scalar)
        return torch.from_numpy(sample).unsqueeze_(0), torch.from_numpy(np.asarray(b_target)), torch.from_numpy(transposed_r_target), torch.from_numpy(transposed_q_target)



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
    for batch_idx, (data, b_target, r_target, q_target) in enumerate(train_loader):
        # move data to device
        data, b_target, r_target, q_target = data.to(device), b_target.to(device), r_target.to(device), q_target.to(device)
        
        # reset optimizer (clear previous gradients)
        optimizer.zero_grad()
        # forward pass (calculate output of network for input)
        output = model(data.float())
        b_output, r_output, q_output = output[:, CHORD_OUT_NUM:].squeeze(1), output[:, :ROOT_OUT_NUM], output[:, ROOT_OUT_NUM:CHORD_OUT_NUM]
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
            mask = [(t[0][0].item() >= 0) for t in r_target]
            r_output = r_output[mask]
            r_target = r_target[mask]
            #c_mask = c_target != -1
            #c_output = c_output[c_mask].reshape(-1, 13, args.context)
            #c_target = c_target[c_mask].reshape(-1, 13, args.context)

            r_zip = zip(r_output, r_target)
            r_list = list(r_zip)
            r_loss_arr = [chord_loss_func(el[0], el[1]) for el in r_list]
            r_loss_mean = sum(r_loss_arr) / batch_size
            loss = loss + r_loss_mean

        ########### QUALITY
            mask = [(t[0][0].item() >= 0) for t in q_target]
            q_output = q_output[mask]
            q_target = q_target[mask]

            q_zip = zip(q_output, q_target)
            q_list = list(q_zip)
            q_loss_arr = [quality_loss_func(el[0], el[1]) for el in q_list]
            q_loss_mean = sum(q_loss_arr) / batch_size
            loss = loss + q_loss_mean
               
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
        for data, b_target, r_target, q_target in unseen_loader:
            # move data to device
            data, b_target, r_target, q_target = data.to(device), b_target.to(device), r_target.to(device), q_target.to(device)
            # forward pass (calculate output of network for input)
            output = model(data.float())
            b_output, r_output, q_output = output[:, CHORD_OUT_NUM:].squeeze(1), output[:, :ROOT_OUT_NUM], output[:, ROOT_OUT_NUM:CHORD_OUT_NUM]
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
                mask = [(t[0][0].item() >= 0) for t in r_target]
                r_output = r_output[mask]
                r_target = r_target[mask]

                r_zip = zip(r_output, r_target)
                r_list = list(r_zip)
                r_loss_arr = [chord_loss_func(el[0], el[1]) for el in r_list]
                r_loss_sum = sum(r_loss_arr)
                sum_unseen_loss = sum_unseen_loss + r_loss_sum

            ######## QUALITY
                mask = [(t[0][0].item() >= 0) for t in q_target]
                q_output = q_output[mask]
                q_target = q_target[mask]

                q_zip = zip(q_output, q_target)
                q_list = list(q_zip)
                q_loss_arr = [quality_loss_func(el[0], el[1]) for el in q_list]
                q_loss_sum = sum(q_loss_arr)
                sum_unseen_loss = sum_unseen_loss + q_loss_sum
                
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

        output_beat = output[:, CHORD_OUT_NUM:]
        output_root = output[:, :ROOT_OUT_NUM]
        output_quality = output[:, ROOT_OUT_NUM:CHORD_OUT_NUM]
        
        _, out_root_val = torch.max(output_root.data, 1) # 0 -> batch, 1 -> 13 output neurons, 2 -> data size
        _, out_quality_val = torch.max(output_quality.data, 1) # 0 -> batch, 1 -> 13 output neurons, 2 -> data size
    return output_beat, out_root_val, out_quality_val


# In[ ]:


def run_training(fold_number):
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
    train_loader = torch.utils.data.DataLoader(TCNMTLSet(train_f, train_b_t, train_r_t_1hot, train_q_t_1hot, args.context, args.hop_size),
                                               batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(TCNMTLSet(valid_f, valid_b_t, valid_r_t_1hot, valid_q_t_1hot, args.context, args.hop_size),
                                               batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(TCNMTLSet(test_f, test_b_t, test_r_t_1hot, test_q_t_1hot, args.context, args.hop_size),
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
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME + '_split' + str(fold_number) + '.model'))
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


# In[ ]:


def run_prediction(test_features, fold_number): 
    args = Args()
    args.context = feature_context #5
    
    torch.manual_seed(SEED)
    
    # load model
    model = TCNMTLNet().to(DEVICE)
    if DEVICE == 'CPU':
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + '_split' + str(fold_number) + '.model'), map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + '_split' + str(fold_number) + '.model')))
    #print('model loaded...')
    
    # calculate actual output for the test data
    b_results_cnn = [None for _ in range(len(test_features))]
    r_results_cnn = [None for _ in range(len(test_features))]
    q_results_cnn = [None for _ in range(len(test_features))]
    # iterate over test tracks
    for test_idx, cur_test_feat in enumerate(test_features):
        if test_idx % 100 == 0:
            completion = int((test_idx / len(test_features))*100)
            #print(str(completion)+'% complete...')
        if VERBOSE:
            #print('file number:', test_idx+1)
            pass
        
        # run the inference method
        b_result, r_result, q_result = predict(model, DEVICE, cur_test_feat, args.context)
        b_results_cnn[test_idx] = b_result.cpu().numpy()
        r_results_cnn[test_idx] = r_result.cpu().numpy()
        q_results_cnn[test_idx] = q_result.cpu().numpy()

    return b_results_cnn, r_results_cnn, q_results_cnn


# In[ ]:


def evaluate(feats, r_targs, q_targs, c_annos, b_annos, fold_number):
    # predict beats and chords
    if VERBOSE:
        #print('predicting...')
        pass
    predicted_beats, predicted_roots, predicted_quals = run_prediction(feats, fold_number) #[test_t[0], test_t[1]]
                    
    # evaluate results
    if VERBOSE:
        #print('evaluating results...')
        pass
        
    #### CHORDS ################
        
    root_p_scores_mic = []
    root_r_scores_mic = []
    root_f1_scores_mic = []
    root_p_scores_w = []
    root_r_scores_w = []
    root_f1_scores_w = []

    qual_p_scores_mic = []
    qual_r_scores_mic = []
    qual_f1_scores_mic = []
    qual_p_scores_w = []
    qual_r_scores_w = []
    qual_f1_scores_w = []
    
    root_weighted_accuracies = []
    qual_weighted_accuracies = []
    majmin_weighted_accuracies = []
    
    for i, _ in enumerate(predicted_roots):        
        
        if r_targs[i][0] != -1:

            pred_r = predicted_roots[i].squeeze(0) # squeeze cause the dimensions are (1, frame_num, cause of the batch)!!!
            pred_q = predicted_quals[i].squeeze(0)
            
            root_p_scores_mic.append(precision_score(r_targs[i], pred_r, average='micro'))
            root_r_scores_mic.append(recall_score(r_targs[i], pred_r, average='micro'))
            root_f1_scores_mic.append(f1_score(r_targs[i], pred_r, average='micro'))

            root_p_scores_w.append(precision_score(r_targs[i], pred_r, average='weighted'))
            root_r_scores_w.append(recall_score(r_targs[i], pred_r, average='weighted'))
            root_f1_scores_w.append(f1_score(r_targs[i], pred_r, average='weighted'))

            qual_p_scores_mic.append(precision_score(q_targs[i], pred_q, average='micro'))
            qual_r_scores_mic.append(recall_score(q_targs[i], pred_q, average='micro'))
            qual_f1_scores_mic.append(f1_score(q_targs[i], pred_q, average='micro'))

            qual_p_scores_w.append(precision_score(q_targs[i], pred_q, average='weighted'))
            qual_r_scores_w.append(recall_score(q_targs[i], pred_q, average='weighted'))
            qual_f1_scores_w.append(f1_score(q_targs[i], pred_q, average='weighted'))
            
            # mir_eval score (weighted accuracy)

            #ref_labels, ref_intervals = labels_to_notataion_and_intervals(c_targs[i])
            ref_labels, ref_intervals = annos_to_labels_and_intervals(c_annos[i], pred_r)

            # ROOTS
            r_est_labels, r_est_intervals = labels_to_notataion_and_intervals(pred_r)

            r_est_intervals, r_est_labels = mir_eval.util.adjust_intervals(
                r_est_intervals, r_est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)

            # print('label length before merge', len(ref_labels), len(est_labels))
            # print('interval length before merge', len(ref_intervals), len(est_intervals))
            r_merged_intervals, r_ref_labels, r_est_labels = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels, r_est_intervals, r_est_labels)
            # print('label length after merge', len(ref_labels), len(est_labels))
            # print('interval length after merge', len(merged_intervals))

            r_durations = mir_eval.util.intervals_to_durations(r_merged_intervals)
            
            # TODO: 1 comparison for root, 1 for quality with set root (majmin), 1 for majmin
            r_comparison = mir_eval.chord.root(r_ref_labels, r_est_labels)
            # comparison = mir_eval.chord.majmin(ref_labels, est_labels)

            r_score = mir_eval.chord.weighted_accuracy(r_comparison, r_durations)

            root_weighted_accuracies.append(r_score)

            # QUALITIES
            q_ref_labels, q_ref_intervals = annos_to_qualities_and_intervals(c_annos[i], pred_r)
            q_est_labels, q_est_intervals = labels_to_qualities_and_intervals(pred_q)

            q_est_intervals, q_est_labels = mir_eval.util.adjust_intervals(
                q_est_intervals, q_est_labels, q_ref_intervals.min(),
                q_ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)

            q_merged_intervals, q_ref_labels, q_est_labels = mir_eval.util.merge_labeled_intervals(q_ref_intervals, q_ref_labels, q_est_intervals, q_est_labels)
            q_durations = mir_eval.util.intervals_to_durations(q_merged_intervals)
            q_comparison = mir_eval.chord.majmin(q_ref_labels, q_est_labels)
            q_score = mir_eval.chord.weighted_accuracy(q_comparison, q_durations)

            qual_weighted_accuracies.append(q_score)

            # MAJMIN
            mm_est_labels, mm_est_intervals = labels_to_majmin_and_intervals(pred_r, pred_q)

            mm_est_intervals, mm_est_labels = mir_eval.util.adjust_intervals(
                mm_est_intervals, mm_est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)

            mm_merged_intervals, mm_ref_labels, mm_est_labels = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels, mm_est_intervals, mm_est_labels)
            mm_durations = mir_eval.util.intervals_to_durations(mm_merged_intervals)
            mm_comparison = mir_eval.chord.majmin(mm_ref_labels, mm_est_labels)
            mm_score = mir_eval.chord.weighted_accuracy(mm_comparison, mm_durations)

            majmin_weighted_accuracies.append(mm_score)
    
    #### BEATS ################    
    
    picked_beats = []
    
    if USE_DBN_BEAT_TRACKER:
        beat_picker = DBNBeatTrackingProcessor(fps=FPS) # TODO: replace with OnsetPeakPickingProcessor(fps=FPS)
    else:
        beat_picker = OnsetPeakPickingProcessor(fps=FPS, threshold=THRESHOLD, pre_avg=PRE_AVG, post_avg=POST_AVG, pre_max=PRE_MAX, post_max=POST_MAX)
            
    for i, pred_beat in enumerate(predicted_beats):

        pred_beat = pred_beat.squeeze(0).squeeze(0)

        picked = beat_picker(pred_beat) # squeeze cause the dimensions are (1, frame_num, cause of the batch)!!!
        picked_beats.append(picked)
                
    evals = []
    for i, beat in enumerate(picked_beats):
        if b_annos[i] is not None:
            e = madmom.evaluation.beats.BeatEvaluation(beat, b_annos[i])
            evals.append(e)
            
    return evals, root_f1_scores_mic, root_f1_scores_w, root_weighted_accuracies, qual_f1_scores_mic, qual_f1_scores_w, qual_weighted_accuracies, majmin_weighted_accuracies

def display_results(beat_eval, root_f1_m, root_f1_w, root_acc, qual_f1_m, qual_f1_w, qual_acc, majmin_acc):
    if VERBOSE and DISPLAY_INTERMEDIATE_RESULTS:
        write_results('\nBEAT EVALUATION:')

        mean_beat_eval = madmom.evaluation.beats.BeatMeanEvaluation(beat_eval).tostring() if len(beat_eval) > 0 else 'no annotations provided!'
        write_results(mean_beat_eval)

        write_results('\nCHORD EVALUATION:')
        
        write_results('Root F-measure (micro): ' + str(np.mean(root_f1_m)) if len(root_f1_m) > 0 else 'no annotations provided!')
        write_results('Root F-measure (weighted): ' +  str(np.mean(root_f1_w)) if len(root_f1_w) > 0 else 'no annotations provided!')
        write_results('Root Weighted accuracies (mir_eval): ' + str(np.mean(root_acc)) if len(root_acc) > 0 else 'no annotations provided!')

        write_results('Quality F-measure (micro): ' + str(np.mean(qual_f1_m)) if len(qual_f1_m) > 0 else 'no annotations provided!')
        write_results('Quality F-measure (weighted): ' +  str(np.mean(qual_f1_w)) if len(qual_f1_w) > 0 else 'no annotations provided!')
        write_results('Quality Weighted accuracies (mir_eval): ' + str(np.mean(qual_acc)) if len(qual_acc) > 0 else 'no annotations provided!')

        write_results('Maj/min Weighted accuracies (mir_eval): ' + str(np.mean(majmin_acc)) if len(majmin_acc) > 0 else 'no annotations provided!')


def write_results(line, mode = 'a+'):
    print(line)
    if WRITE_TO_FILE:
        file_path = os.path.join(RESULTS_FILE_PATH)
        f = open(file_path, mode)
        f.write(line)
        f.write('\n')
        f.close()

dataset_beat_evaluations = [[] for p in FEATURE_PATH]
dataset_f_r_micro_evaluations = np.zeros(DATASET_NUM, np.float32)
dataset_f_r_weighted_evaluations = np.zeros(DATASET_NUM, np.float32)
dataset_r_mireval_evaluations = np.zeros(DATASET_NUM, np.float32)
dataset_f_q_micro_evaluations = np.zeros(DATASET_NUM, np.float32)
dataset_f_q_weighted_evaluations = np.zeros(DATASET_NUM, np.float32)
dataset_q_mireval_evaluations = np.zeros(DATASET_NUM, np.float32)
dataset_mm_mireval_evaluations = np.zeros(DATASET_NUM, np.float32)

unseen_dataset_beat_evaluations = [[] for p in EVAL_FEATURE_PATH]
unseen_dataset_f_micro_evaluations = np.zeros(EVAL_DATASET_NUM, np.float32)
unseen_dataset_f_weighted_evaluations = np.zeros(EVAL_DATASET_NUM, np.float32)
unseen_dataset_mireval_evaluations = np.zeros(EVAL_DATASET_NUM, np.float32)

write_results('Model name: ' + MODEL_NAME, 'w+')

for i in FOLD_RANGE:
    train_f, train_b_t, train_b_anno, train_r_t, train_q_t, train_c_anno, valid_f, valid_b_t, valid_b_anno, valid_r_t, valid_q_t, valid_c_anno, test_f, test_b_t, test_b_anno, test_r_t, test_q_t, test_c_anno, test_per_dataset = datasets_to_splits(datasets, test_per_dataset, i)

    if TRAIN:
        train_r_t_1hot = targets_to_one_hot(train_r_t, ROOT_OUT_NUM)
        valid_r_t_1hot = targets_to_one_hot(valid_r_t, ROOT_OUT_NUM)
        test_r_t_1hot = targets_to_one_hot(test_r_t, ROOT_OUT_NUM)
        train_q_t_1hot = targets_to_one_hot(train_q_t, QUALITY_OUT_NUM)
        valid_q_t_1hot = targets_to_one_hot(valid_q_t, QUALITY_OUT_NUM)
        test_q_t_1hot = targets_to_one_hot(test_q_t, QUALITY_OUT_NUM)

    if VERBOSE and TRAIN and len(train_r_t_1hot) > 0:
        print('example of 1-hot-encoded root target shape:', train_r_t_1hot[0].shape, '\n')
    if VERBOSE and TRAIN and len(train_q_t_1hot) > 0:
        print('example of 1-hot-encoded quality target shape:', train_q_t_1hot[0].shape, '\n')

    beat_loss_func = nn.BCEWithLogitsLoss(weight=class_weight[CHORD_OUT_NUM:].squeeze(1))
    unseen_beat_loss_func = nn.BCEWithLogitsLoss(weight=class_weight[CHORD_OUT_NUM:].squeeze(1), reduction="sum")
    chord_loss_func = nn.BCEWithLogitsLoss(weight=class_weight[:ROOT_OUT_NUM])
    unseen_chord_loss_func = nn.BCEWithLogitsLoss(weight=class_weight[:ROOT_OUT_NUM], reduction="sum")
    quality_loss_func = nn.BCEWithLogitsLoss(weight=class_weight[ROOT_OUT_NUM:CHORD_OUT_NUM])
    unseen_quality_loss_func = nn.BCEWithLogitsLoss(weight=class_weight[ROOT_OUT_NUM:CHORD_OUT_NUM], reduction="sum")

    if TRAIN or TRAIN_EXISTING:
        run_training(i+1)

    write_results('\n\n\n\n\n############ FOLD ' + str(i+1) + ' FINISHED ############')

    # HINT: Obsolete method to calculate metrics on all features combined
    if PREDICT:
        pass
        #beat_eval, p_m, r_m, f_m, p_w, r_w, f_w, mireval_acc = evaluate(test_f, test_c_t, test_b_anno, i+1)
        #print('\nOverall results:')
        #display_results(beat_eval, p_m, r_m, f_m, p_w, r_w, f_w, mireval_acc)

    if PREDICT_PER_DATASET:
        if DISPLAY_INTERMEDIATE_RESULTS:
            write_results('\nRESULTS FOR DATASET TEST SPLITS:')

        for j, s in enumerate(test_per_dataset):
            if DISPLAY_INTERMEDIATE_RESULTS:
                write_results('\nDATASET: ' + s['path'])

            beat_eval, root_f1_m, root_f1_w, root_acc, qual_f1_m, qual_f1_w, qual_acc, majmin_acc = evaluate(s['feat'], s['r_targ'], s['q_targ'], s['c_anno'], s['b_anno'], i+1)
            display_results(beat_eval, root_f1_m, root_f1_w, root_acc, qual_f1_m, qual_f1_w, qual_acc, majmin_acc)

            if len(beat_eval) > 0:
                dataset_beat_evaluations[j] += beat_eval
            if len(root_f1_m) > 0:
                dataset_f_r_micro_evaluations[j] += np.mean(root_f1_m)
                dataset_f_q_micro_evaluations[j] += np.mean(qual_f1_m)
            if len(root_f1_w) > 0:
                dataset_f_r_weighted_evaluations[j] += np.mean(root_f1_w)
                dataset_f_q_weighted_evaluations[j] += np.mean(qual_f1_w)
            if len(root_acc) > 0:
                dataset_r_mireval_evaluations[j] += np.mean(root_acc)
                dataset_q_mireval_evaluations[j] += np.mean(qual_acc)
                dataset_mm_mireval_evaluations[j] += np.mean(majmin_acc)

    if PREDICT_UNSEEN:
        if DISPLAY_INTERMEDIATE_RESULTS:
            write_results('\nRESULTS FOR UNSEEN COMPLETE DATASETS:')

        for j, s in enumerate(evaluation_only_datasets):
            if DISPLAY_INTERMEDIATE_RESULTS:
                write_results('\nDATASET: ' + s['path'])

            beat_eval, root_f1_m, root_f1_w, root_acc, qual_f1_m, qual_f1_w, qual_acc, majmin_acc = evaluate(s['feat'], s['r_targ'], s['q_targ'], s['c_anno'], s['b_anno'], i+1)
            display_results(beat_eval, root_f1_m, root_f1_w, root_acc, qual_f1_m, qual_f1_w, qual_acc, majmin_acc)

            if len(beat_eval) > 0:
                unseen_dataset_beat_evaluations[j] += beat_eval
            if len(root_f1_m) > 0:
                unseen_dataset_f_micro_evaluations[j] += np.mean(root_f1_m)
            if len(root_f1_w) > 0:
                unseen_dataset_f_weighted_evaluations[j] += np.mean(root_f1_w)
            if len(root_acc) > 0:
                unseen_dataset_mireval_evaluations[j] += np.mean(root_acc)

if PREDICT_PER_DATASET:
    write_results('\n\n\n\n\n############ CROSS-VALIDATION RESULTS FOR DATASET TEST SPLITS: ############')
    for i, path in enumerate(FEATURE_PATH):
        write_results('\nDATASET: ' + path + '\n')
        
        write_results('BEAT EVALUATION:')
        if BEAT_ANNOTATION_PATH[i] != None:
            write_results(madmom.evaluation.beats.BeatMeanEvaluation(dataset_beat_evaluations[i]).tostring())
        else:
            write_results('no beat annotations provided!')

        write_results('\nCHORD EVALUATION:')
        if CHORD_ANNOTATION_PATH[i] != None:
            write_results('Root F-measure (micro): ' + str(dataset_f_r_micro_evaluations[i]/len(FOLD_RANGE)))
            write_results('Root F-measure (weighted): ' + str(dataset_f_r_weighted_evaluations[i]/len(FOLD_RANGE)))
            write_results('Root Weighted accuracies (mir_eval): ' + str(dataset_r_mireval_evaluations[i]/len(FOLD_RANGE)))
            write_results('')
            write_results('Quality F-measure (micro): ' + str(dataset_f_q_micro_evaluations[i]/len(FOLD_RANGE)))
            write_results('Quality F-measure (weighted): ' + str(dataset_f_q_weighted_evaluations[i]/len(FOLD_RANGE)))
            write_results('Quality Weighted accuracies (mir_eval): ' + str(dataset_q_mireval_evaluations[i]/len(FOLD_RANGE)))
            write_results('')
            write_results('Maj/min Weighted accuracies (mir_eval): ' + str(dataset_mm_mireval_evaluations[i]/len(FOLD_RANGE)))
        else:
            write_results('no chord annotations provided!')

if PREDICT_UNSEEN:
    write_results('\n\n\n\n\n############ CROSS-VALIDATION RESULTS FOR COMPLETE UNSEEN DATASETS: ############')
    for i, path in enumerate(EVAL_FEATURE_PATH):
        write_results('\nDATASET: ' + path + '\n')

        write_results('BEAT EVALUATION:')
        if EVAL_BEAT_ANNOTATION_PATH[i] != None:
            write_results(madmom.evaluation.beats.BeatMeanEvaluation(unseen_dataset_beat_evaluations[i]).tostring())
        else:
            write_results('no beat annotations provided!')

        write_results('\nCHORD EVALUATION:')
        if EVAL_CHORD_ANNOTATION_PATH[i] != None:
            write_results('F-measure (micro): ' + str(unseen_dataset_f_micro_evaluations[i]/len(FOLD_RANGE)))
            write_results('F-measure (weighted): ' + str(unseen_dataset_f_weighted_evaluations[i]/len(FOLD_RANGE)))
            write_results('Weighted accuracies (mir_eval): ' + str(unseen_dataset_mireval_evaluations[i]/len(FOLD_RANGE)))
        else:
            write_results('no chord annotations provided!')
