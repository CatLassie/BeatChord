import random

import numpy as np
import madmom
from madmom.utils import search_files

from scripts.chord_util import parse_annotations

import scripts.tcn_chord_config as tcnc

FEATURE_CONTEXT = tcnc.FEATURE_CONTEXT
ZERO_PAD = tcnc.ZERO_PAD
FRAME_ONE_START = tcnc.FRAME_ONE_START
TRANSPOSE_FEATURES = tcnc.TRANSPOSE_FEATURES
FILTER_FEATURES = tcnc.FILTER_FEATURES
LOWER_FILTER_IDX = tcnc.LOWER_FILTER_IDX
UPPER_FILTER_IDX = tcnc.UPPER_FILTER_IDX

FPS = tcnc.FPS

FEATURE_EXT = tcnc.FEATURE_EXT
FEATURE_PATH = tcnc.FEATURE_PATH
ANNOTATION_EXT = tcnc.ANNOTATION_EXT
ANNOTATION_PATH = tcnc.ANNOTATION_PATH

TRAIN_SPLIT_POINT = tcnc.TRAIN_SPLIT_POINT
VALIDATION_SPLIT_POINT = tcnc.VALIDATION_SPLIT_POINT

SEED = tcnc.SEED
VERBOSE = tcnc.VERBOSE
MAJMIN = tcnc.MAJMIN
DISPLAY_UNIQUE_CHORDS_AND_CHORD_CONFIGS = tcnc.DISPLAY_UNIQUE_CHORDS_AND_CHORD_CONFIGS





# HELPER FUNCTIONS FOR FEATURES AND TARGETS

def zero_pad_short_features(feat_list):
    # 0 pad short features to at least fit context
    feat_list_padded = []
    for feat in feat_list:

        if len(feat) < FEATURE_CONTEXT:
            diff = FEATURE_CONTEXT - len(feat)
            left = int(np.floor(diff/2))
            right = int(np.ceil(diff/2))
            
            feat_padded = np.zeros((feat.shape[0] + diff, feat.shape[1]), np.float32)
            feat_padded[left : feat.shape[0] + left, : feat.shape[1]] = feat
            feat_list_padded.append(feat_padded)
        else:
            feat_list_padded.append(feat)
            
    return feat_list_padded

def zero_pad_short_targets(target_list):
    # 0 pad targets to at least fit context
    target_list_padded = []
    for target in target_list:

        if len(target) < FEATURE_CONTEXT:
            diff = FEATURE_CONTEXT - len(target)
            left = int(np.floor(diff/2))
            right = int(np.ceil(diff/2))
            
            target_padded = np.full((target.shape[0] + diff), 12, np.int64)
            target_padded[left : target.shape[0] + left] = target
            target_list_padded.append(target_padded)
        else:
            target_list_padded.append(target)
            
    return target_list_padded

def zero_pad_all_features(feat_list):
    # 0 pad features to start from frame 1
    feat_list_padded = []
    for feat in feat_list:

        side = int(np.floor(FEATURE_CONTEXT/2))
        feat_padded = np.zeros((feat.shape[0] + (2*side), feat.shape[1]), np.float32)
        feat_padded[side : feat.shape[0] + side, : feat.shape[1]] = feat
        feat_list_padded.append(feat_padded)

    return feat_list_padded

def zero_pad_all_targets(target_list):
    # 0 pad targets to start from frame 1
    target_list_padded = []
    for target in target_list:

        side = int(np.floor(FEATURE_CONTEXT/2))
        target_padded = np.full((target.shape[0] + (2*side)), 12, np.int64)
        target_padded[side : target.shape[0] + side] = target
        target_list_padded.append(target_padded)
 
    return target_list_padded





# NOTE: if there is an annotation that is after the last frame, ignore it
def compute_target(time_labels, num_frames):
    """
    if len(times) > 0 and np.max(times) * FPS > num_frames and VERBOSE:
        print("Maximum time is larger than number of samples - cutting times.")
        print(np.max(times)*FPS, num_frames)
    """

    target = np.zeros(num_frames, np.int64)
    for i, time_label in enumerate(time_labels):
        time_idx_start = int(np.rint(time_label[0]*FPS))
        time_idx_end = int(np.rint(time_labels[i+1][0]*FPS)) if i < len(time_labels) - 1 else num_frames

        if time_idx_end <= num_frames:
            for j in range(time_idx_start, time_idx_end):
                target[j] = time_label[1]
        else:
            print('WARNING: annotation after last frame detected!', time_idx_end)

    return target

def init_targets(annos, feats):
    targs = []
    for anno, feat in zip(annos, feats):
        targ = compute_target(anno, len(feat)) # time axis for length!
        targs.append(targ)
    return targs





# LOAD FEATURES AND ANNOTATIONS, COMPUTE TARGETS
def init_feats_annos_targets(feat_path_root, anno_path_root):
    feat_paths = search_files(feat_path_root, FEATURE_EXT)

    features = [np.load(p) for p in feat_paths]
    # librosa has time in rows, madmom is transposed! now first index is time as in madmom!
    if TRANSPOSE_FEATURES:
        features = [np.transpose(f) for f in features]

    if FILTER_FEATURES:
        features = [f[:, LOWER_FILTER_IDX : UPPER_FILTER_IDX] for f in features]

    annotations = parse_annotations(anno_path_root, ANNOTATION_EXT, MAJMIN, DISPLAY_UNIQUE_CHORDS_AND_CHORD_CONFIGS);
    targets = init_targets(annotations, features)

    assert len(features) == len(targets)
    return features, annotations, targets

def shuffle_data(features, annotations, targets):
    idxs = list(range(0, len(features)))

    # shuffle indices
    random.seed(SEED)
    random.shuffle(idxs)

    # shuffle data
    features_rand = [features[i] for i in idxs]
    targets_rand = [targets[i] for i in idxs]
    annotations_rand = [annotations[i] for i in idxs]

    return features_rand, annotations_rand, targets_rand





# MAIN FUNCTION
def init_data():

    train_f = []
    train_t = []
    train_anno = []
    valid_f = []
    valid_t = []
    valid_anno = []
    test_f = []
    test_t = []
    test_anno = []

    data_length = 0

    # init features, annotations and targets
    # features, annotations, targets = init_feats_annos_targets()
    datasets = []
    for idx, _ in enumerate(FEATURE_PATH):
        datasets.append(list(init_feats_annos_targets(FEATURE_PATH[idx], ANNOTATION_PATH[idx])))

    for idx, _ in enumerate(datasets):

        # indices: 0 - features, 1 - annotations, 2 - targets

        # 0 pad all features to start from frame 1
        if FRAME_ONE_START:
            if VERBOSE:
                print('Padded data with zeros to start from frame one!\n')
            datasets[idx][0] = zero_pad_all_features(datasets[idx][0])
            datasets[idx][2] = zero_pad_all_targets(datasets[idx][2])

        # 0 pad features that are shorter than 8193 frames
        elif ZERO_PAD:
            if VERBOSE:
                print('Padded data with zeros to match context!\n')
            datasets[idx][0] = zero_pad_short_features(datasets[idx][0])
            datasets[idx][2] = zero_pad_short_targets(datasets[idx][2])

        elif VERBOSE:
            print('Data zero padding disabled!\n')
            
        # shuffle data
        datasets[idx][0], datasets[idx][1], datasets[idx][2] = shuffle_data(datasets[idx][0], datasets[idx][1], datasets[idx][2])

        # find split indices and split data
        first_idx = int(len(datasets[idx][0])*TRAIN_SPLIT_POINT)
        second_idx = int(len(datasets[idx][0])*VALIDATION_SPLIT_POINT)

        train_f = train_f + datasets[idx][0][: first_idx]
        train_t = train_t + datasets[idx][2][: first_idx]
        valid_f = valid_f + datasets[idx][0][first_idx : second_idx]
        valid_t = valid_t + datasets[idx][2][first_idx : second_idx]
        test_f = test_f + datasets[idx][0][second_idx :]
        test_t = test_t + datasets[idx][2][second_idx :]

        train_anno = train_anno + datasets[idx][1][: first_idx]
        valid_anno = valid_anno + datasets[idx][1][first_idx : second_idx]
        test_anno = test_anno + datasets[idx][1][second_idx :]

        data_length = data_length + len(datasets[idx][0])

    if VERBOSE:
        print(data_length, 'feature spectrogram files loaded, with example shape:', datasets[idx][0][0].shape)
        print(data_length, 'feature annotation files loaded, with example shape:', datasets[idx][1][0].shape)
        print(data_length, 'targets computed, with example shape:', datasets[idx][2][0].shape)
        print(len(train_f), 'training features', len(valid_f), 'validation features and', len(test_f), 'test features')
        
    return train_f, train_t, train_anno, valid_f, valid_t, valid_anno, test_f, test_t, test_anno
