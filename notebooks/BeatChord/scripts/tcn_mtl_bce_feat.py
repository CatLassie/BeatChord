import random

import numpy as np
import madmom
from madmom.utils import search_files

from scripts.chord_util import parse_annotations

import scripts.tcn_mtl_bce_config as tmc

FEATURE_CONTEXT = tmc.FEATURE_CONTEXT
ZERO_PAD = tmc.ZERO_PAD
FRAME_ONE_START = tmc.FRAME_ONE_START
TRANSPOSE_FEATURES = tmc.TRANSPOSE_FEATURES
FILTER_FEATURES = tmc.FILTER_FEATURES
LOWER_FILTER_IDX = tmc.LOWER_FILTER_IDX
UPPER_FILTER_IDX = tmc.UPPER_FILTER_IDX

FPS = tmc.FPS

FEATURE_EXT = tmc.FEATURE_EXT
FEATURE_PATH = tmc.FEATURE_PATH
BEAT_ANNOTATION_EXT = tmc.BEAT_ANNOTATION_EXT
BEAT_ANNOTATION_PATH = tmc.BEAT_ANNOTATION_PATH
CHORD_ANNOTATION_EXT = tmc.CHORD_ANNOTATION_EXT
CHORD_ANNOTATION_PATH = tmc.CHORD_ANNOTATION_PATH

TRAIN_SPLIT_POINT = tmc.TRAIN_SPLIT_POINT
VALIDATION_SPLIT_POINT = tmc.VALIDATION_SPLIT_POINT

SEED = tmc.SEED
VERBOSE = tmc.VERBOSE
MAJMIN = tmc.MAJMIN
DISPLAY_UNIQUE_CHORDS_AND_CHORD_CONFIGS = tmc.DISPLAY_UNIQUE_CHORDS_AND_CHORD_CONFIGS





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

def zero_pad_short_targets(target_list, pad_value, pad_type):
    # 0 pad targets to at least fit context
    target_list_padded = []
    for target in target_list:

        if len(target) < FEATURE_CONTEXT:
            diff = FEATURE_CONTEXT - len(target)
            left = int(np.floor(diff/2))
            right = int(np.ceil(diff/2))
            
            target_padded = np.full((target.shape[0] + diff), pad_value, pad_type)
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

def zero_pad_all_targets(target_list, pad_value, pad_type):
    # 0 pad targets to start from frame 1
    target_list_padded = []
    for target in target_list:

        side = int(np.floor(FEATURE_CONTEXT/2))
        target_padded = np.full((target.shape[0] + (2*side)), pad_value, pad_type)
        target_padded[side : target.shape[0] + side] = target
        target_list_padded.append(target_padded)
 
    return target_list_padded





def compute_beat_target(times, num_frames):
    target = np.zeros(num_frames, np.float32)
    for idx, time in enumerate(times):
        time_idx = int(np.rint(time*FPS))
        if time_idx < num_frames:
            target[time_idx] = 1
            
            # for 0.5 probabilities on either side of beat frame
            if(time_idx > 0):
                target[time_idx-1] = 0.5
            if(time_idx < (num_frames-1)):
                target[time_idx+1] = 0.5

    return target

def init_beat_targets(annos, feats):
    targs = []
    for anno, feat in zip(annos, feats):
        targ = compute_beat_target(anno, len(feat)) # time axis for length!
        targs.append(targ)
    return targs

# NOTE: if there is an annotation that is after the last frame, ignore it
def compute_chord_target(time_labels, num_frames):
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

def init_chord_targets(annos, feats):
    targs = []
    for anno, feat in zip(annos, feats):
        targ = compute_chord_target(anno, len(feat)) # time axis for length!
        targs.append(targ)
    return targs





# LOAD FEATURES AND ANNOTATIONS, COMPUTE TARGETS
def init_feats_annos_targets(feat_path_root, beat_anno_path_root, chord_anno_path_root):
    feat_paths = search_files(feat_path_root, FEATURE_EXT)
    beat_anno_paths = search_files(beat_anno_path_root, BEAT_ANNOTATION_EXT)

    features = [np.load(p) for p in feat_paths]
    # librosa has time in rows, madmom is transposed! now first index is time as in madmom!
    if TRANSPOSE_FEATURES:
        features = [np.transpose(f) for f in features]

    if FILTER_FEATURES:
        features = [f[:, LOWER_FILTER_IDX : UPPER_FILTER_IDX] for f in features]

    b_annotations = [madmom.io.load_beats(p) for p in beat_anno_paths]
    b_targets = init_beat_targets(b_annotations, features)
    c_annotations = parse_annotations(chord_anno_path_root, CHORD_ANNOTATION_EXT, MAJMIN, DISPLAY_UNIQUE_CHORDS_AND_CHORD_CONFIGS);
    c_targets = init_chord_targets(c_annotations, features)

    assert len(features) == len(b_targets)
    assert len(features) == len(c_targets)
    return features, b_annotations, b_targets, c_annotations, c_targets

def shuffle_data(features, b_annotations, b_targets, c_annotations, c_targets):
    idxs = list(range(0, len(features)))

    # shuffle indices
    random.seed(SEED)
    random.shuffle(idxs)

    # shuffle data
    features_rand = [features[i] for i in idxs]
    b_targets_rand = [b_targets[i] for i in idxs]
    b_annotations_rand = [b_annotations[i] for i in idxs]
    c_targets_rand = [c_targets[i] for i in idxs]
    c_annotations_rand = [c_annotations[i] for i in idxs]

    return features_rand, b_annotations_rand, b_targets_rand, c_annotations_rand, c_targets_rand





# MAIN FUNCTION
def init_data():

    train_f = []
    train_b_t = []
    train_b_anno = []
    train_c_t = []
    train_c_anno = []
    valid_f = []
    valid_b_t = []
    valid_b_anno = []
    valid_c_t = []
    valid_c_anno = []
    test_f = []
    test_b_t = []
    test_b_anno = []
    test_c_t = []
    test_c_anno = []

    data_length = 0

    # init features, annotations and targets
    # features, annotations, targets = init_feats_annos_targets()
    datasets = []
    for idx, _ in enumerate(FEATURE_PATH):
        datasets.append(list(init_feats_annos_targets(FEATURE_PATH[idx], BEAT_ANNOTATION_PATH[idx], CHORD_ANNOTATION_PATH[idx])))

    for idx, _ in enumerate(datasets):

        # indices: 0 - features, 1 - b-annotations, 2 - b-targets, 3 - c-annotations, 4 - c-targets

        # 0 pad all features to start from frame 1
        if FRAME_ONE_START:
            if VERBOSE:
                print('Padded data with zeros to start from frame one!\n')
            datasets[idx][0] = zero_pad_all_features(datasets[idx][0])
            datasets[idx][2] = zero_pad_all_targets(datasets[idx][2], 0, np.float32)
            datasets[idx][4] = zero_pad_all_targets(datasets[idx][4], 12, np.int64)

        # 0 pad features that are shorter than 8193 frames
        elif ZERO_PAD:
            if VERBOSE:
                print('Padded data with zeros to match context!\n')
            datasets[idx][0] = zero_pad_short_features(datasets[idx][0])
            datasets[idx][2] = zero_pad_short_targets(datasets[idx][2], 0, np.float32)
            datasets[idx][4] = zero_pad_short_targets(datasets[idx][4], 12, np.int64)

        elif VERBOSE:
            print('Data zero padding disabled!\n')
            
        # shuffle data
        datasets[idx][0], datasets[idx][1], datasets[idx][2], datasets[idx][3], datasets[idx][4] = shuffle_data(datasets[idx][0], datasets[idx][1], datasets[idx][2], datasets[idx][3], datasets[idx][4])

        # find split indices and split data
        first_idx = int(len(datasets[idx][0])*TRAIN_SPLIT_POINT)
        second_idx = int(len(datasets[idx][0])*VALIDATION_SPLIT_POINT)

        train_f = train_f + datasets[idx][0][: first_idx]
        train_b_t = train_b_t + datasets[idx][2][: first_idx]
        train_c_t = train_c_t + datasets[idx][4][: first_idx]
        valid_f = valid_f + datasets[idx][0][first_idx : second_idx]
        valid_b_t = valid_b_t + datasets[idx][2][first_idx : second_idx]
        valid_c_t = valid_c_t + datasets[idx][4][first_idx : second_idx]
        test_f = test_f + datasets[idx][0][second_idx :]
        test_b_t = test_b_t + datasets[idx][2][second_idx :]
        test_c_t = test_c_t + datasets[idx][4][second_idx :]

        train_b_anno = train_b_anno + datasets[idx][1][: first_idx]
        valid_b_anno = valid_b_anno + datasets[idx][1][first_idx : second_idx]
        test_b_anno = test_b_anno + datasets[idx][1][second_idx :]

        train_c_anno = train_c_anno + datasets[idx][3][: first_idx]
        valid_c_anno = valid_c_anno + datasets[idx][3][first_idx : second_idx]
        test_c_anno = test_c_anno + datasets[idx][3][second_idx :]

        data_length = data_length + len(datasets[idx][0])

    if VERBOSE:
        print(data_length, 'feature spectrogram files loaded, with example shape:', datasets[idx][0][0].shape)
        print(data_length, 'beat feature annotation files loaded, with example shape:', datasets[idx][1][0].shape)
        print(data_length, 'beat targets computed, with example shape:', datasets[idx][2][0].shape)
        print(data_length, 'chord feature annotation files loaded, with example shape:', datasets[idx][3][0].shape)
        print(data_length, 'chord targets computed, with example shape:', datasets[idx][4][0].shape)
        print(len(train_f), 'training features', len(valid_f), 'validation features and', len(test_f), 'test features')
        
    return train_f, train_b_t, train_b_anno, train_c_t, train_c_anno, valid_f, valid_b_t, valid_b_anno, valid_c_t, valid_c_anno, test_f, test_b_t, test_b_anno, test_c_t, test_c_anno