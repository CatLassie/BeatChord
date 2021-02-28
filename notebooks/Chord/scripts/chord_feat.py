import random

import numpy as np
import madmom
from madmom.utils import search_files

from scripts.chord_util import parse_annotations

import scripts.chord_config as cc

TRANSPOSE_FEATURES = cc.TRANSPOSE_FEATURES
FILTER_FEATURES = cc.FILTER_FEATURES
LOWER_FILTER_IDX = cc.LOWER_FILTER_IDX
UPPER_FILTER_IDX = cc.UPPER_FILTER_IDX

FPS = cc.FPS

FEATURE_EXT = cc.FEATURE_EXT
FEATURE_PATH = cc.FEATURE_PATH
ANNOTATION_EXT = cc.ANNOTATION_EXT
ANNOTATION_PATH = cc.ANNOTATION_PATH

TRAIN_SPLIT_POINT = cc.TRAIN_SPLIT_POINT
VALIDATION_SPLIT_POINT = cc.VALIDATION_SPLIT_POINT

SEED = cc.SEED
VERBOSE = cc.VERBOSE





# NOTE: if there is an annotation that is after the last frame, ignore it
def compute_target(time_labels, num_frames):
    target = np.zeros(num_frames, np.float32)
    for _, time_label in enumerate(time_labels):
        time_idx = int(np.rint(time_label[0]*FPS))
        if time_idx < num_frames:
            target = np.array([time_label[1] if i >= time_idx else t for (i, t) in enumerate(target)])
            
    return target

def init_targets(annos, feats):
    targs = []
    for anno, feat in zip(annos, feats):
        targ = compute_target(anno, len(feat)) # time axis for length!
        targs.append(targ)
    return targs





# LOAD FEATURES AND ANNOTATIONS, COMPUTE TARGETS
def init_feats_annos_targets(feat_path_root, anno_path_root):
    # feat_paths = search_files(feat_path_root, FEATURE_EXT)

    '''
    features = [np.load(p) for p in feat_paths]
    # librosa has time in rows, madmom is transposed! now first index is time as in madmom!
    if TRANSPOSE_FEATURES:
        features = [np.transpose(f) for f in features]

    if FILTER_FEATURES:
        features = [f[:, LOWER_FILTER_IDX : UPPER_FILTER_IDX] for f in features]
    '''

    features = [[0]*17100] * 180 # DELETE ME (180 5 minute "songs")

    annotations = parse_annotations(anno_path_root, ANNOTATION_EXT); # [madmom.io.load_beats(p) for p in anno_paths]
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
        #print(data_length, 'feature spectrogram files loaded, with example shape:', datasets[idx][0][0].shape)
        #print(data_length, 'feature annotation files loaded, with example shape:', datasets[idx][1][0].shape)
        #print(data_length, 'targets computed, with example shape:', datasets[idx][2][0].shape)
        print(len(train_f), 'training features', len(valid_f), 'validation features and', len(test_f), 'test features')

    return train_f, train_t, train_anno, valid_f, valid_t, valid_anno, test_f, test_t, test_anno
