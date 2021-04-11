import random

import numpy as np
import madmom
from madmom.utils import search_files

from scripts.chord_util import parse_annotations
from scripts.scale_util import parse_scale_annotations

import scripts.mtl_chord_scale_config as csc

TRANSPOSE_FEATURES = csc.TRANSPOSE_FEATURES
FILTER_FEATURES = csc.FILTER_FEATURES
LOWER_FILTER_IDX = csc.LOWER_FILTER_IDX
UPPER_FILTER_IDX = csc.UPPER_FILTER_IDX

FPS = csc.FPS

FEATURE_EXT = csc.FEATURE_EXT
FEATURE_PATH = csc.FEATURE_PATH
ANNOTATION_EXT = csc.ANNOTATION_EXT
ANNOTATION_PATH = csc.ANNOTATION_PATH

TRAIN_SPLIT_POINT = csc.TRAIN_SPLIT_POINT
VALIDATION_SPLIT_POINT = csc.VALIDATION_SPLIT_POINT

SEED = csc.SEED
VERBOSE = csc.VERBOSE
MAJMIN = csc.MAJMIN
DISPLAY_UNIQUE_CHORDS_AND_CHORD_CONFIGS = csc.DISPLAY_UNIQUE_CHORDS_AND_CHORD_CONFIGS





# NOTE: if there is an annotation that is after the last frame, ignore it
def compute_target(time_labels, num_frames):
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

    c_annotations = parse_annotations(anno_path_root, ANNOTATION_EXT, MAJMIN, DISPLAY_UNIQUE_CHORDS_AND_CHORD_CONFIGS);
    c_targets = init_targets(c_annotations, features)
    s_annotations = parse_scale_annotations(anno_path_root, ANNOTATION_EXT, DISPLAY_UNIQUE_CHORDS_AND_CHORD_CONFIGS);
    s_targets = init_targets(s_annotations, features)

    assert len(features) == len(c_targets)
    assert len(features) == len(s_targets)
    return features, c_annotations, c_targets, s_annotations, s_targets

def shuffle_data(features, c_annotations, c_targets, s_annotations, s_targets):
    idxs = list(range(0, len(features)))

    # shuffle indices
    random.seed(SEED)
    random.shuffle(idxs)

    # shuffle data
    features_rand = [features[i] for i in idxs]
    c_targets_rand = [c_targets[i] for i in idxs]
    c_annotations_rand = [c_annotations[i] for i in idxs]
    s_targets_rand = [s_targets[i] for i in idxs]
    s_annotations_rand = [s_annotations[i] for i in idxs]

    return features_rand, c_annotations_rand, c_targets_rand, s_annotations_rand, s_targets_rand





# MAIN FUNCTION
def init_data():

    train_f = []
    train_c_t = []
    train_c_anno = []
    train_s_t = []
    train_s_anno = []
    valid_f = []
    valid_c_t = []
    valid_c_anno = []
    valid_s_t = []
    valid_s_anno = []
    test_f = []
    test_c_t = []
    test_c_anno = []
    test_s_t = []
    test_s_anno = []

    data_length = 0

    # init features, annotations and targets
    # features, annotations, targets = init_feats_annos_targets()
    datasets = []
    for idx, _ in enumerate(FEATURE_PATH):
        datasets.append(list(init_feats_annos_targets(FEATURE_PATH[idx], ANNOTATION_PATH[idx])))

    for idx, _ in enumerate(datasets):

        # indices: 0 - features, 1 - c-annotations, 2 - c-targets, 3 - s-annotations, 4 - s-targets
            
        # shuffle data
        datasets[idx][0], datasets[idx][1], datasets[idx][2], datasets[idx][3], datasets[idx][4] = shuffle_data(datasets[idx][0], datasets[idx][1], datasets[idx][2], datasets[idx][3], datasets[idx][4])

        # find split indices and split data
        first_idx = int(len(datasets[idx][0])*TRAIN_SPLIT_POINT)
        second_idx = int(len(datasets[idx][0])*VALIDATION_SPLIT_POINT)

        train_f = train_f + datasets[idx][0][: first_idx]
        train_c_t = train_c_t + datasets[idx][2][: first_idx]
        train_s_t = train_s_t + datasets[idx][4][: first_idx]
        valid_f = valid_f + datasets[idx][0][first_idx : second_idx]
        valid_c_t = valid_c_t + datasets[idx][2][first_idx : second_idx]
        valid_s_t = valid_s_t + datasets[idx][4][first_idx : second_idx]
        test_f = test_f + datasets[idx][0][second_idx :]
        test_c_t = test_c_t + datasets[idx][2][second_idx :]
        test_s_t = test_s_t + datasets[idx][4][second_idx :]

        train_c_anno = train_c_anno + datasets[idx][1][: first_idx]
        valid_c_anno = valid_c_anno + datasets[idx][1][first_idx : second_idx]
        test_c_anno = test_c_anno + datasets[idx][1][second_idx :]

        train_s_anno = train_s_anno + datasets[idx][3][: first_idx]
        valid_s_anno = valid_s_anno + datasets[idx][3][first_idx : second_idx]
        test_s_anno = test_s_anno + datasets[idx][3][second_idx :]

        data_length = data_length + len(datasets[idx][0])

    if VERBOSE:
        print(data_length, 'feature spectrogram files loaded, with example shape:', datasets[idx][0][0].shape)
        print(data_length, 'feature chord annotation files loaded, with example shape:', datasets[idx][1][0].shape)
        print(data_length, 'chord targets computed, with example shape:', datasets[idx][2][0].shape)
        print(data_length, 'feature scale annotation files loaded, with example shape:', datasets[idx][3][0].shape)
        print(data_length, 'scale targets computed, with example shape:', datasets[idx][4][0].shape)
        print(len(train_f), 'training features', len(valid_f), 'validation features and', len(test_f), 'test features')

    return train_f, train_c_t, train_c_anno, train_s_t, train_s_anno, valid_f, valid_c_t, valid_c_anno, valid_s_t, valid_s_anno, test_f, test_c_t, test_c_anno, test_s_t, test_s_anno
