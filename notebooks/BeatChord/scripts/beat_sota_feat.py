import random

import numpy as np
import madmom
from madmom.utils import search_files
import scripts.beat_sota_config as bsc

FEATURE_CONTEXT = bsc.FEATURE_CONTEXT
ZERO_PAD = bsc.ZERO_PAD
FRAME_ONE_START = bsc.FRAME_ONE_START
TRANSPOSE_FEATURES = bsc.TRANSPOSE_FEATURES
FILTER_FEATURES = bsc.FILTER_FEATURES
LOWER_FILTER_IDX = bsc.LOWER_FILTER_IDX
UPPER_FILTER_IDX = bsc.UPPER_FILTER_IDX

FPS = bsc.FPS

FEATURE_EXT = bsc.FEATURE_EXT
FEATURE_PATH = bsc.FEATURE_PATH
ANNOTATION_EXT = bsc.ANNOTATION_EXT
ANNOTATION_PATH = bsc.ANNOTATION_PATH

TRAIN_SPLIT_POINT = bsc.TRAIN_SPLIT_POINT
VALIDATION_SPLIT_POINT = bsc.VALIDATION_SPLIT_POINT

SEED = bsc.SEED
VERBOSE = bsc.VERBOSE





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
            
            target_padded = np.zeros((target.shape[0] + diff), np.float32)
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
        target_padded = np.zeros((target.shape[0] + (2*side)), np.float32)
        target_padded[side : target.shape[0] + side] = target
        target_list_padded.append(target_padded)
 
    return target_list_padded





# compute a target array for 1 feature (1 (0.5 on neighbouring frames) for beat, 0 for non-beat in each frame)
# NOTE: if there is an annotation that is after the last frame, ignore it
def compute_target(times, num_frames):
    """
    if len(times) > 0 and np.max(times) * FPS > num_frames and VERBOSE:
        print("Maximum time is larger than number of samples - cutting times.")
        print(np.max(times)*FPS, num_frames)
    """

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

def init_targets(annos, feats):
    targs = []
    for anno, feat in zip(annos, feats):
        targ = compute_target(anno, len(feat)) # time axis for length!
        targs.append(targ)
    return targs





# LOAD FEATURES AND ANNOTATIONS, COMPUTE TARGETS
def init_feats_annos_targets(feat_path_root, anno_path_root):
    feat_paths = search_files(feat_path_root, FEATURE_EXT)
    anno_paths = search_files(anno_path_root, ANNOTATION_EXT)

    features = [np.load(p) for p in feat_paths]
    # librosa has time in rows, madmom is transposed! now first index is time as in madmom!
    if TRANSPOSE_FEATURES:
        features = [np.transpose(f) for f in features]

    if FILTER_FEATURES:
        features = [f[:, LOWER_FILTER_IDX : UPPER_FILTER_IDX] for f in features]

    annotations = [madmom.io.load_beats(p) for p in anno_paths]
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

    # print(sort_idxs)
    # print(features_rand[31][0][:5])
    # print(targets_rand[31][:50])
    # print(annotations_rand[31][:5])

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

    # Conacatenate spectrograms along the time axis
    # features = np.concatenate(features, axis=1)
    # print(features.shape)
        
    return train_f, train_t, train_anno, valid_f, valid_t, valid_anno, test_f, test_t, test_anno
