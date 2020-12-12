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
def init_feats_annos_targets():
    feat_paths = search_files(FEATURE_PATH, FEATURE_EXT)
    anno_paths = search_files(ANNOTATION_PATH, ANNOTATION_EXT)

    features = [np.load(p) for p in feat_paths]
    # librosa has time in rows, madmom is transposed! now first index is time as in madmom!
    if TRANSPOSE_FEATURES:
        features = [np.transpose(f) for f in features]

    annotations = [madmom.io.load_beats(p) for p in anno_paths]
    targets = init_targets(annotations, features)

    assert len(features) == len(targets)
    return features, annotations, targets

def shuffle_data(features, annotations, targets):
    # get sort indices by length
    feture_lengths = [len(x) for x in features]
    sort_idxs = np.argsort(feture_lengths)

    # sort by feature length
    features_sort = [features[i] for i in sort_idxs]
    targets_sort = [targets[i] for i in sort_idxs]
    annotations_sort = [annotations[i] for i in sort_idxs]

       ######## optionally filter out tracks of length less than <length> ########
    # filter out 12 shortes tracks (>10sec), for IAMA dataset only!
    # features_sort = features_sort[12:]
    # targets_sort = targets_sort[12:]
    # annotations_sort = annotations_sort[12:]
       ###########################################################################

    # print(sort_idxs)
    # print(features_sort[164][0][:5])
    # print(targets_sort[164][:50])
    # print(annotations_sort[164][:5])

    if VERBOSE:
        print('shortest track is:', len(features_sort[0]), 'frames at', FPS, 'FPS')
        print('longest track is:', len(features_sort[len(features_sort)-1]), 'frames at', FPS, 'FPS')

    # get sort indices by length again
    feture_lengths = [len(x) for x in features_sort]
    sort_idxs = np.argsort(feture_lengths)

    # shuffle indices
    random.seed(SEED)
    random.shuffle(sort_idxs)

    # shuffle data
    features_rand = [features_sort[i] for i in sort_idxs]
    targets_rand = [targets_sort[i] for i in sort_idxs]
    annotations_rand = [annotations_sort[i] for i in sort_idxs]

    # print(sort_idxs)
    # print(features_rand[31][0][:5])
    # print(targets_rand[31][:50])
    # print(annotations_rand[31][:5])

    return features_rand, annotations_rand, targets_rand





# MAIN FUNCTION
def init_data():
    # init features, annotations and targets
    features, annotations, targets = init_feats_annos_targets()
    
    # 0 pad all features to start from frame 1
    if FRAME_ONE_START:
        if VERBOSE:
            print('Padded data with zeros to start from frame one!\n')
        features = zero_pad_all_features(features)
        targets = zero_pad_all_targets(targets)

    # 0 pad features that are shorter than 8193 frames
    elif ZERO_PAD:
        if VERBOSE:
            print('Padded data with zeros to match context!\n')
        features = zero_pad_short_features(features)
        targets = zero_pad_short_targets(targets)

    elif VERBOSE:
        print('Data zero padding disabled!\n')
        
    # shuffle data (optionally filter out short tracks)
    features_rand, annotations_rand, targets_rand = shuffle_data(features, annotations, targets)

    # find split indices and split data
    first_idx = int(len(features_rand)*TRAIN_SPLIT_POINT)
    second_idx = int(len(features_rand)*VALIDATION_SPLIT_POINT)

    train_f = features_rand[: first_idx]
    train_t = targets_rand[: first_idx]
    valid_f = features_rand[first_idx : second_idx]
    valid_t = targets_rand[first_idx : second_idx]
    test_f = features_rand[second_idx :]
    test_t = targets_rand[second_idx :]

    train_anno = annotations_rand[: first_idx]
    valid_anno = annotations_rand[first_idx : second_idx]
    test_anno = annotations_rand[second_idx :]

    if VERBOSE:
        print(len(features_rand), 'feature spectrogram files loaded, with example shape:', features_rand[0].shape)
        print(len(annotations_rand), 'feature annotation files loaded, with example shape:', annotations_rand[0].shape)
        print(len(targets_rand), 'targets computed, with example shape:', targets_rand[0].shape)
        print(len(train_f), 'training features', len(valid_f), 'validation features and', len(test_f), 'test features')

    # Conacatenate spectrograms along the time axis
    # features = np.concatenate(features, axis=1)
    # print(features.shape)
        
    return train_f, train_t, train_anno, valid_f, valid_t, valid_anno, test_f, test_t, test_anno
