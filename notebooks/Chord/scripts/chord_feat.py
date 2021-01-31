import random

import numpy as np
import madmom
from madmom.utils import search_files
import scripts.chord_config as cc

SEED = cc.SEED
VERBOSE = cc.VERBOSE






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

    return train_f, train_t, train_anno, valid_f, valid_t, valid_anno, test_f, test_t, test_anno
