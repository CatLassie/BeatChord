import os
import torch





######## GLOBAL VARIABLES ########

# random seed
SEED = 1

# cuda configuration
DISABLE_CUDA = False
USE_CUDA = not DISABLE_CUDA and torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

# paths
CURRENT_PATH = os.getcwd()
MODEL_BASE_PATH = os.path.join(CURRENT_PATH, 'data/models')
FEATURE_BASE_PATH = os.path.join(CURRENT_PATH, '../../../../../data2/datasets/downbeat') # os.path.join(CURRENT_PATH, 'data/features')
ANNOTATION_BASE_PATH = os.path.join(CURRENT_PATH, '../../../../../data2/datasets/downbeat') # os.path.join(CURRENT_PATH, 'data/annotations')

DATASET_NAME = 'chord_beatles' # '<DATASET_FOLDER_NAME>' # e.g. 'bealtes'

MODEL_NAME = 'chord_beatles' # '<MODEL_NAME>' # e.g. 'beat_sota_beatles_1025c_250h_0z_1b_16l_4p' (context, hop size, 0pad, batch size, conv. layer num., patience)
MODEL_PATH = os.path.join(MODEL_BASE_PATH, DATASET_NAME)
    
FEATURE_EXT = '.npy' # e.g. .feat.npy or .npy
FEATURE_PATH = [
    os.path.join(FEATURE_BASE_PATH, 'beatles/audio/feat_cache_boeck'), # os.path.join(FEATURE_BASE_PATH, 'beatles'),
    # os.path.join(FEATURE_BASE_PATH, 'robbie_williams/audio/feat_cache_boeck')
] # os.path.join(FEATURE_BASE_PATH, DATASET_NAME) # os.path.join(FEATURE_BASE_PATH, 'beatles/audio/feat_cache_boeck') # os.path.join(CURRENT_PATH, '../../../../../data2/datasets/downbeat/beatles/audio/feat_cache_boeck')
ANNOTATION_EXT = '.chords'
ANNOTATION_PATH = [
    os.path.join(ANNOTATION_BASE_PATH, 'beatles/annotations/chords'), # os.path.join(ANNOTATION_BASE_PATH, 'beatles'),
    # os.path.join(ANNOTATION_BASE_PATH, 'robbie_williams/annotations/beats')
] # os.path.join(ANNOTATION_BASE_PATH, DATASET_NAME) # os.path.join(ANNOTATION_BASE_PATH, 'beatles/annotations/beats')

# feature parameters
# SR = 44100 # samping rate
# FRAME_SIZE = 2048 # number of samples per frame
FPS = 100 # frames / second
# HOP_SIZE = int(SR / FPS) # hop size = 10ms or 441 samples
# TODO: Mel bins should be from 30 to 17000 Hz !!!
# NUM_BANDS = 81 # number of mel bins

TRAIN_SPLIT_POINT = 0.75
VALIDATION_SPLIT_POINT = 0.875





######## TRAINING PARAMETERS ########

# number of epochs
NUM_EPOCHS = 50 #1 10 25 ?

# learning rate
LR = 0.001 # reduce by a factor of five whenever <condition from paper> is reached
# lr = 0.01 ?

# context for 1 feature (e.g. 4096 frames on either side, that would be 8193)
FEATURE_CONTEXT = 25 #8193 #800 #1000
TRAINING_HOP_SIZE = 25 #512 #40 #100

BATCH_SIZE = 8
PATIENCE = 4 #9999





######## !!!!IMPORTANT!!!! NETWORK PARAMETERS ########

# value 8 is based on paper (1,8): 81 -> 79 -> 26 -> 24 -> 8 -> 1
# value 9 is needed based on server features (1,9): 91 -> 89 -> 29 -> 27 -> 9 -> 1
# LAST_CNN_KERNEL_FREQUENCY_SIZE = 9 # 8

######################################################





######## COMMAND LINE SUPPORT ARGUMENTS ########

TRAIN = True
PREDICT = True

# for features computed with librosa use this flag, for madmom disable it (server features used madmom)
TRANSPOSE_FEATURES = False # True
# server features are multiple different features stacked on top of each other, use this flag to filter out relevant data
FILTER_FEATURES = True # False
# indices used to filter out 2048 window spectogram from feature matrix (it has 1024, 2048 and spectral flux features stacked together)
LOWER_FILTER_IDX = 132
UPPER_FILTER_IDX = 223

VERBOSE = True # args.verbose

########