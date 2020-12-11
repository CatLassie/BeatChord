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
FEATURE_BASE_PATH = os.path.join(CURRENT_PATH, 'data/features')
ANNOTATION_BASE_PATH = os.path.join(CURRENT_PATH, 'data/annotations')

DATASET_NAME = '<DATASET_FOLDER_NAME>'

MODEL_NAME = 'beat_sota'
MODEL_PATH = os.path.join(MODEL_BASE_PATH, DATASET_NAME)
    
FEATURE_EXT = '.feat.npy'
FEATURE_PATH = os.path.join(FEATURE_BASE_PATH, DATASET_NAME)
ANNOTATION_EXT = '.beats'
ANNOTATION_PATH = os.path.join(ANNOTATION_BASE_PATH, DATASET_NAME)

# feature parameters
# SR = 44100 # samping rate
# FRAME_SIZE = 2048 # number of samples per frame
FPS = 100 # frames / second
# HOP_SIZE = int(SR / FPS) # hop size = 10ms or 441 samples
# TODO: Mel bins should be from 30 to 17000 Hz !!!
# NUM_BANDS = 81 # number of mel bins

TRAIN_SPLIT_POINT = 0.7
VALIDATION_SPLIT_POINT = 0.85





######## TRAINING PARAMETERS ########

# number of epochs
NUM_EPOCHS = 50 #1 10 25 ?

# learning rate
LR = 0.001 # reduce by a factor of five whenever <condition from paper> is reached
# lr = 0.01 ?

# context for 1 feature (e.g. 4096 frames on either side, that would be 8193)
FEATURE_CONTEXT = 8193 #800 #1000
TRAINING_HOP_SIZE = 512 #40 #100

BATCH_SIZE = 1
PATIENCE = 4 #9999





######## COMMAND LINE SUPPORT ARGUMENTS ########

TRAIN = False
PREDICT = False
ZERO_PAD = True
VERBOSE = True # args.verbose
COMPLETE_DISPLAY_INTERVAL = 5 # desired completion dislpay frequency in percentage





########