#!/usr/bin/env python
# coding: utf-8

# # Beat SOTA Preprocess

# In[ ]:


import os
import madmom
from madmom.utils import search_files, match_file
import librosa
import librosa.display
import numpy as np
import argparse


# In[ ]:


# command line args
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true', help='output additional information')
parser.add_argument('sr', type=int, help='sampling rate')
parser.add_argument('frame_size', type=int, help='number of samples per frame')
parser.add_argument('fps', type=int, help='number of frames per second')
parser.add_argument('num_bands', type=int, help='number of mel bins')
args = parser.parse_args()

VERBOSE = args.verbose
SR = args.sr
FRAME_SIZE = args.frame_size
FPS = args.fps
NUM_BANDS = args.num_bands
if VERBOSE:
    print('Comman line arguments: ', args)


# In[ ]:


audio_paths = search_files('../../../datasets/beat_boeck', '.flac')

CURRENT_PATH = os.getcwd()
FEATURE_PATH = os.path.join(CURRENT_PATH, 'features')

if not os.path.exists(FEATURE_PATH):
    os.makedirs(FEATURE_PATH)  


# In[ ]:


SR = args.sr #44100 # samping rate
FRAME_SIZE = args.frame_size #2048 # number of samples per frame
FPS = args.fps #100 # frames / second
HOP_SIZE = int(SR / FPS) # hop size = 10ms or 441 samples
# TODO: Mel bins should be from 30 to 17000 Hz !!!
NUM_BANDS = args.num_bands #81 # number of mel bins


# In[ ]:


def pre_process(filename, **kwargs):
    signal, _ = librosa.load(filename, sr=SR)
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=SR, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=NUM_BANDS)
    db_mel_spec = librosa.power_to_db(mel_spec)
    return db_mel_spec


# In[ ]:


feat_ext = '.feat.npy'

for path in audio_paths:
    spec = pre_process(path)

    base_path, name = os.path.split(path)
    base_name, ext = os.path.splitext(name)
    feat_path = os.path.join(FEATURE_PATH, base_name + feat_ext)
    np.save(feat_path, spec)    


# In[ ]:

if VERBOSE:
    print('\n---- EXECUTION FINISHED ----\n')