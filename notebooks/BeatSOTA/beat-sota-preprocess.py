#!/usr/bin/env python
# coding: utf-8

# # Beat SOTA Preprocess

# EXAMPLE ARGUMENTS:
# 44100 2048 100 81 -v --audio_format .wav --audio_path data/audio/beatles --feature_path data/features/beatles

# In[ ]:


import os
import madmom
from madmom.utils import search_files, match_file
import librosa
# import librosa.display
import numpy as np
import argparse


# In[ ]:


CURRENT_PATH = os.getcwd()
DEFAULT_AUDIO_FORMAT = '.flac'
DEFAULT_AUDIO_PATH = os.path.join(CURRENT_PATH, 'audio')
DEFAULT_FEATURE_EXT = '.feat.npy'
DEFAULT_FEATURE_PATH = os.path.join(CURRENT_PATH, 'features')


# In[ ]:


# command line args
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true', help='output additional information')
parser.add_argument('sr', type=int, help='sampling rate')
parser.add_argument('frame_size', type=int, help='number of samples per frame')
parser.add_argument('fps', type=int, help='number of frames per second')
parser.add_argument('num_bands', type=int, help='number of mel bins')
parser.add_argument('--audio_format', help='audio file format', default=DEFAULT_AUDIO_FORMAT)
parser.add_argument('--audio_path', help='relative path to aduio files folder', default=DEFAULT_AUDIO_PATH)
parser.add_argument('--feature_path', help='relative path to computed feature files folder', default=DEFAULT_FEATURE_PATH)
args = parser.parse_args()

VERBOSE = args.verbose

if VERBOSE:
    print('\n---- EXECUTION STARTED ----\n')
    print('Command line arguments:\n\n', args, '\n')


# In[ ]:


AUDIO_FORMAT = args.audio_format
AUDIO_PATH = args.audio_path # '../../../datasets/beat_boeck'
FEATURE_PATH = args.feature_path

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


audio_paths = search_files(AUDIO_PATH, AUDIO_FORMAT)

for i, path in enumerate(audio_paths):
    spec = pre_process(path)

    base_path, name = os.path.split(path)
    base_name, ext = os.path.splitext(name)
    feat_path = os.path.join(FEATURE_PATH, base_name + DEFAULT_FEATURE_EXT)
    np.save(feat_path, spec)

    if VERBOSE and (i + 1) % 50 == 0:
        print(i+1, ' features have been computed...')


# In[ ]:


if VERBOSE:
    print('\n---- EXECUTION FINISHED ----\n')