import numpy as np

from madmom.audio.spectrogram import LogarithmicFilteredSpectrogram, SpectrogramDifference
from madmom.audio.filters import LogarithmicFilterbank

import os

import scripts.mtl_8fold_config as conf
#paths = conf.FEATURE_PATH

#TODO: fix the path
paths = [
    '/data3/datasets/chords/zweieck/audio/feat_cache_boeck/'
]

DEFAULT_SETTINGS = dict()
DEFAULT_SETTINGS['frame_sizes'] = [1024, 2048, 4096]
# feat length:   21 (21) 45 (45) 91 (91)
DEFAULT_SETTINGS['feat_size'] = 314
DEFAULT_SETTINGS['fps'] = 100
DEFAULT_SETTINGS['num_bands'] = [3, 6, 12]
DEFAULT_SETTINGS['fmin'] = 30
DEFAULT_SETTINGS['fmax'] = 17000
DEFAULT_SETTINGS['norm_filters'] = True
DEFAULT_SETTINGS['start_silence'] = 0
DEFAULT_SETTINGS['diff'] = True
DEFAULT_SETTINGS['diff_ratio'] = 0.5
DEFAULT_SETTINGS['positive_diffs'] = True
DEFAULT_SETTINGS['num_channels'] = 1
DEFAULT_SETTINGS['sample_rate'] = 44100
DEFAULT_SETTINGS['name'] = 'boeck'


def extract_feature(audiofile, settings=DEFAULT_SETTINGS):
    spectrograms = []
    for size_idx, frame_size in enumerate(settings['frame_sizes']):
        spectrogram = LogarithmicFilteredSpectrogram(
            audiofile, num_channels=settings['num_channels'], sample_rate=settings['sample_rate'],
            filterbank=LogarithmicFilterbank, frame_size=frame_size, fps=settings['fps'],
            num_bands=settings['num_bands'][size_idx], fmin=settings['fmin'], fmax=settings['fmax'],
            norm_filters=settings['norm_filters'], start_silence=settings['start_silence'])
        spectrograms.append(spectrogram)

        if settings['diff']:
            spectrogram_diff = SpectrogramDifference(
                spectrogram, diff_ratio=DEFAULT_SETTINGS['diff_ratio'],
                positive_diffs=DEFAULT_SETTINGS['positive_diffs'],
                stack_diffs=np.hstack)
            spectrograms.append(spectrogram_diff)

    return np.hstack(spectrograms)


def main():
    for _, feat_path in enumerate(paths):
        audio_path = os.path.join(feat_path, '..')
        files = os.listdir(audio_path)

        #TODO: fix the path
        feat_path = '/home/pryab/workspace/python/BeatChord/data/features/common/zweieck'

        for _, f in enumerate(files):
            
            feat_file = os.path.join(feat_path, os.path.splitext(f)[0])
            if not os.path.exists(feat_file) or not os.path.isfile(feat_file):

                if f.lower().endswith('.wav') or f.lower().endswith('.flac'):
                    audio_file = os.path.join(audio_path, f)
                    try:
                        feature = extract_feature(audio_file)
                        if not os.path.exists(feat_path):
                            os.makedirs(feat_path)
                        np.save(feat_file, feature)
                        print('audio_file', audio_file)
                        print('feat_file', feat_file)
                    except:
                        print('COULD NOT EXTRACT FEATURES FOR AUDIO FILE!', audio_file)

if __name__ == '__main__':
    main()