import numpy as np
import scripts.beat_sota_config as bsc

FEATURE_CONTEXT = bsc.FEATURE_CONTEXT
FPS = bsc.FPS

# HELPER FUNCTIONS FOR FEATURES AND TARGETS

# TODO: pass context as a param to functions

def zero_pad_short_features(feat_list):
    # 0 pad features to start from frame 1
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
    # 0 pad targets to start from frame 1
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