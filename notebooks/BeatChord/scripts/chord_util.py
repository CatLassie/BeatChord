import os
from madmom.utils import search_files
import numpy as np
import mir_eval


note_labels = {
    'C': 0,
    'D': 2,
    'E': 4,
    'F': 5,
    'G': 7,
    'A': 9,
    'B': 11,
    'H': 11,
    'N': 12
}

labels_to_letters = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B',
    12: 'N'
}

quality_labels = {
    ':maj': 0,
    ':min': 1,
    'N': 2
}

labels_to_quality = {
    0: ':maj',
    1: ':min',
    2: 'N'
}

def parse_annotations(anno_path_root, anno_ext, display_unique_chords_and_chord_configs = False):
    if anno_path_root == None:
        return None, None

    anno_paths = search_files(anno_path_root, anno_ext)
    annotations = [load_chords(p) for p in anno_paths]

    if display_unique_chords_and_chord_configs:
        unique_labels = set()
        for _, anno in enumerate(annotations):
            for _, line in enumerate(anno):
                unique_labels.add(line[1])
        unique_labels = list(unique_labels)

        print('All unique chords:\n')
        [print(l, ' ==> ', majmin_to_target(chord_to_majmin(l)[0], chord_to_majmin(l)[1])) for l in sorted(unique_labels)]
        print('\n')

    if display_unique_chords_and_chord_configs:
        unique_chord_configs = set()
        for _, anno in enumerate(annotations):
            for _, line in enumerate(anno):
                label = line[1]
                root = chord_to_root(label)
                label = label.split(root)
                unique_chord_configs.add(label[1])
        unique_chord_configs = list(unique_chord_configs)
        print('All unique chord configurations:\n')
        [print(cc) for cc in sorted(unique_chord_configs)]
        print('\n')

    mapped_annotations = None
    mapped_annotations = [np.array([[line[0], majmin_to_target(chord_to_majmin(line[1])[0], chord_to_majmin(line[1])[1])] for line in anno]) for anno in annotations]

    original_annotations = [np.array([[line[0], line[1]] for line in anno]) for anno in annotations]

    return mapped_annotations, original_annotations

# FUNCTIONS FOR PARSING CHORD ANNOTATIONS

def load_chords(path):
    file = open(path, 'r')
    lines = file.readlines()

    time_labels = [parse_chord(l) for l in lines]

    return time_labels

def parse_chord(time_label_string):
    time_label = time_label_string.split()

    # fix for robbie_williams dataset (second column contains ending times for chords which are redundant)
    if(len(time_label) == 3):
        time_label.pop(1)

    if(len(time_label) != 2):
        raise Exception('Invalid input file format! Each line must contain exactly 1 timestamp and 1 chord!')

    time_label[0] = round(float(time_label[0]), 6)

    return time_label

# FUNCTIONS FOR MAPPING CHORDS TO ROOT

def chord_to_root(label):
    root = label[0]
    if(len(label) > 1 and (label[1] == '#' or label[1] == 'b')):
        root += label[1]
    return root

def root_to_target(root):
    target = note_labels.get(root[0], 'NOT_FOUND')

    if(target == 'NOT_FOUND'):
        raise Exception('Invalid root label!')

    if(len(root) > 1):
        if(root[1] == '#'):
            target = (target + 1) % 12
        elif(root[1] == 'b'):
            target = (target - 1) % 12

    return target

def quality_to_target(quality):
    target = quality_labels.get(quality, 'NOT_FOUND')

    if(target == 'NOT_FOUND'):
        raise Exception('Invalid quality label!')

    return target

# FUNCTIONS FOR MAPPING CHORDS TO MAJOR/MINOR

def chord_to_majmin(label):
    if(label == 'N'):
        return 'N', 'N'

    # check if label contains a major third
    maj_semitones = np.array(mir_eval.chord.QUALITIES['maj'])
    min_semitones = np.array(mir_eval.chord.QUALITIES['min'])
    _, label_semitones, _ = mir_eval.chord.encode_many([label], False)

    is_maj = np.all(np.equal(label_semitones[0][4:5], maj_semitones[4:5]))
    is_min = np.all(np.equal(label_semitones[0][3:4], min_semitones[3:4]))

    #is_maj = np.all(np.equal(label_semitones[0][:8], maj_semitones[:8]))
    #is_min = np.all(np.equal(label_semitones[0][:8], min_semitones[:8]))

    root = chord_to_root(label)
    # every chord that has no major third in it is mapped to a minor chord
    quality = ':maj' if is_maj else ':min'

    return root, quality

def majmin_to_target(root, quality):
    if(root == 'N'):
        return root_to_target(root), quality_to_target(quality)

    r_targ = root_to_target(root)
    q_targ = quality_to_target(quality)

    return r_targ, q_targ

# FUNCTIONS FOR 1-hot encoding

def target_to_one_hot(targ, out_size):
    if targ == -1:
        dummy_one_hot_target = np.full(out_size, -1, np.float32)
        return dummy_one_hot_target

    one_hot_target = np.zeros(out_size, np.float32)
    one_hot_target[targ] = 1
    return one_hot_target

def targets_to_one_hot(targ_list, out_size):
    one_hot_list = []
    for _, targ in enumerate(targ_list):
        one_hot_targ = np.array([target_to_one_hot(t, out_size) for t in targ])
        one_hot_list.append(one_hot_targ)

    return one_hot_list 

# FUNCTIONS FOR MAPPING OUTPUT LABELS TO NOTES AND INTERVALS (for mir_eval evaluation)

# FOR ROOTS
def labels_to_notataion_and_intervals(labels):
    curr_label = labels[0]

    out_labels = np.empty(0)
    lb = labels_to_letters.get(curr_label)
    out_labels = np.append(out_labels, lb + (':(5)' if lb != 'N' else ''))

    out_intervals = np.empty((0, 2))
    out_intervals = np.append(out_intervals, [[0,0]], axis=0)

    for i, l in enumerate(labels):

        if l != curr_label:
            time = i / 100
            out_intervals[len(out_intervals) - 1][1] = time
            out_intervals = np.append(out_intervals, [[time, 0]], axis=0)
            
            lb = labels_to_letters.get(l)
            out_labels = np.append(out_labels, lb + (':(5)' if lb != 'N' else ''))

            curr_label = l

        if i == len(labels) - 1:
            end_time = i/100
            out_intervals[len(out_intervals) - 1][1] = end_time
    
    return out_labels, out_intervals

# FOR QUALITIES
def labels_to_qualities_and_intervals(labels):
    curr_label = labels[0]

    out_labels = np.empty(0)
    lb = labels_to_quality.get(curr_label)
    out_labels = np.append(out_labels, ('C' if lb != 'N' else '') + lb)

    out_intervals = np.empty((0, 2))
    out_intervals = np.append(out_intervals, [[0,0]], axis=0)

    for i, l in enumerate(labels):

        if l != curr_label:
            time = i / 100
            out_intervals[len(out_intervals) - 1][1] = time
            out_intervals = np.append(out_intervals, [[time, 0]], axis=0)
            
            lb = labels_to_quality.get(l)
            out_labels = np.append(out_labels, ('C' if lb != 'N' else '') + lb)

            curr_label = l

        if i == len(labels) - 1:
            end_time = i/100
            out_intervals[len(out_intervals) - 1][1] = end_time
    
    return out_labels, out_intervals

# FOR MAJMIN
def labels_to_majmin_and_intervals(r_labels, q_labels):
    r_curr_label = r_labels[0]
    q_curr_label = q_labels[0]

    out_labels = np.empty(0)
    r = labels_to_letters.get(r_curr_label)
    q = labels_to_quality.get(q_curr_label)

    out_labels = np.append(out_labels, r)
    if r !='N':
        if q !='N':
            out_labels[0] = out_labels[0] + q
        else:
            out_labels[0] = out_labels[0] + ':(5)'

    out_intervals = np.empty((0, 2))
    out_intervals = np.append(out_intervals, [[0,0]], axis=0)

    for i, (rl, ql) in enumerate(zip(r_labels, q_labels)):

        if rl != r_curr_label or ql != q_curr_label:
            time = i / 100
            out_intervals[len(out_intervals) - 1][1] = time
            out_intervals = np.append(out_intervals, [[time, 0]], axis=0)
            
            r = labels_to_letters.get(rl)
            q = labels_to_quality.get(ql)
            out_labels = np.append(out_labels, r)
            if r != 'N':
                if q !='N':
                    out_labels[len(out_labels) - 1] = out_labels[len(out_labels) - 1] + q
                else:
                    out_labels[len(out_labels) - 1] = out_labels[len(out_labels) - 1] + ':(5)'

            r_curr_label = rl
            q_curr_label = ql

        if i == len(r_labels) - 1:
            end_time = i/100
            out_intervals[len(out_intervals) - 1][1] = end_time
    
    return out_labels, out_intervals

# format annot intervals for mir_eval
def annos_to_labels_and_intervals(annos, predicted_labels):
    end_time = len(predicted_labels) - 1

    out_labels = np.empty(0)
    out_intervals = np.empty((0, 2))
    
    for i, _ in enumerate(annos):
        out_labels = np.append(out_labels, annos[i][1])
        if i  < len(annos) - 1:
            out_intervals = np.append(out_intervals, [[annos[i][0], annos[i+1][0]]], axis=0)
        else:
            out_intervals = np.append(out_intervals, [[annos[i][0], end_time/100]], axis=0)

    return out_labels, np.around(out_intervals.astype(np.float64), 2)

# format annot quality intervals for mir_eval
def annos_to_qualities_and_intervals(annos, predicted_labels):
    end_time = len(predicted_labels) - 1

    out_labels = np.empty(0)
    out_intervals = np.empty((0, 2))
    
    for i, _ in enumerate(annos):
        label = annos[i][1]
        if label != 'N':
            root = chord_to_root(label)
            label = label.split(root)
            label = 'C' + label[1]

        out_labels = np.append(out_labels, label)
        if i  < len(annos) - 1:
            out_intervals = np.append(out_intervals, [[annos[i][0], annos[i+1][0]]], axis=0)
        else:
            out_intervals = np.append(out_intervals, [[annos[i][0], end_time/100]], axis=0)

    return out_labels, np.around(out_intervals.astype(np.float64), 2)
