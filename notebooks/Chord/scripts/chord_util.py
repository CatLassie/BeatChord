import os
from madmom.utils import search_files
import numpy as np



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

def parse_annotations(anno_path_root, anno_ext, majmin = False, display_unique_chords_and_chord_configs = False):
    anno_paths = search_files(anno_path_root, anno_ext)
    annotations = [load_chords(p) for p in anno_paths]

    if display_unique_chords_and_chord_configs:
        unique_labels = set()
        for _, anno in enumerate(annotations):
            for _, line in enumerate(anno):
                unique_labels.add(line[1])
        unique_labels = list(unique_labels)

        print('All unique chords:\n')
        if majmin:
            [print(l, ' ==> ', majmin_to_target(chord_to_majmin(l))) for l in sorted(unique_labels)]
        else:
            [print(l, ' ==> ', root_to_target(chord_to_root(l))) for l in sorted(unique_labels)]
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
    if majmin:
        mapped_annotations = [np.array([[line[0], majmin_to_target(chord_to_majmin(line[1]))] for line in anno]) for anno in annotations]
    else:
        mapped_annotations = [np.array([[line[0], root_to_target(chord_to_root(line[1]))] for line in anno]) for anno in annotations]

    return mapped_annotations

# FUNCTIONS FOR PARSING CHORD ANNOTATIONS

def load_chords(path):
    file = open(path, 'r')
    lines = file.readlines()

    time_labels = [parse_chord(l) for l in lines]

    return time_labels

def parse_chord(time_label_string):
    time_label = time_label_string.split()

    # fix for robbie_williams dataset (second column contains ending times for chords which are redundant)
    #if(len(time_label) == 3):
    #    time_label.pop(1)

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

# FUNCTIONS FOR MAPPING CHORDS TO MAJOR/MINOR

def chord_to_majmin(label):
    if(label == 'N'):
        return label

    root = chord_to_root(label)
    majmin = root

    if(':' not in label):
        majmin +=  ':maj'
    else:
        affix = label.split(':')[1]
        affix_3 = affix[:3]
        affix_4 = affix[:4]
        
        if(affix_3 == 'min' or affix_3 == 'dim' or affix_4 == 'hdim'):
            majmin += ':min'
        elif(affix_3 == 'maj' or affix_3 == 'aug'):
            majmin += ':maj'
        elif(affix_3 == 'sus'):
            majmin = 'N'
        else:
            majmin += ':maj'

    return majmin

def majmin_to_target(majmin):
    if(majmin == 'N'):
        return root_to_target(majmin)

    majminsplit = majmin.split(':')
    if(len(majminsplit) != 2):
        raise Exception('Invalid chord label format! Must be of the form "<root>:<maj/min>"')

    root = majminsplit[0]
    affix = majminsplit[1]
    target = root_to_target(root)

    if(affix == 'min'):
        target += 13

    return target

# FUNCTIONS FOR MAPPING OUTPUT LABELS TO NOTES AND INTERVALS (for mir_eval evaluation)

def labels_to_notataion_and_intervals(labels):
    # out_labels = np.empty(len(labels), dtype='object')

    curr_label = labels[0]

    out_l = np.empty(0)
    out_l = np.append(out_l, labels_to_letters.get(curr_label))

    out_intervals = np.empty((0, 2))
    out_intervals = np.append(out_intervals, [[0,0]], axis=0)

    for i, l in enumerate(labels):
        # out_labels[i] = labels_to_letters.get(l)

        if l != curr_label:
            time = i / 100
            out_intervals[len(out_intervals) - 1][1] = time
            out_intervals = np.append(out_intervals, [[time, 0]], axis=0)

            out_l = np.append(out_l, labels_to_letters.get(l))

            curr_label = l

        if i == len(labels) - 1:
            end_time = i/100
            out_intervals[len(out_intervals) - 1][1] = end_time
    
    #return out_labels, out_intervals
    return out_l, out_intervals