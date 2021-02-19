import os
from madmom.utils import search_files
import numpy as np

CURRENT_PATH = os.getcwd()

ANNOTATION_BASE_PATH = os.path.join(CURRENT_PATH, 'data/annotations') # os.path.join(CURRENT_PATH, 'data/annotations')
ANNOTATION_EXT = '.chords'

ANNOTATION_PATH = [
    os.path.join(ANNOTATION_BASE_PATH, 'beatles'),
]

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

def parse_annotations(majmin = False):

    anno_paths = search_files(ANNOTATION_PATH[0], ANNOTATION_EXT)

    annotations = [load_chords(p) for p in anno_paths]

    unique_labels = set()
    for _, anno in enumerate(annotations):
        for _, line in enumerate(anno):
            unique_labels.add(line[1])
            #print(line[1])
    unique_labels = list(unique_labels)

    #[print(l, '==>', chord_to_root(l), '==>', root_to_target(chord_to_root(l))) for l in sorted(unique_labels)]
    #[print(l, ' ==> ', root_to_target(chord_to_root(l))) for l in sorted(unique_labels)]

    [print(l, ' ==> ', chord_to_majmin(l), ' ==> ', majmin_to_target(chord_to_majmin(l))) for l in sorted(unique_labels)]



    unique_chord_configs = set()
    for _, anno in enumerate(annotations):
        for _, line in enumerate(anno):
            label = line[1]
            root = chord_to_root(label)
            label = label.split(root)
            unique_chord_configs.add(label[1])
    unique_chord_configs = list(unique_chord_configs)
    # [print(cc) for cc in sorted(unique_chord_configs)]


def load_chords(path):
    file = open(path, 'r')
    lines = file.readlines()

    time_labels = [parse_chord(l) for l in lines]

    return time_labels

def parse_chord(time_label_string):
    time_label = time_label_string.split()
    if(len(time_label) != 2):
        raise Exception('Invalid input file format! Each line must contain exactly 1 timestamp and 1 chord!')

    time_label[0] = round(float(time_label[0]), 6)

    return time_label

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

    return target+1

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