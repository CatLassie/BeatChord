import os
from madmom.utils import search_files
import numpy as np



scale_labels = {
    'maj': 0,
    'min': 1,
    'N': 2
}

labels_to_scale = {
    0: 'maj',
    1: 'min',
    2: 'N'
}

def parse_scale_annotations(anno_path_root, anno_ext, display_unique_chords_and_chord_configs = False):
    anno_paths = search_files(anno_path_root, anno_ext)
    annotations = [load_chords(p) for p in anno_paths]

    if display_unique_chords_and_chord_configs:
        unique_labels = set()
        for _, anno in enumerate(annotations):
            for _, line in enumerate(anno):
                unique_labels.add(line[1])
        unique_labels = list(unique_labels)

        print('All unique chords:\n')
        [print(l, ' ==> ', scale_to_target(chord_to_scale(l))) for l in sorted(unique_labels)]
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
        [print(cc, ' ==>  ', scale_to_target(chord_to_scale(cc))) for cc in sorted(unique_chord_configs)]
        print('\n')

    mapped_annotations = None
    mapped_annotations = [np.array([[line[0], scale_to_target(chord_to_scale(line[1]))] for line in anno]) for anno in annotations]

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
    if(len(time_label) == 3):
        time_label.pop(1)

    if(len(time_label) != 2):
        raise Exception('Invalid input file format! Each line must contain exactly 1 timestamp and 1 chord!')

    time_label[0] = round(float(time_label[0]), 6)

    return time_label

# FUNCTIONS FOR MAPPING CHORDS TO SCALE

def chord_to_scale(label):
    maj_label = 'maj'
    min_label = 'min'
    aug_label = 'aug'
    dim_label = 'dim'
    sus_label = 'sus'
    no_label = 'N'
    maj_third = '3'
    min_third = 'b3'
    no_maj_third = '*3'
    no_min_third = '*b3'

    # no chord label
    if(label == no_label):
        return no_label

    # explicit absence of third
    if((no_maj_third in label) or (no_min_third in label)):
        return no_label

    # NOTE: explicit absence of 1 (*1) should be taken into account? (also for root targets?)
    # explicit absence of first
    #if('*1' in label):
    #    return no_label

    # short label present (maj, min, minmaj, aug, dim, hdim)
    # order of next 2 conditions covers minmaj 7 chords (IMPORTANT!)
    if(min_label in label):
        return min_label

    if(maj_label in label):
        return maj_label

    if(aug_label in label):
        return maj_label

    if(dim_label in label):
        return min_label

    # explicit presence of third
    if(min_third in label):
        return min_label

    label_split = label.split('13')
    for i, l_s in enumerate(label_split):
        if(maj_third in l_s):
            return maj_label

    # sus label present (after 3 and b3 in hierarchy cause suspended chords might have a third appended to them which means they are maj or min)
    if(sus_label in label):
        return no_label

    # not explicit chord build
    if(('(' not in label) and (')' not in label)):
        return maj_label
    
    # explicit chord build but is not full build (doesnt start with 1)
    if(('(1,' not in label) and ('(1)' not in label)):
        return maj_label

    # explicit chord build and is full build (starts with 1)
    if(('(1,' in label) or ('(1)' in label)):
        return no_label

    return 'NOT_FOUND'

def scale_to_target(scale):
    target = scale_labels.get(scale, 'NOT_FOUND')

    if(target == 'NOT_FOUND'):
        raise Exception('Invalid scale label!')

    return target

# FUNCTIONS FOR MAPPING CHORDS TO ROOT

def chord_to_root(label):
    root = label[0]
    if(len(label) > 1 and (label[1] == '#' or label[1] == 'b')):
        root += label[1]
    return root

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
    curr_label = labels[0]

    out_labels = np.empty(0)
    out_labels = np.append(out_labels, labels_to_scale.get(curr_label))

    out_intervals = np.empty((0, 2))
    out_intervals = np.append(out_intervals, [[0,0]], axis=0)

    for i, l in enumerate(labels):

        if l != curr_label:
            time = i / 100
            out_intervals[len(out_intervals) - 1][1] = time
            out_intervals = np.append(out_intervals, [[time, 0]], axis=0)
            
            out_labels = np.append(out_labels, labels_to_scale.get(l))

            curr_label = l

        if i == len(labels) - 1:
            end_time = i/100
            out_intervals[len(out_intervals) - 1][1] = end_time
    
    return out_labels, out_intervals