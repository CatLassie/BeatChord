import mir_eval
import numpy as np

maj_semitones = np.array(mir_eval.chord.QUALITIES['maj'][:8])
min_semitones = np.array(mir_eval.chord.QUALITIES['min'][:8])

print(maj_semitones)
print(min_semitones)

idx = 0
test_chord = ['G:sus4', 'G:maj6', 'G:maj7/3', 'G:maj9', 'G:min', 'G:min/5', 'G:min/5', 'G:min7', 'G:min7/5']

em_r, em_semi, em_bass = mir_eval.chord.encode_many(test_chord, False)
print(test_chord[idx])
print(em_semi[idx][0:8])

#is_maj = np.all(np.equal(ref_semitones[:, :8], maj_semitones), axis=1)
is_maj = np.all(np.equal(em_semi[idx][4:5], maj_semitones[4:5]))
is_min = np.all(np.equal(em_semi[idx][4:5], min_semitones[4:5]))

print(is_maj)
print(is_min)