# Receptive Field size calculation for dilated convolutional layers
def receptive_field(init_kernel, kernel, layer_num):
  # init_kernel is simply the receptive field that is present prior to dilated layers
  # (e.g. 11 -> 3 + 3 + 3 + 5 from conv layers + the first d-conv layer)
  # even though the first d-conv layer has kernel 5, that kernel already looks at 7 frames per point of data, hence 11

  dilation_factor = 2**(layer_num-1)

  if layer_num == 1:
    return 1 + (init_kernel-1) * dilation_factor

  prev_r_field = receptive_field(init_kernel, kernel, layer_num-1)
  return prev_r_field + (kernel-1) * dilation_factor

r1 = receptive_field(2, 2, 4) # wavenet
r2 = receptive_field(3, 3, 3) # https://theaisummer.com/receptive-field/
r3 = receptive_field(11, 5, 8) # BeatChordMTLDCNN
print(r1)
print(r2)
print(r3)



import numpy as np

# DEMONSTRATE PITCH SHIFTING
def pitch_shift():
  shift = 3
  n_shift = -3
  x = np.array([1,2,3,4,5,6,7,8,9,10])
  y = np.array([1,2,3,4,5,6,7,8,9,10])

  print('sequence:', x)

  print('part to replace (shift up):', y[shift:])
  print('by part:', y[:-shift])
  y[shift:] = y[:-shift]
  print('shifted up sequence:', y)

  print('part to replace (shift down):', x[:n_shift])
  print('by part:', x[-n_shift:])
  x[:n_shift] = x[-n_shift:]
  print('shifted down sequence:', x)

pitch_shift()