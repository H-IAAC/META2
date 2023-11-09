import math
import numpy as np

def interpolate_ponderated(arr, idx):
  if(math.ceil(idx) - idx != 0) or (idx - math.floor(idx) != 0):
    if(idx < 10):
      print(f"index {idx}, interpolating:\n{arr[0:6, math.floor(idx)]} and\n{arr[0:6, math.ceil(idx)]}, result is \n{arr[0:6, math.floor(idx)]*abs(math.ceil(idx) - idx) + arr[0:6, math.ceil(idx)]*abs(idx - math.floor(idx))}")
    return arr[:, math.floor(idx)]*abs(math.ceil(idx) - idx) + arr[:, math.ceil(idx)]*abs(idx - math.floor(idx))
  else:
     return arr[:, math.floor(idx)]
  
def change_samples(array, samples_to, n_sensors):

  samples_from = array.shape[1] - 1
  new_arr = np.zeros((array.shape[0], samples_to*n_sensors + 1))
  new_arr[:, 0] = array[:, 0]

  for i in range(samples_to*n_sensors):
    new_arr[:, i + 1] = interpolate_ponderated(array[:, 1:],(i*samples_from)/(n_sensors*samples_to))
  return new_arr
