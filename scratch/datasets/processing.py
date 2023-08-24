import numpy as np
import pandas as pd

def apply_fourier_transform(array : np.ndarray , lower_index : int, top_limit : int) -> np.ndarray:
  """
  Applies FFT in data between columns lower_index and top_limit

    Args:
      array(np.ndarray) : the array in which the fourier transform is going to be applied
      lower_index(int) : the first column value to apply FFT (inclusive)
      top_limit(int) : where FFT stops (exclusive)

    Returns:
      np.ndarray : the parameter "array" with FFT applied on selected columns, note it is not an inPlace transform

  """
  array[:, lower_index:top_limit] = np.fft.fft(array[:, lower_index:top_limit])

def group_sensors(array : np.ndarray, distance : int) -> np.ndarray:
  """
  Groups sensor types

    Args:
      array(np.ndarray) : the array containing sensors to be grouped
      distance(int) : the distance between sensors to be gouped

    Returns:
      np.ndarray: An array containing the original sensors but grouped column-wise

    Example:
    Suppose I have an array of sensors with columns (acc1, mag1, acc2, mag2), in this case it should return (acc1, acc2, mag1, mag2)

  """

  #keeps activity ID and timestamp
  permutation = [0, 1]
  #groups the rest
  for i in range(distance):
    for j in range((array.shape[1]-1)//distance):
      permutation.append(2 + i + (distance*j))
  return array[:, permutation]

def clear_first_excess(subject_arr : np.ndarray, num_samples : int = 60) -> np.ndarray:
  """
  Clears sample excess so that when resizing, there is no row with multiple activityID's.

    Args:
      subject_arr(np.ndarray) : the array containing activityID and timestamp as first column.
      num_samples(int) : the number of samples that amount one row of the result. Defaults to 60.

    Returns:
      np.ndarray: The trimmed array, with only one group of rows (at max num_samples - 1) trimmed.

  """
  for i in range(2, subject_arr.shape[0]):
    if(subject_arr[i][1] != subject_arr[i-1][1] and (i%num_samples != 0)):
      #stop and remove from (i//num_samples)*num_samples until i (not included)
      subject_arr = np.delete(subject_arr, range((i//num_samples)*num_samples, i), axis = 0)
      return subject_arr

  #trim excess
  subject_arr = np.delete(subject_arr, range((subject_arr.shape[0]//num_samples)*num_samples, subject_arr.shape[0]), axis = 0)
  return subject_arr

def check_excess(subject_arr : np.ndarray, num_samples : int = 60 ) -> bool:
  """
  Checks excess in order to continue trimming with clearFirstExcess function

  Args:
    subject_arr(np.ndarray) : an array where activityID is in the second column
    num_samples(int): the amount of samples to group in a line. Defaults to 60.

  Returns:
    bool: If the number of samples that can be grouped is not num_samples
  """
  for i in range(2, subject_arr.shape[0]):
    if(subject_arr[i][1] != subject_arr[i-1][1] and (i%num_samples != 0)):
      return True
  if(subject_arr.shape[0]%num_samples != 0):
      return True
  return False

def reshape_arrays(df_list : list[pd.DataFrame], num_samples : int = 60, num_features : int = 32) -> list[np.ndarray]:
  """
  Reshapes array so that one row of the array has num_samples samples

    Args:
      df_list(list[pd.DataFrame]): a list of DataFrames containing data with column 0 as activityID
      num_samples(int): the number of samples that amount one row of the result. Defaults to 60.
      num_features(int): the number of features in the dataFrame (includes activityID and timestamp). Defaults to 32.

    Returns:
      np.ndarray: The reshaped array.

  """
  df_arrays = []
  reshaped_arrays = []
  for subject in df_list:
    df_arrays.append(subject.to_numpy())
  for subject_arr in df_arrays:
    while(check_excess(subject_arr, num_samples)):
      subject_arr = clear_first_excess(subject_arr, num_samples)
    subject_arr = subject_arr.reshape(-1, num_features*num_samples)
    reshaped_arrays.append(subject_arr)
  return reshaped_arrays

def get_inbetween_data_indices(df : pd.DataFrame, ms_to_remove : int = 10000) -> list[int]:
  """
  Removes data in-between activities (state transition)

    Args:
      df(pd.DataFrame) : the dataFrame that is going to have its in-between data removed, should be sampled to 100Hz
      ms_to_remove(int) : miliseconds to remove before and after activityID changes. Defaults to 10000

    Returns:
      list[int]: A list with the indices to be removed.

  """
  remove_samples = ms_to_remove // 10
  removed = []
  key = 0
  iterator = 0
  miliseconds = 0
  while iterator<df.shape[0]:
    if df["activityID"][iterator] != key or iterator + 1 == df.shape[0]:
      key = df["activityID"][iterator]
      if iterator - remove_samples > 0:
        iterator -= remove_samples
        miliseconds = 0
        while miliseconds < 2*remove_samples and (iterator < df.shape[0]):
          removed.append(iterator)
          miliseconds += 1
          iterator += 1
      else:
        while miliseconds < remove_samples and (iterator < df.shape[0]):
          removed.append(iterator)
          miliseconds += 1
          iterator += 1
    iterator += 1
  return removed