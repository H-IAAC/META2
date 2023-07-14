import numpy as np
import pandas as pd
import os
import json
import time
from sklearn.manifold import TSNE
from scratch.dataset.processing import reshape_arrays, apply_fourier_transform, get_inbetween_data_indices, group_sensors 
from scratch.plotting.processing_plots import create_count_histogram, plot_similarity_matrix
import configparser

class PAMAP2DataProcessor():
  '''
  Data processor class for PAMAP2 dataset.

  Responsible for reading data, checking if there is a saved file with the processed data and processes data.
  '''

  def __init__(self, persist : bool = True, use_cfg : bool = True, config_file : str = "PAMAP_5.2_20.cfg",
      radix_name : str = "base", frequency : float = 20.0, time_window : float = 5.2, use_optional_data : bool = True, ms_to_remove : int = 10000,
      drop_cols : list[str]= [
            "IMU_hand_acc6X",
            "IMU_hand_acc6Y",
            "IMU_hand_acc6Z",
            "IMU_hand_X",
            "IMU_hand_Y",
            "IMU_hand_Z",
            "IMU_hand_W",
            "IMU_chest_acc6X",
            "IMU_chest_acc6Y",
            "IMU_chest_acc6Z",
            "IMU_chest_X",
            "IMU_chest_Y",
            "IMU_chest_Z",
            "IMU_chest_W",
            "IMU_ankle_acc6X",
            "IMU_ankle_acc6Y",
            "IMU_ankle_acc6Z",
            "IMU_ankle_X",
            "IMU_ankle_Y",
            "IMU_ankle_Z",
            "IMU_ankle_W"
      ],
      normalize_cols : list[str] = [],
      fft_cols : list[str] = []):

      """Constructor

        :param persist: if True, saves processed data as csv.
        :param use_json: if True, uses configuration file instead of arguments.
        :param config_file: a string with the configuration file's name.
        :param radix_name: a string to identify processing arguments when saving data.
        :param frequency: indicates approximate frequency of sensors' data to use.
        :param time_window: indicates approximate time window to group sensors' data.
        :param use_optional_data: if True, loads PAMAP2 optional data.
        :param ms_to_remove: amount of miliseconds of data to remove between different activities.
        :param drop_cols: sensor names to drop when processing data.
        :param normalize_cols: sensor names to normalize when processing data.
        :param fft_cols: sensor names to apply Fourier transform when processing data;

      """

      #If it is ever transformed to a python file, undo the following comments:

      self.file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
      self.use_cfg = use_cfg
      #And remove this line
      self.config_file = config_file
      self.namecols =['timestamp', 'activityID', 'heartRate', 'IMU_hand_temp', 'IMU_hand_accX', 'IMU_hand_accY', 'IMU_hand_accZ', 'IMU_hand_acc6X',
            'IMU_hand_acc6Y', 'IMU_hand_acc6Z', 'IMU_hand_gyroX', 'IMU_hand_gyroY', 'IMU_hand_gyroZ', 'IMU_hand_magX', 'IMU_hand_magY',
            'IMU_hand_magZ', 'IMU_hand_X', 'IMU_hand_Y', 'IMU_hand_Z', 'IMU_hand_W', 'IMU_chest_temp', 'IMU_chest_accX', 'IMU_chest_accY', 'IMU_chest_accZ', 'IMU_chest_acc6X',
            'IMU_chest_acc6Y', 'IMU_chest_acc6Z', 'IMU_chest_gyroX', 'IMU_chest_gyroY', 'IMU_chest_gyroZ', 'IMU_chest_magX', 'IMU_chest_magY',
            'IMU_chest_magZ', 'IMU_chest_X', 'IMU_chest_Y', 'IMU_chest_Z', 'IMU_chest_W','IMU_ankle_temp', 'IMU_ankle_accX', 'IMU_ankle_accY', 'IMU_ankle_accZ', 'IMU_ankle_acc6X',
            'IMU_ankle_acc6Y', 'IMU_ankle_acc6Z', 'IMU_ankle_gyroX', 'IMU_ankle_gyroY', 'IMU_ankle_gyroZ', 'IMU_ankle_magX', 'IMU_ankle_magY',
            'IMU_ankle_magZ', 'IMU_ankle_X', 'IMU_ankle_Y', 'IMU_ankle_Z', 'IMU_ankle_W']

      if(use_cfg):
        self.cfgparser = configparser.ConfigParser()
        self.cfgparser.optionxform = str
        self.cfgparser.read(os.path.join(self.file_path, "Configs", "preprocessing", self.config_file))
        self.persist = self.cfgparser["file"]["persist"]
        self.file_dir = os.path.join(self.file_path, "Datasets", "Raw", "PAMAP2")   
        self.folder_name = (str(self.cfgparser["file"]["radix_name"])+ "_" + self.cfgparser["base_parameters"]["time_window"] + "_" + self.cfgparser["base_parameters"]["frequency"])
        self.frequency = float(self.cfgparser["base_parameters"]["frequency"])
        self.time_window = float(self.cfgparser["base_parameters"]["time_window"])
        self.use_optional_data = self.cfgparser["base_parameters"].getboolean("use_optional_data")
        self.ms_to_remove = int(self.cfgparser["base_parameters"]["ms_to_remove"])
        
        self.drop_cols = [i for i in self.cfgparser['use_cols'] if not self.cfgparser['use_cols'].getboolean(i)]

        self.normalize_cols =  [i for i in self.cfgparser['normalize_cols'] if self.cfgparser['normalize_cols'].getboolean(i)]

        self.fft_cols =  [i for i in self.cfgparser['fft_cols'] if self.cfgparser['fft_cols'].getboolean(i)]

        self.use_cols = [i for i in self.namecols if i not in self.drop_cols]

      else:
        self.cfgparser = None
        self.persist = persist
        self.radix_name = radix_name
        self.file_dir = os.path.join(self.file_path, "Datasets", "Raw", "PAMAP2")   
        self.folder_name = (str(radix_name)+ str("_")+ str(time_window) + "_" + str(frequency))
        self.frequency = frequency
        self.time_window = time_window
        self.use_optional_data = use_optional_data
        self.ms_to_remove = ms_to_remove
        self.drop_cols = drop_cols
        self.normalize_cols = normalize_cols
        self.fft_cols = fft_cols
        self.use_cols = [i for i in self.namecols if i not in self.drop_cols]
        self.processing_time = 0.0

  def get_param_cfg(self):
    paramcfg = configparser.ConfigParser()
    paramcfg.optionxform = str
    paramcfg["base_parameters"] = {'frequency': self.frequency,
                                   'time_window' : self.time_window,
                                   'use_optional_data' : self.use_optional_data,
                                   'ms_to_remove' : self.ms_to_remove
                                  }
    paramcfg["file"] = {'persist': self.persist,
                        'radix_name' : self.radix_name,
                        'process_time' : "0.0"
                        }
    paramcfg["use_cols"] = {i : (i in self.use_cols) for i in self.namecols}
    paramcfg["normalize_cols"] = {i : (i in self.normalize_cols) for i in self.namecols}
    paramcfg["fft_cols"] = {i : (i in self.fft_cols) for i in self.namecols}
    return paramcfg


  def process_data(self) -> pd.DataFrame:
    '''
    When called, this function processes the dataset according to its parameters.

    Returns:
      pd.DataFrame : processed PAMAP2 data.
    '''

    start_time = time.time()
    #identify already created processing parameters
    if(not self.use_cfg):
        self.cfgparser = self.get_param_cfg()
    
    for i in os.listdir(os.path.join(self.file_path, "Datasets", "Processed", "PAMAP2")):
      rdparser = configparser.ConfigParser()
      rdparser.optionxform = str  

      rdparser.read(os.path.join(self.file_path, "Datasets", "Processed", "PAMAP2", i, "configs.cfg"))
        
      if not (False in [self.cfgparser[i] == rdparser[i] for i in self.cfgparser.sections() if i != "file"]):
          print("Found processed data according to pre-processing, copying data...")
          total_df = pd.read_table(os.path.join(self.file_path, "Datasets", "Processed", "PAMAP2", i, "dataset.csv"), sep='\s+')
          return total_df

    # Check whether frequency and time_window are compatible with the dataset
    if(not (self.time_window * self.frequency).is_integer()):
      print(f"Warning! time_window: {self.time_window} seconds is not compatible to the chosen frequency : {self.frequency} Hz, defaulting to 5.2 sec / 20 Hz")
      self.frequency = 20
      self.time_window = 5.2

    if(not os.path.exists(self.file_dir)):
      print("Não foram encontrados os arquivos do PAMAP2 no sistema")

    # Reads raw data
    protocol_raw_data = []
    for i in range(1, 10):
      protocol_raw_data.append(pd.read_table(os.path.join(self.file_dir, "Protocol", "subject10" + str(i) + ".dat"), sep="\s+", names = self.namecols))

    if(self.use_optional_data):
      optional_raw_data = []
      optional_subjects = [1, 5, 6, 8, 9]
      for i in optional_subjects:
        optional_raw_data.append(pd.read_table(os.path.join(self.file_dir, "Optional", "subject10" + str(i) + ".dat"), sep="\s+", names = self.namecols))
    else:
      optional_subjects = []
      optional_raw_data = []

    # Fills NaN's
    for subject in protocol_raw_data + optional_raw_data:
      subject.interpolate(inplace = True)

    # Remove in-between data and adequate frequency
    for subject in protocol_raw_data + optional_raw_data:
      subject.drop([j for j in range(subject.shape[0]) if j%int(100/self.frequency)!=0] + get_inbetween_data_indices(subject, self.ms_to_remove), axis=0, inplace=True)

    # Concatenate optional and protocol
    raw_data_list = []
    for i in range(1,10):
      if i in optional_subjects:
        raw_data_list.append(pd.concat([protocol_raw_data[i - 1], optional_raw_data[optional_subjects.index(i)]]))
      else:
        raw_data_list.append(protocol_raw_data[i - 1])

    # Drop label 0
    for subject in raw_data_list:
      subject['activityID'].replace(0, np.nan, inplace=True)
      subject.dropna(how='any', axis=0, inplace=True)

    # Drop the features that are not used
    for subject in raw_data_list:
      subject.drop(self.drop_cols, axis = 1, inplace = True)

    # Normalize
    for subject in raw_data_list:
      for nrm_col in self.normalize_cols:
        subject[nrm_col] = (subject[nrm_col] - np.average(subject[nrm_col]))/ np.linalg.norm(subject[nrm_col])

    # Reshape arrays and group sensors
    reshaped_arrays = reshape_arrays(raw_data_list, int(self.frequency * self.time_window), len(self.use_cols))

    excess_trimmed_arrays = []

    drop_excess_ID = list(range(len(self.use_cols), reshaped_arrays[0].shape[1], len(self.use_cols))) + list(range(len(self.use_cols) + 1, reshaped_arrays[0].shape[1], len(self.use_cols)))

    for subject_arr in reshaped_arrays:
      excess_trimmed_arrays.append(np.delete(subject_arr, drop_excess_ID, axis = 1))

    first_preproc_arrays = []
    for i in excess_trimmed_arrays:
      first_preproc_arrays.append(group_sensors(i, len(self.use_cols) - 2))

    #Add column labels
    column_names = ["timestamp", "activityID"]
    feature_names = []
    for sensor_name in self.use_cols:
      if sensor_name not in column_names:
        for i in range(int(self.frequency * self.time_window)):
          feature_names.append(sensor_name + "_" + str(i+1))

    column_names  = column_names + feature_names
    correct_column_order = ["timestamp", "subjectID"] + feature_names + ["activityID"]

    #ApplyFFT
    for data in first_preproc_arrays:
      for fft_sensor in self.fft_cols:
        lower_index = column_names.index(fft_sensor + "_1")
        top_limit = lower_index + int(self.frequency * self.time_window) #exclusive
        apply_fourier_transform(data, lower_index, top_limit)

    data_frames = []
    #concatenate everything and save once, adding column with "subjectID"
    for i, data in enumerate(first_preproc_arrays):
      data_frames.append(pd.DataFrame(data = np.concatenate((data, np.ones((data.shape[0], 1)) + i), axis = 1), columns = column_names + ["subjectID"]))
      data_frames[i] = data_frames[i].reindex(columns = correct_column_order)

    if(self.persist):
      destination_dir = os.path.join(self.file_path, "Datasets", "Processed", "PAMAP2", self.folder_name)
      os.makedirs(destination_dir , exist_ok = True)

    total_df = pd.concat(data_frames, axis = 0)
    total_df = total_df.set_axis([i for i in range(total_df.shape[0])], axis = "index")
    if(self.persist):
      total_df.to_csv(os.path.join(destination_dir, "dataset.csv"), sep=' ', index = False)

    #Create first graph
    create_count_histogram(total_df, activities = {1: "Lying", 2: "Sitting", 3: "Standing", 4: "Walking", 5: "Running", 6: "Cycling", 7: "Nordic Walking", 8: "none", 9: "Wathcing TV", 10: "computer work", 11:"car driving", 12: "ascending stairs", 13: "descending stairs", 14:"none", 15:"none", 16: "vacuum cleaning", 17: "ironing", 18: "folding laundry", 19: "house cleaning",
          20: "playing soccer", 21:"none", 22:"none", 23:"none", 24: "rope jumping"}, filepath = destination_dir)

    #Create second graph
    reducer_columns = [i for i in correct_column_order if i not in ["activityID"]]
    reducer = TSNE(n_components=2, perplexity = 32*4 + 4, early_exaggeration = (26*1.5)+1)
    plot_similarity_matrix(reducer,  (total_df.reindex(columns = ["activityID"] + reducer_columns)).to_numpy(), label = {1: "Lying", 2: "Sitting", 3: "Standing", 4: "Walking", 5: "Running", 6: "Cycling", 7: "Nordic Walking",  9: "Wathcing TV",10: "computer work", 11:"car driving", 12: "ascending stairs", 13: "descending stairs", 16: "vacuum cleaning", 17: "ironing", 18: "folding laundry", 19: "house cleaning",
          20: "playing soccer", 24: "rope jumping"}, filepath = destination_dir)

    end_time = time.time()
    print(f"Elapsed time in seconds: {round((end_time - start_time), 2)}")
    self.processing_time = round((end_time - start_time), 2)
     
    if(self.persist):
      #create cfg file to get already processed data
      self.cfgparser["file"]["process_time"] = str(self.processing_time)
      with open(os.path.join(destination_dir, "configs.cfg"), "w") as dump_file:
        self.cfgparser.write(dump_file)

    return total_df
