import numpy as np
import pandas as pd
import os
import time
from sklearn.manifold import TSNE
from scratch.datasets.processing import apply_fourier_transform
from scratch.plotting.processing_plots import create_count_histogram, plot_similarity_matrix, plot_clustering
from scratch.utils.experimentmanager import ExperimentManager
import configparser

class DSADSDataProcessor():
  '''
  Data processor class for PAMAP2 dataset.

  Responsible for reading data, checking if there is a saved file with the processed data and processes data.
  '''

  def __init__(self, persist : bool = True, use_cfg : bool = True, config_file : str = "DSADS_5_25.cfg",
      radix_name : str = "base", frequency : float = 25.0, time_window : float = 5.0, use_test_data : bool = True,
      drop_cols : list[str]= [],
      normalize_cols : list[str] = [],
      fft_cols : list[str] = []):

      """Constructor

        :param persist: if True, saves processed data as csv.
        :param use_json: if True, uses configuration file instead of arguments.
        :param config_file: a string with the configuration file's name.
        :param radix_name: a string to identify processing arguments when saving data.
        :param frequency: indicates approximate frequency of sensors' data to use.
        :param time_window: indicates approximate time window to group sensors' data.
      """

      #If it is ever transformed to a python file, undo the following comments:
      self.exp = ExperimentManager()  
      self.file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
      self.use_cfg = use_cfg
      #And remove this line
      self.config_file = config_file
      imus = ["IMU_torso", "IMU_RArm", "IMU_LArm", "IMU_RLeg", "IMU_LLeg"]
      orientation = ["X", "Y", "Z"]
      sensor = ["_acc", "_gyro", "_mag"]
      self.namecols = []
      for i in imus:
        for j in sensor:
          for k in orientation:
            self.namecols.append(i+j+k)

      #Mudar a região abaixo de acordo com a solução proposta pelo Wallace
      if(use_cfg):
        self.cfgparser = configparser.ConfigParser()
        self.cfgparser.optionxform = str
        self.cfgparser.read(os.path.join(self.file_path, "Configs", "preprocessing", self.config_file))
        self.persist = self.cfgparser["file"]["persist"]
        self.file_dir = os.path.join(self.exp.get_dir_path("datasets"), "Raw", "DSADS")
        self.folder_name = (str(self.cfgparser["file"]["radix_name"])+ "_" + self.cfgparser["base_parameters"]["time_window"] + "_" + self.cfgparser["base_parameters"]["frequency"])
        self.frequency = float(self.cfgparser["base_parameters"]["frequency"])
        self.time_window = float(self.cfgparser["base_parameters"]["time_window"])
        

        self.drop_cols = [i for i in self.cfgparser['use_cols'] if not self.cfgparser['use_cols'].getboolean(i)]

        self.fft_cols =  [i for i in self.cfgparser['fft_cols'] if self.cfgparser['fft_cols'].getboolean(i)]

        self.normalize_cols = [i for i in self.cfgparser['normalize_cols'] if self.cfgparser['normalize_cols'].getboolean(i)]

        self.use_cols = [i for i in self.namecols if i not in self.drop_cols]

      else:
        self.cfgparser = None
        self.persist = persist
        self.radix_name = radix_name
        self.file_dir = os.path.join(self.exp.get_dir_path("datasets"), "Raw", "DSADS")
        self.folder_name = (str(radix_name)+ str("_")+ str(time_window) + "_" + str(frequency))
        self.frequency = frequency
        self.time_window = time_window
        self.drop_cols = drop_cols
        self.fft_cols = fft_cols
        self.normalize_cols = normalize_cols
        self.use_cols = [i for i in self.namecols if i not in self.drop_cols]
        self.processing_time = 0.0

  def get_param_cfg(self) -> configparser.ConfigParser():
    """
    Gets class parameters as ConfigParser

    Returns:
      configparser.ConfigParser: An configParser object containing this objest's parameters.
      Used to compare preprocessing parameters with previous saved ones, reducing processing time.
  """
    paramcfg = configparser.ConfigParser()
    paramcfg.optionxform = str
    paramcfg["base_parameters"] = {'frequency': self.frequency,
                                   'time_window' : self.time_window,
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
    os.makedirs(os.path.join(self.exp.get_dir_path("datasets"), "Processed", "DSADS") , exist_ok = True)
    start_time = time.time()
    #identify already created processing parameters
    if(not self.use_cfg):
        self.cfgparser = self.get_param_cfg()

    for i in os.listdir(os.path.join(self.exp.get_dir_path("datasets"), "Processed", "DSADS")):
      rdparser = configparser.ConfigParser()
      rdparser.optionxform = str

      rdparser.read(os.path.join(self.exp.get_dir_path("datasets"), "Processed", "DSADS", i, "configs.cfg"))

      if not (False in [self.cfgparser[i] == rdparser[i] for i in self.cfgparser.sections() if i != "file"]):
          print("Found processed data according to pre-processing, copying data...")
          total_df = pd.read_table(os.path.join(self.exp.get_dir_path("datasets"), "Processed", "DSADS", i, "dataset.csv"), sep='\s+')
          return total_df

    # Check whether frequency and time_window are compatible with the dataset
    if(not (self.time_window == 5.0)):
      print(f"Warning! time_window: {self.time_window} seconds is not compatible with DSADS, defaulting to 5.0 sec / 25 Hz")
      self.frequency = 25
      self.time_window = 5.0

    if(not os.path.exists(self.file_dir)):
      print("Não foram encontrados os arquivos do DSADS no sistema")

    # Reads raw data -> total
    extend_sensors = []
    for i in range(125):
      for j in self.namecols:
        extend_sensors.append(j + "_" + str(i + 1))

    right_order_sensors = []

    for j in self.namecols:
      for i in range(125):
        right_order_sensors.append(j + "_" + str(i + 1))

    tables = []
    data_str = os.path.join(self.file_dir, "data")
    for actv in os.listdir(data_str):
      for subject in os.listdir(os.path.join(data_str, actv)):
        for segment in os.listdir(os.path.join(data_str, actv, subject)):   
          data = pd.DataFrame(np.genfromtxt(os.path.join(data_str, actv, subject, segment), delimiter=','))
          data = data[~np.isnan(data).any(axis=1)]
          tables.append(pd.DataFrame(np.concatenate([np.array([[int(actv[-2:]), int(subject[-1:])]]), data.to_numpy().reshape(1, data.shape[0]*data.shape[1])], axis = 1), columns=["activityID", "subjectID"] + extend_sensors))

    total = pd.concat(tables, axis = 0)

    # Drop the features that are not used
    total.drop(self.drop_cols, axis = 1, inplace = True)

    correct_column_order = ["subjectID"] + right_order_sensors + ["activityID"]
    column_names = ["activityID", "subjectID"] + right_order_sensors

    #ApplyFFT
    for fft_sensor in self.fft_cols:
      lower_index = column_names.index(fft_sensor + "_1")
      top_limit = lower_index + int(self.frequency * self.time_window) #exclusive
      apply_fourier_transform(total.to_numpy(), lower_index, top_limit) #using df.to_numpy() casts fft to original dataframe

    # Normalize
    for nrm_col in self.normalize_cols:
      total[nrm_col] = (total[nrm_col] - np.average(total[nrm_col]))/ np.linalg.norm(total[nrm_col])

    #reindexing
    total = total.reindex(columns = correct_column_order)
    total_df = total.set_axis([i for i in range(total.shape[0])], axis = "index")

    if(self.persist):
      destination_dir = os.path.join(self.exp.get_dir_path("datasets"), "Processed", "DSADS", self.folder_name)
      os.makedirs(destination_dir , exist_ok = True)

    if(self.persist):
      total.to_csv(os.path.join(destination_dir, "dataset.csv"), sep=' ', index = False)

    activity_labels = {1: "Sitting", 2: "Standing", 3:"Lying Back", 4:"Lying to the side", 5:"Walking upstairs",
                   6:"Walking downstairs", 7:"Standing on elevator", 8:"Moving around elevator", 9:"Walking",
                   10:"Walking threadmill", 11:"Walking on inclined threadmill", 12:"Running on threadmill",
                   13:"Exercising on stepper", 14:"Cross trainer", 15:"Cycling in horizontal", 16:"Cycling in vertical",
                   17: "Rowing", 18:"Jumping", 19:"Playing basketball"}

    #Create first graph
    create_count_histogram(total_df, activities = activity_labels,
                           filepath = destination_dir)

    #Create second graph
    reducer_class_columns = [i for i in correct_column_order if i not in ["activityID"]]
    reducer = TSNE(n_components=2, perplexity = 32*4 + 4, early_exaggeration = (26*1.5)+1)
    plot_similarity_matrix(reducer,  ((total_df.reindex(columns = ["activityID"] + reducer_class_columns)).drop(labels = ["subjectID"], axis = 'columns')).to_numpy(),
                           label = activity_labels, filepath = destination_dir, title = "classes heatmap")

    plot_clustering(reducer,  ((total_df.reindex(columns = ["activityID"] + reducer_class_columns)).drop(labels = ["subjectID"], axis = 'columns')).to_numpy(),
                    labels = activity_labels,
                     title = "classes clustering", filepath = destination_dir, filename = "classes clustering")

    reducer_subject_columns = [i for i in correct_column_order if i not in ["subjectID"]]
    reducer = TSNE(n_components=2, perplexity = 32*4 + 4, early_exaggeration = (26*1.5)+1)
    plot_similarity_matrix(reducer,  ((total_df.reindex(columns = ["subjectID"] + reducer_subject_columns)).drop(labels = ["activityID"], axis = 'columns')).to_numpy(),
                           label = {i : "subject" + str(i) for i in range(1, 9)}, filepath = destination_dir, title = "subjects heatmap")

    plot_clustering(reducer,  ((total_df.reindex(columns = ["subjectID"] + reducer_subject_columns)).drop(labels = ["activityID"], axis = 'columns')).to_numpy(),
                           labels = {i : "subject" + str(i) for i in range(1, 9)}, title = "classes clustering", filepath = destination_dir, filename = "subjects clustering")

    end_time = time.time()
    print(f"Elapsed time in seconds: {round((end_time - start_time), 2)}")
    self.processing_time = round((end_time - start_time), 2)

    if(self.persist):
      #create cfg file to get already processed data
      self.cfgparser["file"]["process_time"] = str(self.processing_time)
      with open(os.path.join(destination_dir, "configs.cfg"), "w") as dump_file:
        self.cfgparser.write(dump_file)

    return total_df
