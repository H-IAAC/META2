import numpy as np
import pandas as pd
import os
import time
from sklearn.manifold import TSNE
from scratch.datasets.processing import apply_fourier_transform
from scratch.plotting.processing_plots import create_count_histogram, plot_similarity_matrix, plot_clustering
import configparser
from scratch.utils.experimentmanager import ExperimentManager

class UCIHARDataProcessor():
  '''
  Data processor class for PAMAP2 dataset.

  Responsible for reading data, checking if there is a saved file with the processed data and processes data.
  '''

  def __init__(self, persist : bool = True, use_cfg : bool = True, config_file : str = "UCIHAR_2.56_50.cfg",
      radix_name : str = "base", frequency : float = 50.0, time_window : float = 2.56, use_test_data : bool = True,
      drop_cols : list= [],
      normalize_cols : list = [],
      fft_cols : list = []):
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
      self.namecols =["IMU_body_accX", "IMU_body_accY", "IMU_body_accZ", "IMU_body_gyroX",
                      "IMU_body_gyroY", "IMU_body_gyroZ", "IMU_total_accX", "IMU_total_accY", "IMU_total_accZ"]

      #Mudar a região abaixo de acordo com a solução proposta pelo Wallace
      if(use_cfg):
        self.cfgparser = configparser.ConfigParser()
        self.cfgparser.optionxform = str
        self.cfgparser.read(os.path.join(self.exp.get_dir_path("configs"), "preprocessing", self.config_file))
        self.persist = self.cfgparser["file"]["persist"]
        self.file_dir = os.path.join(self.exp.get_dir_path("datasets"), "Raw", "UCIHAR")   
        self.folder_name = (str(self.cfgparser["file"]["radix_name"])+ "_" + self.cfgparser["base_parameters"]["time_window"] + "_" + self.cfgparser["base_parameters"]["frequency"])
        self.frequency = float(self.cfgparser["base_parameters"]["frequency"])
        self.time_window = float(self.cfgparser["base_parameters"]["time_window"])
        self.use_test_data = self.cfgparser["base_parameters"].getboolean("use_test_data")
        self.drop_cols = [i for i in self.cfgparser['use_cols'] if not self.cfgparser['use_cols'].getboolean(i)]

        self.fft_cols =  [i for i in self.cfgparser['fft_cols'] if self.cfgparser['fft_cols'].getboolean(i)]

        self.use_cols = [i for i in self.namecols if i not in self.drop_cols]

      else:
        self.cfgparser = None
        self.persist = persist
        self.radix_name = radix_name
        self.file_dir = os.path.join(self.exp.get_dir_path("datasets"), "Raw", "UCIHAR")   
        self.folder_name = (str(radix_name)+ str("_")+ str(time_window) + "_" + str(frequency))
        self.frequency = frequency
        self.time_window = time_window
        self.use_test_data = use_test_data
        self.drop_cols = drop_cols
        self.fft_cols = fft_cols
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
                                   'use_test_data' : self.use_test_data,
                                  }
    paramcfg["file"] = {'persist': self.persist,
                        'radix_name' : self.radix_name,
                        'process_time' : "0.0"
                        }
    paramcfg["use_cols"] = {i : (i in self.use_cols) for i in self.namecols}
    paramcfg["fft_cols"] = {i : (i in self.fft_cols) for i in self.namecols}
    return paramcfg


  def process_data(self) -> pd.DataFrame:
    '''
    When called, this function processes the dataset according to its parameters.

    Returns:
      pd.DataFrame : processed PAMAP2 data.
    '''
    os.makedirs(os.path.join(self.exp.get_dir_path("datasets"), "Processed", "UCIHAR") , exist_ok = True)
    start_time = time.time()
    #identify already created processing parameters
    if(not self.use_cfg):
        self.cfgparser = self.get_param_cfg()
    
    for i in os.listdir(os.path.join(self.exp.get_dir_path("datasets"), "Processed", "UCIHAR")):
      rdparser = configparser.ConfigParser()
      rdparser.optionxform = str  
      
      rdparser.read(os.path.join(self.exp.get_dir_path("datasets"), "Processed", "UCIHAR", i, "configs.cfg"))
        
      if not (False in [self.cfgparser[i] == rdparser[i] for i in rdparser.sections() if i != "file"]):
          print("Found processed data according to pre-processing, copying data...")
          total = pd.read_table(os.path.join(self.exp.get_dir_path("datasets"), "Processed", "UCIHAR", i, "dataset.csv"), sep='\s+')
          return total

    # Check whether frequency and time_window are compatible with the dataset
    if(not (self.time_window == 2.56)):
      print(f"Warning! time_window: {self.time_window} seconds is not compatible with UCIHAR, defaulting to 2.56 sec / 50 Hz")
      self.frequency = 50
      self.time_window = 2.56

    if(not os.path.exists(self.file_dir)):
      print("Não foram encontrados os arquivos do UCIHAR no sistema")

    std_names = {"body_acc_x": "IMU_body_accX", "body_acc_y": "IMU_body_accY", "body_acc_z": "IMU_body_accZ",
                 "body_gyro_x": "IMU_body_gyroX", "body_gyro_y": "IMU_body_gyroY", "body_gyro_z": "IMU_body_gyroZ", 
                 "total_acc_x": "IMU_total_accX", "total_acc_y": "IMU_total_accY", "total_acc_z": "IMU_total_accZ"} 
                 
    # Reads raw data
    df_list = []
    for file in os.listdir(os.path.join(self.file_dir, "UCI HAR Dataset", "train", "Inertial Signals")):
      df_list.append(pd.read_table(os.path.join(self.file_dir, "UCI HAR Dataset", "train", "Inertial Signals", file), sep="\s+", names = [std_names[file[:-10]] + "_" + str(i + 1) for i in range(128)]))
    y_train = pd.read_table(os.path.join(self.file_dir, "UCI HAR Dataset", "train", "y_train.txt"), sep="\s+", names = ["activityID"])
    subjects = pd.read_table(os.path.join(self.file_dir, "UCI HAR Dataset", "train", "subject_train.txt"), sep="\s+", names = ["subjectID"])
    sensors =  pd.concat(df_list, axis = 1)
    train_total = pd.concat([y_train, subjects, sensors], axis = 1)

    # Concatenate "test" and "train" data
    if(self.use_test_data):
      test_list = []
      for file in os.listdir(os.path.join(self.file_dir, "UCI HAR Dataset", "test", "Inertial Signals")):
        test_list.append( pd.read_table(os.path.join(self.file_dir, "UCI HAR Dataset", "test", "Inertial Signals", file), sep="\s+", names = [std_names[file[:-9]] + "_" + str(i + 1) for i in range(128)]))
      y_test = pd.read_table(os.path.join(self.file_dir, "UCI HAR Dataset", "test", "y_test.txt"), sep="\s+", names = ["activityID"])
      test_subjects = pd.read_table(os.path.join(self.file_dir, "UCI HAR Dataset", "test", "subject_test.txt"), sep="\s+", names = ["subjectID"])
      test_sensors =  pd.concat(test_list, axis = 1)
      test_total = pd.concat([y_test, test_subjects, test_sensors], axis = 1)
      total = pd.concat([train_total, test_total], axis = 0)
    else:
      total = train_total

    # Drop the features that are not used
    total.drop(self.drop_cols, axis = 1, inplace = True)
    
    #Add column labels
    feature_names = []
    for sensor_name in self.use_cols:
      for i in range(int(self.frequency * self.time_window)):
          feature_names.append(sensor_name + "_" + str(i+1))

    correct_column_order = ["subjectID"] + feature_names + ["activityID"]
    column_names = ["activityID", "subjectID"] + feature_names
    
    #ApplyFFT
    for fft_sensor in self.fft_cols:
      lower_index = column_names.index(fft_sensor + "_1")
      top_limit = lower_index + int(self.frequency * self.time_window) #exclusive
      apply_fourier_transform(total.to_numpy(), lower_index, top_limit) #using df.to_numpy() casts fft to original dataframe
    
    total = total.set_axis([i for i in range(total.shape[0])], axis = "index")
    total = total.reindex(columns = correct_column_order)
    
    if(self.persist):
      destination_dir = os.path.join(self.exp.get_dir_path("datasets"), "Processed", "UCIHAR", self.folder_name)
      os.makedirs(destination_dir , exist_ok = True)

    if(self.persist):
      total.to_csv(os.path.join(destination_dir, "dataset.csv"), sep=' ', index = False)
    
    #Create first graph
    create_count_histogram(total, activities = {1 : "Walking", 2 : "Walking upstairs", 3 : "Walking downstairs", 4 : "Sitting", 5 : "Standing", 6 : "Laying"},
                           filepath = destination_dir, num_subjects = 30)

    #Create second graph
    reducer_class_columns = [i for i in correct_column_order if i not in ["activityID"]]
    reducer = TSNE(n_components=2, perplexity = 32*4 + 4, early_exaggeration = (26*1.5)+1)
    plot_similarity_matrix(reducer,  ((total.reindex(columns = ["activityID"] + reducer_class_columns)).drop(labels = ["subjectID"], axis = 'columns')).to_numpy(), 
                           label = {1 : "Walking", 2 : "Walking upstairs", 3 : "Walking downstairs", 4 : "Sitting", 5 : "Standing", 6 : "Laying"}, filepath = destination_dir, title = "classes heatmap")
    
    plot_clustering(reducer,  ((total.reindex(columns = ["activityID"] + reducer_class_columns)).drop(labels = ["subjectID"], axis = 'columns')).to_numpy(), 
                    labels = {1 : "Walking", 2 : "Walking upstairs", 3 : "Walking downstairs", 4 : "Sitting", 5 : "Standing", 6 : "Laying"},
                     title = "classes clustering", filepath = destination_dir, filename = "classes clustering")
    
    reducer_subject_columns = [i for i in correct_column_order if i not in ["subjectID"]]
    reducer = TSNE(n_components=2, perplexity = 32*4 + 4, early_exaggeration = (26*1.5)+1)
    plot_similarity_matrix(reducer,  ((total.reindex(columns = ["subjectID"] + reducer_subject_columns)).drop(labels = ["activityID"], axis = 'columns')).to_numpy(),
                           label = {i : "subject" + str(i) for i in range(1, 31)}, filepath = destination_dir, title = "subjects heatmap")
    
    plot_clustering(reducer,  ((total.reindex(columns = ["subjectID"] + reducer_subject_columns)).drop(labels = ["activityID"], axis = 'columns')).to_numpy(),
                           labels = {i : "subject" + str(i) for i in range(1, 31)}, title = "classes clustering", filepath = destination_dir, filename = "subjects clustering")

    end_time = time.time()
    print(f"Elapsed time in seconds: {round((end_time - start_time), 2)}")
    self.processing_time = round((end_time - start_time), 2)
     
    if(self.persist):
      #create cfg file to get already processed data
      self.cfgparser["file"]["process_time"] = str(self.processing_time)
      with open(os.path.join(destination_dir, "configs.cfg"), "w") as dump_file:
        self.cfgparser.write(dump_file)

    return total
