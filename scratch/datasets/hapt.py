# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 10 2021

@co-author: Rebecca Adaimi

HAPT dataset loading and preprocessing
Participants 29 and 30 used as test data 
"""

import numpy as np
import os
import pandas as pd
import os
import time
from sklearn.manifold import TSNE
from scratch.datasets.processing import apply_fourier_transform
from scratch.plotting.processing_plots import create_count_histogram, plot_similarity_matrix, plot_clustering
import configparser
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scratch.utils.experimentmanager import ExperimentManager

SAMPLING_FREQ = 50 # Hz

SLIDING_WINDOW_LENGTH = int(2.56*SAMPLING_FREQ)

sensors = 6

SLIDING_WINDOW_STEP = int(SLIDING_WINDOW_LENGTH/2)


def standardize(mat):
    """ standardize each sensor data columnwise"""
    for i in range(mat.shape[1]):
        mean = np.mean(mat[:, [i]])
        std = np.std(mat[:, [i]])
        mat[:, [i]] -= mean
        mat[:, [i]] /= std

    return mat


def __rearrange(a,y,s, window, overlap):
    l, f = a.shape
    shape = (int( (l-overlap)/(window-overlap) ), window, f)
    stride = (a.itemsize*f*(window-overlap), a.itemsize*f, a.itemsize)
    X = np.lib.stride_tricks.as_strided(a, shape=shape, strides=stride)
    #import pdb; pdb.set_trace()

    l,f = y.shape
    shape = (int( (l-overlap)/(window-overlap) ), window, f)
    stride = (y.itemsize*f*(window-overlap), y.itemsize*f, y.itemsize)
    Y = np.lib.stride_tricks.as_strided(y, shape=shape, strides=stride)
    Y = Y.max(axis=1)

    l,f = s.shape
    shape = (int( (l-overlap)/(window-overlap) ), window, f)
    stride = (s.itemsize*f*(window-overlap), s.itemsize*f, s.itemsize)
    S = np.lib.stride_tricks.as_strided(s, shape=shape, strides=stride)
    S = S.max(axis=1)


    return X, Y.flatten(), S.flatten()

def normalize(data):
    """ l2 normalization can be used"""

    y = data[:, 0].reshape(-1, 1)
    X = np.delete(data, 0, axis=1)
    transformer = Normalizer(norm='l2', copy=True).fit(X)
    X = transformer.transform(X)

    return np.concatenate((y, X), 1)


def normalize_df(data):
    """ l2 normalization can be used"""

    #y = data[:, 0].reshape(-1, 1)
    #X = np.delete(data, 0, axis=1)
    transformer = Normalizer(norm='l2', copy=True).fit(data)
    data = transformer.transform(data)

    return data

def min_max_scaler(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    return data

def read_dir(DIR, user_test):

    folder1=sorted(os.listdir(DIR))
    #import pdb; pdb.set_trace()

    labels = np.genfromtxt(os.path.join(DIR,folder1[-1]), delimiter=' ')
    accel_files = folder1[:int(len(folder1[:-1])/2)]
    gyro_files = folder1[int(len(folder1[:-1])/2):-1]

    train_d = []
    test_d = []
    test_subject_d = []
    train_subject_d = []
    for a_file,g_file in zip(accel_files,gyro_files):
        #import pdb; pdb.set_trace()
        a_ff = os.path.join(DIR, a_file)
        g_ff = os.path.join(DIR, g_file)
        a_df = np.genfromtxt(a_ff, delimiter=' ')
        g_df = np.genfromtxt(g_ff, delimiter=' ')
        ss = a_file.split('.')[0].split('_')
        exp, user = int(ss[1][-2:]), int(ss[2][-2:])

        indices = labels[labels[:,0]==exp]
        indices = indices[indices[:,1]==user]
        for ii in range(len(indices)):
            a_sub = a_df[int(indices[ii][-2]):int(indices[ii][-1]),:]
            g_sub = g_df[int(indices[ii][-2]):int(indices[ii][-1]),:]
            subject_id = np.full(len(a_sub), user)
            if user in user_test:
                test_d.extend(np.append(np.append(a_sub,g_sub,axis=1),np.array([indices[ii][-3]]*len(a_sub))[:,None],axis=1))
                test_subject_d.extend(subject_id)
            else:
                train_d.extend(np.append(np.append(a_sub,g_sub,axis=1),np.array([indices[ii][-3]]*len(a_sub))[:,None],axis=1))
                train_subject_d.extend(subject_id)

    train_x = np.array(train_d)[:,:-1]
    test_x = np.array(test_d)[:,:-1]
    train_y = np.array(train_d)[:,-1]
    test_y = np.array(test_d)[:,-1]
    train_subject = np.array(train_subject_d)
    test_subject = np.array(test_subject_d)

    train_x = normalize(train_x)
    test_x = normalize(test_x)
    train_x, train_y, subject_train = __rearrange(train_x, train_y.astype(int).reshape((-1,1)),train_subject.astype(int).reshape((-1,1)), SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    test_x, test_y, subject_test = __rearrange(test_x, test_y.astype(int).reshape((-1,1)),test_subject.astype(int).reshape((-1,1)), SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
   
    return train_x, train_y, test_x, test_y, subject_train, subject_test

class HAPTDataProcessor():
  '''
  Data processor class for HAPT dataset.

  Responsible for reading data, checking if there is a saved file with the processed data and processes data.
  '''

  def __init__(self, persist : bool = True, use_cfg : bool = True, config_file : str = "HAPT_2.56_50.cfg",
      radix_name : str = "base", frequency : float = 50.0, time_window : float = 2.56, use_test_data : bool = True,
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
      self.namecols =["IMU_body_accX", "IMU_body_accY", "IMU_body_accZ", "IMU_body_gyroX",
                      "IMU_body_gyroY", "IMU_body_gyroZ"]

      #Mudar a região abaixo de acordo com a solução proposta pelo Wallace
      if(use_cfg):
        self.cfgparser = configparser.ConfigParser()
        self.cfgparser.optionxform = str
        self.cfgparser.read(os.path.join(self.exp.get_dir_path("configs"), "preprocessing", self.config_file))
        self.persist = self.cfgparser["file"]["persist"]
        self.file_dir = os.path.join(self.exp.get_dir_path("datasets"), "Raw", "HAPT")   
        self.folder_name = (str(self.cfgparser["file"]["radix_name"])+ "_" + self.cfgparser["base_parameters"]["time_window"] + "_" + self.cfgparser["base_parameters"]["frequency"])
        self.frequency = float(self.cfgparser["base_parameters"]["frequency"])
        self.time_window = float(self.cfgparser["base_parameters"]["time_window"])
        
        self.drop_cols = [i for i in self.cfgparser['use_cols'] if not self.cfgparser['use_cols'].getboolean(i)]

        self.fft_cols =  [i for i in self.cfgparser['fft_cols'] if self.cfgparser['fft_cols'].getboolean(i)]

        self.use_cols = [i for i in self.namecols if i not in self.drop_cols]

      else:
        self.cfgparser = None
        self.persist = persist
        self.radix_name = radix_name
        self.file_dir = os.path.join(self.exp.get_dir_path("datasets"), "Raw", "HAPT")   
        self.folder_name = (str(radix_name)+ str("_")+ str(time_window) + "_" + str(frequency))
        self.frequency = frequency
        self.time_window = time_window
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
                                   'time_window' : self.time_window
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
      pd.DataFrame : processed HAPT data.
    '''
    os.makedirs(os.path.join(self.exp.get_dir_path("datasets"), "Processed", "HAPT") , exist_ok = True)
    start_time = time.time()
    #identify already created processing parameters
    if(not self.use_cfg):
        self.cfgparser = self.get_param_cfg()
    
    for i in os.listdir(os.path.join(self.exp.get_dir_path("datasets"), "Processed", "HAPT")):
      rdparser = configparser.ConfigParser()
      rdparser.optionxform = str  
      
      rdparser.read(os.path.join(self.exp.get_dir_path("datasets"), "Processed", "HAPT", i, "configs.cfg"))
        
      if not (False in [self.cfgparser[i] == rdparser[i] for i in rdparser.sections() if i != "file"]):
          print("Found processed data according to pre-processing, copying data...")
          total = pd.read_table(os.path.join(self.exp.get_dir_path("datasets"), "Processed", "HAPT", i, "dataset.csv"), sep='\s+')
          return total

    # Check whether frequency and time_window are compatible with the dataset
    if(not (self.time_window == 2.56)):
      print(f"Warning! time_window: {self.time_window} seconds is not compatible with HAPT, defaulting to 2.56 sec / 50 Hz")
      self.frequency = 50
      self.time_window = 2.56

    if(not os.path.exists(self.file_dir)):
      print("Não foram encontrados os arquivos do HAPT no sistema")

    path = os.path.join(self.file_dir, 'RawData')
  
    user_tst = np.arange(1,11)
    train_data, train_labels, test_data, test_labels, train_subject, test_subject = read_dir(path, user_tst)
    
    train_X = train_data.reshape((len(train_data), int(sensors * SLIDING_WINDOW_LENGTH)))
    test_X = test_data.reshape((len(test_data), int(sensors * SLIDING_WINDOW_LENGTH)))

    # Calculate the decimal precision of the floats in the array
    decimal_precision = len(str(np.max(np.abs(train_X))).split('.')[1])

    data_train = np.hstack((train_X, train_labels[np.newaxis].T, train_subject[np.newaxis].T))
    data_test = np.hstack((test_X, test_labels[np.newaxis].T, test_subject[np.newaxis].T))
    data = np.vstack((data_train, data_test))

    feature_names = []
    for i in range(int(self.frequency * self.time_window)):
      for sensor_name in self.use_cols:
          feature_names.append(sensor_name + "_" + str(i+1))
    column_names = feature_names + ["activityID", "subjectID"]
    total = pd.DataFrame(data = data, columns = column_names)
    # Drop the features that are not used
    total.drop(self.drop_cols, axis = 1, inplace = True)

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
      destination_dir = os.path.join(self.exp.get_dir_path("datasets"), "Processed", "HAPT", self.folder_name)
      os.makedirs(destination_dir , exist_ok = True)

    if(self.persist):
      total.to_csv(os.path.join(destination_dir, "dataset.csv"), sep=' ', index = False)
    
    #Create first graph
    create_count_histogram(total, activities = {1 : "Walking", 2 : "Walking upstairs", 3 : "Walking downstairs", 4 : "Sitting", 5 : "Standing", 6 : "Laying", 
                                                7: "Stand_to_sit", 8: "Sit_to_stand", 9: "Sit_to_lie", 10: "Lie_to_sit", 11: "Stand_to_lie", 12: "Lie_to_stand"},
                           filepath = destination_dir, num_subjects = 30)

    #Create second graph
    reducer_class_columns = [i for i in correct_column_order if i not in ["activityID"]]
    reducer = TSNE(n_components=2, perplexity = 32*4 + 4, early_exaggeration = (26*1.5)+1)
    plot_similarity_matrix(reducer,  ((total.reindex(columns = ["activityID"] + reducer_class_columns)).drop(labels = ["subjectID"], axis = 'columns')).to_numpy(), 
                           label = {1 : "Walking", 2 : "Walking upstairs", 3 : "Walking downstairs", 4 : "Sitting", 5 : "Standing", 6 : "Laying", 
                                                7: "Stand_to_sit", 8: "Sit_to_stand", 9: "Sit_to_lie", 10: "Lie_to_sit", 11: "Stand_to_lie", 12: "Lie_to_stand"}, filepath = destination_dir, title = "classes heatmap")
    
    plot_clustering(reducer,  ((total.reindex(columns = ["activityID"] + reducer_class_columns)).drop(labels = ["subjectID"], axis = 'columns')).to_numpy(), 
                    labels = {1 : "Walking", 2 : "Walking upstairs", 3 : "Walking downstairs", 4 : "Sitting", 5 : "Standing", 6 : "Laying", 
                                                7: "Stand_to_sit", 8: "Sit_to_stand", 9: "Sit_to_lie", 10: "Lie_to_sit", 11: "Stand_to_lie", 12: "Lie_to_stand"},
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
