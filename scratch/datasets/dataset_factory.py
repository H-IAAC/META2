import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scratch.datasets.pamap import PAMAP2DataProcessor
from scratch.datasets.ucihar import UCIHARDataProcessor
from scratch.datasets.dsads import DSADSDataProcessor
from scratch.datasets.hapt import HAPTDataProcessor
from scratch.benchmarks.split import ClassSplit, SubjectSplit, SamplerInterface

class HARDataset(Dataset):
    '''
    Inherits from torch.Dataset class, is used to shape HAR Datasets into trainable data on Avalanche.

    Responsible for mapping activities from any order to [0, actv]. Makes Tensors from a Dataset.

    '''
    def __init__(self, dataframe : pd.DataFrame, target_column : str = "activityID", activities : list = [] , *args, **kwargs):
        """Constructor

        :param dataframe: The dataframe containing data. As it is a dataframe based on HAR,
            it should contain "timestamp", "subjectID" and "activityID" columns.
            It should also contain the same number of samles for all sensors.
        :param target_column: a string containing the target column for the Dataset.
        :param activities: a list containing all activity ID's (in ascending order)
        
      """
        self.target = np.array(list(map(lambda x, activities = activities: activities.index(int(x)), list(dataframe[target_column]))))
        self.subject = dataframe["subjectID"].to_numpy()
        self.timestamp = dataframe["timestamp"].to_numpy()
        self.grouped_samples = 0

        self.data = (dataframe.drop([target_column] + ["subjectID"] + ["timestamp"], axis = 1, inplace=False))

        for i in self.data.columns:
          if self.data.columns[0][0: -2] in i:
            self.grouped_samples += 1
        correct_order = []

        for i in range(1, self.grouped_samples + 1):
          column_str = "_" + str(i)
          for j in self.data.columns:
            if j.endswith(column_str):
              correct_order.append(j)

        print(len(correct_order))
        self.data = self.data.reindex(columns = correct_order)
        self.data = self.data.to_numpy()
        self.data = np.reshape(self.data, (self.data.shape[0], self.grouped_samples, self.data.shape[1] // self.grouped_samples))
        self.data = np.expand_dims(self.data, 1)
        self.tensors = torch.from_numpy(self.data).type(torch.float32)
        self.targets = torch.from_numpy(self.target).type(torch.int32)
        print("features shape:" , self.data.shape)

    def __len__(self):
        return self.tensors.size(0)

    def __getitem__(self, idx):
        return self.tensors[idx, :, :], self.targets[idx]

    def get_subjectID(self):
        return self.subject

    def get_activityID(self):
        return self.targets

    def get_timestamp(self):
        return self.timestamp
    
class DatasetFactory():
    '''
    A class responsible for returning Datasets.

    '''
    @staticmethod
    def get_dataset(dataset : str, config_file : str, sampler : SamplerInterface = None,
                     activities_to_use : list[int] = [1,2,3,4,5,6,7,12,13,16,17,24], **kwargs) -> list[HARDataset]:
        """
        A static function that returns HAR Datasets based on its configuration file and dataset name.

        Parameters:
        :param dataset: a string identifying the Dataset
        :param config_file: a string containing preprocessing.cfg to be read
        :param sampler: a sampler interface for spliting data into n data streams, for training
        :param activities_to_use: a list containing the activityID's that are going to be used.


        Returns:
        A list of HARDataset objects containing diferent data streams.

        """
        if dataset == "PAMAP2":
          processor = PAMAP2DataProcessor(use_cfg = True, config_file = config_file, **kwargs)
          total_df = processor.process_data()
          use_df = ClassSplit([activities_to_use]).split(total_df)
          dfs = sampler.split(use_df[0])
          datasets = []
          for df in dfs:
            datasets.append(HARDataset(df, activities = activities_to_use))

        elif dataset == "UCIHAR":
          processor = UCIHARDataProcessor(use_cfg = True, config_file = config_file, **kwargs)
          total_df = processor.process_data()
          use_df = ClassSplit([activities_to_use]).split(total_df)
          dfs = sampler.split(use_df[0])
          datasets = []
          for df in dfs:
            datasets.append(HARDataset(df, activities = activities_to_use))

        elif dataset == "DSADS":
          processor = DSADSDataProcessor(use_cfg = True, config_file = config_file, **kwargs)
          total_df = processor.process_data()
          use_df = ClassSplit([activities_to_use]).split(total_df)
          dfs = sampler.split(use_df[0])
          datasets = []
          for df in dfs:
            datasets.append(HARDataset(df, activities = activities_to_use))

        elif dataset == "HAPT":
          processor = HAPTDataProcessor(use_cfg = True, config_file = config_file, **kwargs)
          total_df = processor.process_data()
          use_df = ClassSplit([activities_to_use]).split(total_df)
          dfs = sampler.split(use_df[0])
          datasets = []
          for df in dfs:
            datasets.append(HARDataset(df, activities = activities_to_use))

        return datasets
