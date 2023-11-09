import pandas as pd
import numpy as np
import abc

class SamplerInterface(abc.ABC):
    
    @abc.abstractmethod
    def split(self, df : pd.DataFrame) -> list[pd.DataFrame]:
      '''
      Splits the dataframe into n-dataframes.
      '''
      ...


class SubjectSplit(SamplerInterface):
    '''
    Data spliting class. Splits the dataset based on an subject sequence.
    '''
    def __init__(self, split_sequence : list[list[int]] = []):
      '''
      Constructor

      :param split_sequence: a list of n lists containing each the subjectID's for a split
      '''
      self.split_sequence = split_sequence

    def split(self, df : pd.DataFrame) -> list[pd.DataFrame]:
      '''
      Splits the dataframe into n dataframes.

      :param df: a pandas Dataframe that is going to be split into this' object n groups

      '''
      if(len(self.split_sequence) == 0):
        print("can't split dataset into 0 subject groups")
        raise ValueError

      dfs = []

      for split in self.split_sequence:
        indices = []
        for idx, i in enumerate(list(df["subjectID"])):
          if i in split:
            indices.append(idx)
        dfs.append(df.iloc[indices])

      return dfs
    
class ClassSplit(SamplerInterface):
    '''
    Data spliting class. Splits the dataset based on an activity sequence.
    '''
    def __init__(self, split_sequence : list[list[int]] = []):
      '''
      Constructor

      :param split_sequence: a list of n lists containing each the activtyID's for a split
      '''
      self.split_sequence = split_sequence

    def split(self, df : pd.DataFrame) -> list[pd.DataFrame]:
      '''
      Splits the dataframe into n dataframes.

      :param df: a pandas Dataframe that is going to be split into this' object n groups
      '''
      if(len(self.split_sequence) == 0):
        print("can't split dataset into 0 subject groups")
        raise ValueError

      dfs = []

      for split in self.split_sequence:
        indices = []
        for idx, i in enumerate(list(df["activityID"])):
          if i in split:
            indices.append(idx)
        dfs.append(df.iloc[indices])

      return dfs