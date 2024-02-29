import pandas as pd
import numpy as np
import abc
from scratch.utils.experimentmanager import ExperimentManager

class SamplerInterface(abc.ABC):
    
    @abc.abstractmethod
    def split(self, df : pd.DataFrame) -> list:
      '''
      Splits the dataframe into n-dataframes.
      '''
      ...


class SubjectSplit(SamplerInterface):
    
    '''
    Data spliting class. Splits the dataset based on an subject sequence.
    '''
    def __init__(self, split_sequence : list = []):
      '''
      Constructor

      :param split_sequence: a list of n lists containing each the subjectID's for a split
      '''
      self.split_sequence = split_sequence

    def split(self, df : pd.DataFrame) -> list:
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
      
      exp = ExperimentManager()
      
      exp.set_scenario_params({"train_subjects": self.split_sequence[0], "test_subjects": self.split_sequence[1]})

      return dfs
    
class ClassSplit(SamplerInterface):
    '''
    Data spliting class. Splits the dataset based on an activity sequence.
    '''
    def __init__(self, split_sequence : list = []):
      '''
      Constructor

      :param split_sequence: a list of n lists containing each the activtyID's for a split
      '''
      self.split_sequence = split_sequence

    def split(self, df : pd.DataFrame) -> list:
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
    
class RandomSubjectSplit(SubjectSplit):

    def __init__(self, train_split):
      super().__init__([])
      self.train_split = train_split

    def split(self, df):
      n_subjects = np.unique(df["subjectID"].to_numpy()).shape[0]
      

      n_train_subjects = int(round(n_subjects * self.train_split))
      

      subjects = np.unique(df["subjectID"].to_numpy())
      np.random.shuffle(subjects)
      randomized_subjects = list(subjects)
      train_subjects = randomized_subjects[:n_train_subjects]
      test_subjects = randomized_subjects[n_train_subjects:]

      self.split_sequence = [train_subjects, test_subjects]

      return super().split(df)


