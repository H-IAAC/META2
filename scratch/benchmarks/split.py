import pandas as pd
import numpy as np

class SubjectSplit():

    def __init__(self, split_sequence = []):
      self.split_sequence = split_sequence

    def split(self, df):
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
    
class ClassSplit():

    def __init__(self, split_sequence = []):
      self.split_sequence = split_sequence

    def split(self, df):
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