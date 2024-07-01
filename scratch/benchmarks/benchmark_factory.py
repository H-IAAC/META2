import avalanche as avl
from avalanche.benchmarks.scenarios.generic_scenario import CLScenario
from avalanche.benchmarks.scenarios import NCScenario
import pandas as pd
import numpy as np
import os
import torch
import configparser
from scratch.datasets.dataset_factory import DatasetFactory
from scratch.benchmarks.split import SubjectSplit, ClassSplit, RandomSubjectSplit
from scratch.utils.experimentmanager import ExperimentManager

class BenchmarkFactory():
  '''
    A class responsible for returning avalanche Benchmarks.

  '''
  @staticmethod
  def generate_benchmark(scenario_cfg : str, dataset_cfg : str) -> CLScenario:
    """
      A static function that returns benchmarks according to parameters in scenario_cfg and dataset_cfg

      Parameters:
      :param scenario_cfg: a string identifying the configuration file containing scenario parameters
      :param dataset_cfg: a string identifying the configuration file containing dataset parameters

      Returns:
      A benchmark that contains dataset data in a Continual Learning scenario.

    """
    file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    exp = ExperimentManager()
    benchmark_parser = configparser.ConfigParser()  
    benchmark_parser.optionxform = str
    benchmark_parser.read(os.path.join(exp.get_dir_path("scenario"), scenario_cfg))
    benchmark_parser.read(os.path.join(exp.get_dir_path("preprocessing"), dataset_cfg))
    
    if(benchmark_parser["scenario_configs"]["scenario_id"] == "task_incremental"):
      return BenchmarkFactory.generate_ti_benchmark(benchmark_parser["base_parameters"]["dataset_id"],
                                                    benchmark_parser.getboolean("scenario_configs", "use_subject_fraction_to_split"), 
                                                    float(benchmark_parser["scenario_configs"]["train_subjects_fraction"]),
                                                    [int(i) for i in benchmark_parser["scenario_configs"]["train_subjects"].split(',')],
                                                    [int(i) for i in benchmark_parser["scenario_configs"]["test_subjects"].split(',')],
                                                    [int(i) for i in benchmark_parser["scenario_configs"]["activities_to_use"].split(',')],
                                                    int(benchmark_parser["scenario_configs"]["classes_per_exp"]), dataset_cfg, exp)

  @staticmethod
  def generate_ti_benchmark(dataset : str, use_fraction : bool, train_fraction : float, train_subjects : list, test_subjects : list, activities : list, 
                            classes_per_exp : int, dataset_cfg : str, exp : ExperimentManager, **kwargs) -> NCScenario:
    """
      A static function that returns a task incremental benchmark

      Parameters:
      :param dataset: a string identifying the Dataset used.
      :param train_subjects: subjects used for training.
      :param test_subjects: subjects used for testing.
      :param activities: activityID's of data to be used.
      :param classes_per_exp: the number of classes to be used in each task.
      :param dataset_cfg: a string with the cfg file's name to use.

      Returns:
      A benchmark that contains HAR data in a Task Incremental scenario.

    """
    if not use_fraction:
      split_list = []
      split_list.append(train_subjects)
      split_list.append(test_subjects)
      split_dataset = DatasetFactory.get_dataset(dataset, config_file = dataset_cfg, sampler = SubjectSplit(split_list), activities_to_use = activities)
    else:
      split_dataset = DatasetFactory.get_dataset(dataset, config_file = dataset_cfg, sampler = RandomSubjectSplit(train_fraction), activities_to_use = activities)
    
    
    if 'fixed_class_order' in kwargs:
      benchmark = avl.benchmarks.generators.nc_benchmark(train_dataset = split_dataset[0], test_dataset = split_dataset[1], n_experiences = len(activities) // classes_per_exp, task_labels =  False, fixed_class_order = kwargs.get('fixed_class_order'))
    # shuffle = False -> para reprodutibilidade
    elif 'per_exp_classes' in kwargs: 
      benchmark = avl.benchmarks.generators.nc_benchmark(train_dataset = split_dataset[0], test_dataset = split_dataset[1], shuffle=False, n_experiences = len(activities) // classes_per_exp, task_labels =  False, per_exp_classes = kwargs.get('per_exp_classes'))
    else:
      if len(activities) % classes_per_exp == 0:
        benchmark = avl.benchmarks.generators.nc_benchmark(train_dataset = split_dataset[0], test_dataset = split_dataset[1], shuffle=False, n_experiences = len(activities) // classes_per_exp, task_labels =  False)
      else:
        first_exp_class = {((len(activities) + 1) // classes_per_exp) - 1: len(activities) % classes_per_exp}
        benchmark = avl.benchmarks.generators.nc_benchmark(train_dataset = split_dataset[0], test_dataset = split_dataset[1], shuffle=False, n_experiences = (len(activities) + 1) // classes_per_exp, task_labels =  False, per_exp_classes=first_exp_class)

    exp.set_scenario_params({"scenario_id": "task_incremental", "activities": activities, "classes_per_experience": classes_per_exp})
    exp.set_dataset_params(dataset_cfg)
    
    return benchmark