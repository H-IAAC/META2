import avalanche as avl
import pandas as pd
import numpy as np
import os
import configparser
from scratch.datasets.dataset_factory import DatasetFactory
from scratch.datasets.dataset_factory import SubjectSplit, ClassSplit

class BenchmarkFactory():
  
  @staticmethod
  def generate_benchmark(scenario_cfg, dataset_cfg):
    file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    benchmark_parser = configparser.ConfigParser()  
    benchmark_parser.optionxform = str
    benchmark_parser.read(os.path.join(file_path, "Configs", "TAGS", scenario_cfg))
    benchmark_parser.read(os.path.join(file_path, "Configs", "preprocessing", dataset_cfg))
    if(benchmark_parser["scenario_configs"]["scenario_id"] == "task_incremental"):
      return BenchmarkFactory.generate_ti_benchmark(benchmark_parser["base_parameters"]["dataset_id"],
                                                    [int(i) for i in benchmark_parser["scenario_configs"]["train_subjects"].split(',')],
                                                    [int(i) for i in benchmark_parser["scenario_configs"]["test_subjects"].split(',')],
                                                    [int(i) for i in benchmark_parser["scenario_configs"]["activities_to_use"].split(',')],
                                                    int(benchmark_parser["scenario_configs"]["classes_per_exp"]), dataset_cfg)

  @staticmethod
  def generate_ti_benchmark(dataset, train_subjects, test_subjects, activities, classes_per_exp, dataset_cfg, **kwargs):

    split_list = []
    split_list.append(train_subjects)
    split_list.append(test_subjects)

    split_dataset = DatasetFactory.get_dataset(dataset, config_file = dataset_cfg, sampler = SubjectSplit(split_list), activities_to_use = activities)

    if 'fixed_class_order' in kwargs:
      benchmark = avl.benchmarks.generators.nc_benchmark(train_dataset = split_dataset[0], test_dataset = split_dataset[1], n_experiences = len(activities) // classes_per_exp, task_labels =  False, fixed_class_order = kwargs.get('fixed_class_order'))
    elif 'per_exp_classes' in kwargs:
      benchmark = avl.benchmarks.generators.nc_benchmark(train_dataset = split_dataset[0], test_dataset = split_dataset[1], n_experiences = len(activities) // classes_per_exp, task_labels =  False, per_exp_classes = kwargs.get('per_exp_classes'))
    else:
      benchmark = avl.benchmarks.generators.nc_benchmark(train_dataset = split_dataset[0], test_dataset = split_dataset[1], n_experiences = len(activities) // classes_per_exp, task_labels =  False)

    return benchmark