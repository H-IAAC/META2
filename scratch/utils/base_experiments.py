'''
file to save base experiment, to make testing easier
'''
import os

from scratch.utils.experimentmanager import ExperimentManager
from scratch.plotting import save_test_stream_metrics
import json

def get_scenario_dict(exp):
    scenario_dict = dict()
    scenario_dict['scenario_id'] = exp.exp_parser.get('scenario_configs', "scenario_id")
    scenario_dict['activities'] = exp.exp_parser.get('scenario_configs', "activities")
    scenario_dict['classes_per_experience'] = exp.exp_parser.get('scenario_configs', "classes_per_experience")
    scenario_dict['train_subjects'] = exp.exp_parser.get('scenario_configs', "train_subjects")
    scenario_dict['test_subjects'] = exp.exp_parser.get('scenario_configs', "test_subjects")
    return scenario_dict

def get_dataset_dict(exp):
    dataset_dict = dict()
    dataset_dict['used_sensors'] = exp.exp_parser.get('dataset_configs', "used_sensors")
    dataset_dict['dataset_id'] = exp.exp_parser.get('dataset_configs', "dataset_id")
    dataset_dict['frequency'] = exp.exp_parser.get('dataset_configs', "frequency")
    dataset_dict['time_window'] = exp.exp_parser.get('dataset_configs', "time_window")
    return dataset_dict
    
def get_training_dict(exp):
    training_dict = dict()
    for param in exp.exp_parser.options("training"):
        training_dict[param] = exp.exp_parser.get('training', param)
    return training_dict


def run_base_experiment(benchmark, strategy, eval_plugin, sklearn_metrics_plugin):
    exp = ExperimentManager()
    metadata = dict()
    print('Starting experiment...')
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(strategy.eval(benchmark.test_stream))

    # Maybe change with Experiment Manager the filepath to which we're saving results
    result_dict = save_test_stream_metrics(
        eval_plugin.get_all_metrics(), sklearn_metrics_plugin, exp.exp_folder)
    
    metadata["results"] = result_dict

    #Get train_params
    metadata["training_parameters"] = get_training_dict(exp)

    #Get dataset_params
    metadata["dataset"] = get_dataset_dict(exp)

    #Get scenario_params
    metadata["scenario"]  = get_scenario_dict(exp)

    os.makedirs(os.path.join(exp.exp_folder), exist_ok=True)
    with open(os.path.join(exp.exp_folder, 'metadata.json'), 'w') as json_file:
        json.dump(metadata, json_file)
