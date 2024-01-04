'''
file to save base experiment, to make testing easier
'''
import os

from scratch.utils.experimentmanager import ExperimentManager
from scratch.plotting import save_test_stream_metrics


def run_base_experiment(benchmark, strategy, eval_plugin, sklearn_metrics_plugin):
    exp = ExperimentManager()
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
    save_test_stream_metrics(
        eval_plugin.get_all_metrics(), sklearn_metrics_plugin, exp.exp_folder)
