'''
based on standar test
'''

import torch
from torch.nn import CrossEntropyLoss
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.supervised import Naive

from scratch.utils.base_experiments import run_base_experiment
from scratch.utils.runtime_plugins import ClassPrecisionPlugin
from scratch.benchmarks import BenchmarkFactory
from scratch.strategic import CEKDLossPlugin, WAMDFPlugin, BaseConvolutionalModel

from scratch.utils.experimentmanager import ExperimentManager
import os

if __name__ == "__main__":

    print("Iniciando Experimento...")

    exp = ExperimentManager()
    exp.read_experiment(os.path.join(exp.get_dir_path("experiments"), "waadb-har.cfg"))

    #Change benchmark/dataset
    print("Preparando Benchmark...")
    scenario_cfg = exp.exp_parser.get("benchmark", "scenario_cfg")
    dataset_cfg = exp.exp_parser.get("benchmark", "dataset_cfg")
    benchmark = BenchmarkFactory.generate_benchmark(scenario_cfg, dataset_cfg)
    
    # # TODO: Instanciar modelo utilizando cfg
    # #Experiment setup below, changes model, strategy, replay, etc.
    # model = BaseConvolutionalModel(9, 128, 6)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = CrossEntropyLoss()

    # sklearn_metrics_plugin = ClassPrecisionPlugin(6)
    # loss_plugin = CEKDLossPlugin()
    # wamdf = WAMDFPlugin()
    
    # eval_plugin = EvaluationPlugin(
    #     accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    #     loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    #     forgetting_metrics(experience=True, stream=True),
    #     loggers=[InteractiveLogger()])

    # replay_plugin = ReplayPlugin(mem_size = 48)

    # strategy = Naive(
    #     model, optimizer, loss_plugin,
    #     evaluator = eval_plugin, plugins=[sklearn_metrics_plugin, loss_plugin, wamdf, replay_plugin], train_mb_size=32, eval_mb_size=128, train_epochs = 6)
    
    # #Here's how you run the experiment
    # run_base_experiment(benchmark, strategy, eval_plugin, sklearn_metrics_plugin)
    
    # #Do not forget eval_plugin and sklearn_metrics_plugin