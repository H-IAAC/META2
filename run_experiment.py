'''
based on standard test
'''

import torch
from torch.nn import CrossEntropyLoss
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EarlyStoppingPlugin
from scratch.utils.base_experiments import run_base_experiment
from scratch.utils.runtime_plugins import ClassPrecisionPlugin, TrainEarlyStoppingPlugin
from scratch.benchmarks import BenchmarkFactory
from scratch.strategic import CEKDLossPlugin, FullyConnectedNetwork

from scratch.utils.experimentmanager import ExperimentManager
import os
import json

from scratch.strategic.strategic_factory import StrategicFactory

if __name__ == "__main__":

    exp = ExperimentManager()
    exp.read_experiment(os.path.join(
        exp.get_dir_path("experiments"), "default.cfg"))

    # used to debug
    exp.print_attributes('exp_parser')

    # Change benchmark/dataset
    scenario_cfg = exp.exp_parser.get("benchmark", "scenario_cfg")
    dataset_cfg = exp.exp_parser.get("benchmark", "dataset_cfg")
    benchmark = BenchmarkFactory.generate_benchmark(scenario_cfg, dataset_cfg)

    # Experiment setup below, changes model, strategy, replay, etc.

    # TODO: Instanciar modelo, otimizador e critério utilizando cfg
    if exp.exp_parser.get("benchmark", "name") == "UCIHAR_TI":
        model = FullyConnectedNetwork(input_shape=(1, 128, 9),
                                      hidden_layer_dimensions=[512, 256, 128],
                                      num_classes=6)
    if exp.exp_parser.get("benchmark", "name") == "PAMAP_TI":
        model = FullyConnectedNetwork(input_shape=(1, 104, 31),
                                      hidden_layer_dimensions=[486, 243, 121],
                                      num_classes=12)
    if exp.exp_parser.get("benchmark", "name") == "DSADS_TI":
        model = FullyConnectedNetwork(input_shape=(1, 125, 45),
                                      hidden_layer_dimensions=[
                                          405, 202, 202, 101],
                                      num_classes=19)
    if exp.exp_parser.get("benchmark", "name") == "HAPT_TI":
        model = FullyConnectedNetwork(input_shape=(1, 128, 6),
                                      hidden_layer_dimensions=[1122, 561, 280],
                                      num_classes=12)

    optimizer = torch.optim.Adam(model.parameters(), lr=exp.exp_parser.getfloat('training', 'learning_rate'), weight_decay=exp.exp_parser.getfloat('training', 'weight_decay'))
    criterion = CrossEntropyLoss()

    sklearn_metrics_plugin = ClassPrecisionPlugin(6)
    loss_plugin = CEKDLossPlugin()
    es_plugin = TrainEarlyStoppingPlugin(2, 0.01)

    avl_plugins = StrategicFactory.init_plugins(exp.get_plugins_list())

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True,
                         experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # TODO: Instanciar estratégia utilizando cfg
    strategy = Naive(
        model, optimizer, loss_plugin,
        evaluator=eval_plugin, plugins=[
            sklearn_metrics_plugin, loss_plugin, es_plugin] + avl_plugins,
        train_mb_size = exp.exp_parser.getint('training', 'batch_size'), eval_mb_size=exp.exp_parser.getint('training', 'batch_size'),
        train_epochs=exp.exp_parser.getint('training', 'epochs'), device=device)
    
    # Here's how you run the experiment
    result_dict = run_base_experiment(benchmark, strategy, eval_plugin,
                        sklearn_metrics_plugin)

    
    # Do not forget eval_plugin and sklearn_metrics_plugin
