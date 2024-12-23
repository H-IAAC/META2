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
from scratch.strategic import CEKDLossPlugin, FullyConnectedNetwork, Microtransformer, BaseConvolutionalModel, HARTransformer, CrossAttnHARTransformer
from scratch.strategic import PlasticityStrategy, MetaPlasticityStrategy
from scratch.utils.experimentmanager import ExperimentManager
import os
import json

from scratch.strategic.strategic_factory import StrategicFactory

if __name__ == "__main__":

    exp = ExperimentManager()
    exp.read_experiment(os.path.join(
        exp.get_dir_path("experiments"), exp.env_parser.get('results', "experiment_file")))

    # used to debug
    exp.print_attributes('exp_parser')

    # Change benchmark/dataset
    scenario_cfg = exp.exp_parser.get("benchmark", "scenario_cfg")
    dataset_cfg = exp.exp_parser.get("benchmark", "dataset_cfg")
    benchmark = BenchmarkFactory.generate_benchmark(scenario_cfg, dataset_cfg)

    # Experiment setup below, changes model, strategy, replay, etc.
    num_classes = 18
    # TODO: Instanciar modelo, otimizador e critério utilizando cfg
    if exp.exp_parser.get("benchmark", "name") == "UCIHAR_TI":      
        
        model = HARTransformer((1, 128, 9), 6, 1, 6, dropout=0.15, sensor_group=1)
        num_classes = 6
        '''
        model = BaseConvolutionalModel(height=9, width=128,
                                      output_classes = 6)
        num_classes = 6'''

    if exp.exp_parser.get("benchmark", "name") == "PAMAP_TI":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HARTransformer((1, 104, 27), 8, 1, 12, dropout=0.15, sensor_group=1)
        '''model = FullyConnectedNetwork(input_shape=(1, 104, 27),
                                      hidden_layer_dimensions=[486, 243, 121],
                                      num_classes=12)'''
        num_classes=12
    if exp.exp_parser.get("benchmark", "name") == "DSADS_TI":
        '''model = FullyConnectedNetwork(input_shape=(1, 125, 45),
                                      hidden_layer_dimensions=[
                                          405, 202, 202, 101],
                                      num_classes=19)'''
        model = HARTransformer((1, 125, 45), 2, 1, 19, dropout=0.15, sensor_group=1)
        num_classes=18
    if exp.exp_parser.get("benchmark", "name") == "HAPT_TI":
        '''model = FullyConnectedNetwork(input_shape=(1, 128, 6),
                                      hidden_layer_dimensions=[1122, 561, 280],
                                      num_classes=12)'''
        model = CrossAttnHARTransformer((1, 128, 6), 4, 12, dropout=0.15, sensor_group=3, wordsize=128)
        num_classes=12

    optimizer = torch.optim.Adam(model.parameters(), lr=exp.exp_parser.getfloat('training', 'learning_rate'), weight_decay=exp.exp_parser.getfloat('training', 'weight_decay'))
    criterion = CrossEntropyLoss()

    sklearn_metrics_plugin = ClassPrecisionPlugin(num_classes)
    loss_plugin = CEKDLossPlugin()
    es_plugin = TrainEarlyStoppingPlugin(10, 0.001)

    avl_plugins = StrategicFactory.init_plugins(exp.get_plugins_list(), exp.exp_parser.get("benchmark", "name"))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True,
                         experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # TODO: Instanciar estratégia utilizando cfg

    if exp.exp_parser.getfloat('training', 'meta_plasticity') < 1:
        strategy = MetaPlasticityStrategy(
            model, optimizer, loss_plugin,
            evaluator=eval_plugin, plugins=[
                sklearn_metrics_plugin, loss_plugin, es_plugin] + avl_plugins,
            train_mb_size = exp.exp_parser.getint('training', 'batch_size'), eval_mb_size=exp.exp_parser.getint('training', 'batch_size'),
            train_epochs=exp.exp_parser.getint('training', 'epochs'), device=device, meta_plasticity_factor=exp.exp_parser.getfloat('training', 'meta_plasticity'))

    elif exp.exp_parser.getfloat('training', 'plasticity_factor') < 1:

    
        strategy = PlasticityStrategy(
            model, optimizer, loss_plugin,
            evaluator=eval_plugin, plugins=[
                sklearn_metrics_plugin, loss_plugin, es_plugin] + avl_plugins,
            train_mb_size = exp.exp_parser.getint('training', 'batch_size'), eval_mb_size=exp.exp_parser.getint('training', 'batch_size'),
            train_epochs=exp.exp_parser.getint('training', 'epochs'), plasticity_factor = exp.exp_parser.getfloat('training', 'plasticity_factor'), device=device)
    else:

        strategy = Naive(
            model, optimizer, loss_plugin,
            evaluator=eval_plugin, plugins=[
                sklearn_metrics_plugin, loss_plugin, es_plugin] + avl_plugins,
            train_mb_size = exp.exp_parser.getint('training', 'batch_size'), eval_mb_size=exp.exp_parser.getint('training', 'batch_size'),
            train_epochs=exp.exp_parser.getint('training', 'epochs'), device=device)
        
    
    # Here's how you run the experiment'
    result_dict = run_base_experiment(benchmark, strategy, eval_plugin,
                        sklearn_metrics_plugin)

    
    # Do not forget eval_plugin and sklearn_metrics_plugin
