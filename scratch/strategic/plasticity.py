from typing import Optional, Sequence, List, Union
import math

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin

from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates import SupervisedTemplate
from avalanche.logging import InteractiveLogger

from collections import defaultdict
import torch
import copy
import warnings

def get_plasticity_lrs(plasticity_factor, lr_base, model):

    if plasticity_factor == 1:
        return [lr_base for i in model.parameters()]

    factor = 0
    for name, j in list(model.named_parameters())[::-1]:
        if 'bias' not in name:
            factor += 1
    num_layers = factor + 1

    lr_adapted = lr_base * num_layers * ( ( 1 - plasticity_factor ) / ( 1 - (plasticity_factor ** num_layers)) )
    factor = 0
    lrs = []
    for name, j in list(model.named_parameters())[::-1]:
        lrs.append(lr_adapted * (plasticity_factor ** factor))
        if 'bias' not in name:
            factor += 1
    
    return lrs[::-1]

class PlasticityStrategy(SupervisedTemplate):

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator,
        eval_every=-1,
        plasticity_factor = 0.85,
        **base_kwargs
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
        self.plasticity_factor = plasticity_factor

    def make_optimizer(self, **kwargs):
        
        lr_base = self.optimizer.param_groups[0]['lr']
        '''
        deprecated
        sum_params = sum([i.numel() for i in self.model.parameters()])
        '''
        
        lrs = get_plasticity_lrs(self.plasticity_factor, lr_base, self.model)

        base_param_dict = copy.deepcopy(self.optimizer.state_dict())
        
        params = []
        for i, lr in zip(self.model.parameters(), lrs):
            
            param_dict = dict()
            param_dict['params'] = i
            param_dict['lr'] = lr
            param_dict['weight_decay'] = base_param_dict["param_groups"][0]['weight_decay']
            params.append(param_dict)

        new_optimizer = type(self.optimizer)(params, lr=0.01)
        self.optimizer = new_optimizer
        self.optimizer.defaults['lr'] = 0.001 
        

class MPLRScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_list):
        
        self.lr_list = lr_list
        super(MPLRScheduler, self).__init__(optimizer)
    
    def update_list(self, lr_list):
        self.lr_list = lr_list

    def get_lr(self):
        return self.lr_list
    
class MetaPlasticityPlugin():

    def __init__(self, model, optimizer, meta_plasticity_factor = 0.975):
        
        self.model = model
        self.optimizer = optimizer
        self.meta_plasticity_factor = meta_plasticity_factor
        self.scheduler = None
        self.lr_base = 0.01
        self.counter = 0
        self.set_plugin()
    
    def set_plugin(self, **kwargs):
        self.counter = 0
        lr_base = self.optimizer.param_groups[0]['lr']
        
        self.lr_base = lr_base

        lrs = get_plasticity_lrs(1, lr_base, self.model)

        base_param_dict = copy.deepcopy(self.optimizer.state_dict())
        
        params = []
        for i in self.model.parameters():
            
            param_dict = dict()
            param_dict['params'] = i
            param_dict['lr'] = lr_base
            param_dict['weight_decay'] = base_param_dict["param_groups"][0]['weight_decay']
            params.append(param_dict)

        new_optimizer = type(self.optimizer)(params, lr=0.01)
        self.optimizer = new_optimizer
        self.optimizer.defaults['lr'] = 0.001 
        self.scheduler = MPLRScheduler(self.optimizer, lrs)

    def step(self):
        self.counter += 1
        self.scheduler.step()
        self.scheduler.update_list(get_plasticity_lrs((self.meta_plasticity_factor)**self.counter, self.lr_base, self.model))

class MetaPlasticityStrategy(SupervisedTemplate):

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator,
        eval_every=-1,
        meta_plasticity_factor = 1,
        **base_kwargs
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
        self.meta_plasticity_factor = meta_plasticity_factor
        self.scheduler = None
        self.lr_base = 0.01

    def make_optimizer(self, **kwargs):
        
        lr_base = self.optimizer.param_groups[0]['lr']
        
        self.lr_base = lr_base

        lrs = get_plasticity_lrs(1, lr_base, self.model)

        base_param_dict = copy.deepcopy(self.optimizer.state_dict())
        
        params = []
        for i in self.model.parameters():
            
            param_dict = dict()
            param_dict['params'] = i
            param_dict['lr'] = lr_base
            param_dict['weight_decay'] = base_param_dict["param_groups"][0]['weight_decay']
            params.append(param_dict)

        new_optimizer = type(self.optimizer)(params, lr=0.01)
        self.optimizer = new_optimizer
        self.optimizer.defaults['lr'] = 0.001 
        self.scheduler = MPLRScheduler(self.optimizer, lrs)

    def after_update(self, strategy: "SupervisedTemplate", **kwargs):
        self.scheduler.step()
        
    def before_training_exp(self,  strategy: "SupervisedTemplate", **kwargs):
        self.scheduler.update_list(get_plasticity_lrs((self.meta_plasticity_factor)**strategy.experience.current_experience, self.lr_base, self.model))