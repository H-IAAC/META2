from typing import Optional, Sequence, List, Union

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin

from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.logging import InteractiveLogger

from collections import defaultdict
import torch
import copy
import warnings

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
        plasticity_factor = 0.9,
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
        
        sum_params = sum([i.numel() for i in self.model.parameters()])
        
        sum_weighted_params = 0
        factor = 0
        for j in list(self.model.parameters())[::-1]:
            sum_weighted_params += j.numel() * (self.plasticity_factor ** factor)
            if len(j.shape) > 1:
                factor += 1
        
        lr_adapted = lr_base * sum_params / sum_weighted_params
        factor = 0
        lrs = []
        for j in list(self.model.parameters())[::-1]:
            lrs.append(lr_adapted * (self.plasticity_factor ** factor))
            if len(j.shape) > 1:
                factor += 1

        base_param_dict = copy.deepcopy(self.optimizer.state_dict())
        
        params = []
        for i, lr in zip(self.model.parameters(), lrs[::-1]):
            
            param_dict = dict()
            param_dict['params'] = i
            param_dict['lr'] = lr
            param_dict['weight_decay'] = base_param_dict["param_groups"][0]['weight_decay']
            params.append(param_dict)

        new_optimizer = type(self.optimizer)(params, lr=0.01)
        self.optimizer = new_optimizer
        self.optimizer.defaults['lr'] = 0.001 
        
        

        
