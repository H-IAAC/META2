import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCELoss

from avalanche.core import SupervisedPlugin
# from avalanche.training.regularization import cross_entropy_with_oh_targets
# from avalanche.training import cross_entropy_with_oh_targets
from avalanche.training.templates import SupervisedTemplate

class CEKDLossPlugin(SupervisedPlugin):
    """
    ICaRLLossPlugin
    Similar to the Knowledge Distillation Loss. Works as follows:
        The target is constructed by taking the one-hot vector target for the
        current sample and assigning to the position corresponding to the
        past classes the output of the old model on the current sample.
        Doesn't work if classes observed in previous experiences might be
        observed again in future training experiences.
    """

    def __init__(self, temperature = 200):
        super().__init__()
        self.criterion = BCELoss()
        self.temperature = temperature
        self.new_classes = []
        self.old_classes = []
        self.old_model = None
        self.old_logits = None
        self.lbda = 0

    def before_training_exp(self, strategy:SupervisedTemplate, *args, **kwargs):
      self.new_classes = list(strategy.experience.classes_in_this_experience)
      self.lbda = len(self.old_classes) / len(self.old_classes + self.new_classes)
      return super().before_training_exp(strategy, *args, **kwargs)

    def before_forward(self, strategy, **kwargs):
        if self.old_model is not None:
            with torch.no_grad():
                self.old_logits = self.old_model(strategy.mb_x)

    def __call__(self, logits, targets):
        predictions = torch.sigmoid(logits)

        one_hot = torch.zeros(
            targets.shape[0],
            logits.shape[1],
            dtype=torch.float,
            device=logits.device,
        )
        one_hot[range(len(targets)), targets.long()] = 1

        lossCE = self.criterion(predictions, one_hot)

        #float vector of size targets.shape[0] -> minibatch

        #lossKD needs to be float of size targets.shape[0].

        #For new classes it should be 0

        #For old classes, calculate
        one_hot_KD = torch.zeros(
            targets.shape[0],
            logits.shape[1],
            dtype=torch.float,
            device=logits.device,
        )
        #???????x = torch.as_tensor([[1, 2], [3, 4]])

        if self.old_logits is not None:
          #Não aplicar sigmoide?

          q_hat = torch.exp(self.old_logits / self.temperature) / torch.sum(torch.exp(self.old_logits / self.temperature), 1).unsqueeze(-1)
          q = torch.exp(logits / self.temperature) / torch.sum(torch.exp(logits / self.temperature), 1).unsqueeze(-1)
          #zerar q onde logits pertencem a new_classes
          q[:, self.new_classes] = one_hot_KD[: ,self.new_classes]
          #zerar q onde targets pertencem a new_classes
          torch.where(torch.isnan(q_hat), 1, q_hat)
          torch.where(torch.isnan(q), 1, q)
          lossKD = self.criterion(q, q_hat)
          self.old_logits = None
        else:
          lossKD = torch.zeros(targets.shape[0])

        return (1-self.lbda)*lossCE + torch.mean(lossKD)*self.lbda

    def after_training_exp(self, strategy, **kwargs):
        if self.old_model is None:
            old_model = copy.deepcopy(strategy.model)
            self.old_model = old_model.to(strategy.device)

        self.old_model.load_state_dict(strategy.model.state_dict())

        self.old_classes += np.unique(strategy.experience.dataset.targets).tolist()
