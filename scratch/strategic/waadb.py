from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from collections import defaultdict
import torch

import warnings

class WAADBPlugin(SupervisedPlugin):

    def __init__(self):
        super().__init__()
        self.n_samples_max = 0

    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
      # Counts the number of instances of each class in the current experiment.
      self.count = defaultdict(int)

      for batch in strategy.dataloader:
        targets = batch[1]
        for c in targets:
          self.count[c.item()] += 1
      print(self.count)

      self.n_samples_max = max(self.n_samples_max, max(self.count.values()))

    def after_update(self, strategy: "SupervisedTemplate", **kwargs):

      # Rescales the weight vectors of previously seen classes.
      prev_classes = strategy.experience.previous_classes
      wvector = list(strategy.model.named_children())[-1][-1][-1].weight
      with torch.no_grad():
        for c in prev_classes:
          # TODO: Check if count[c] exists
          if self.count[c] == 0:
            # raise Exception(f"There are no instances of class {c} in memory on experiece {strategy.experience.current_experience}.")
            warnings.warn(f"There are no instances of class {c} in memory on experiece {strategy.experience.current_experience}.")
          else:
            wvector[c] = wvector[c] * ((self.count[c] / self.n_samples_max))