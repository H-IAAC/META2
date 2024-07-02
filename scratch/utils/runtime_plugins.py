import numpy as np
import torch
from avalanche.training.templates import SupervisedTemplate
from avalanche.core import SupervisedPlugin
from sklearn.metrics import classification_report, confusion_matrix
import operator
import warnings
from copy import deepcopy
from avalanche.training.plugins import SupervisedPlugin

def map_to_arange(tensor, value_to_arange):
    
    # Create the output tensor by mapping original values to their corresponding arange values
    mapped_tensor = tensor.clone()
    for value, arange_value in value_to_arange.items():
        mapped_tensor[tensor == value] = arange_value
    
    return mapped_tensor


class ClassPrecisionPlugin(SupervisedPlugin):

    def __init__(self, num_classes):
        super().__init__()

        self.confusion_matrices = []
        self.classifications = []
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.predictions = torch.empty((0), dtype=torch.int64).to(self.device)
        self.true = torch.empty((0), dtype=torch.int64).to(self.device)

        self.classes_yet = []
        self.num_classes = num_classes
        self.mistakes = np.zeros((num_classes))
        self.total = np.zeros((num_classes))
        self.history = []

    def before_training_exp(self, strategy: "SupervisedTemplate", *args, **kwargs):
        self.classes_yet = self.classes_yet + list(strategy.experience.classes_in_this_experience)
        return super().before_backward(strategy, *args, **kwargs)

    def after_eval_forward(self, strategy: "SupervisedTemplate", **kwargs):
        super().after_eval_forward(strategy, **kwargs)
        
        with torch.no_grad():
            # Adds mistakes to array

            self.predictions = torch.cat(
                (self.predictions, torch.argmax(strategy.mb_output, dim=1)))
            self.true = torch.cat((self.true, strategy.mbatch[1]))

    def after_eval(self, strategy: "SupervisedTemplate", **kwargs):
        

        used_true = torch.empty((0), dtype=torch.int64).to(self.device)
        used_pred = torch.empty((0), dtype=torch.int64).to(self.device)
        value_to_arange = dict()
        for idx, i in enumerate(self.classes_yet):
            mask = self.true == i
            used_true = torch.cat((used_true, self.true[mask]))
            used_pred = torch.cat((used_pred, self.predictions[mask]))
            value_to_arange[i] = idx
            
        
        
        self.classifications.append(classification_report(
            map_to_arange(used_true.to('cpu'), value_to_arange), map_to_arange(used_pred.to('cpu'), value_to_arange), output_dict=True))
        self.confusion_matrices.append(
            confusion_matrix(self.true.to('cpu'), self.predictions.to('cpu')))
        self.predictions = torch.empty((0), dtype=torch.int64).to(self.device)
        self.true = torch.empty((0), dtype=torch.int64).to(self.device)

class TrainEarlyStoppingPlugin(SupervisedPlugin):
    """Early stopping and model checkpoint plugin.

    The plugin checks a metric and stops the training loop when the accuracy
    on the metric stopped progressing for `patience` epochs.
    After training, the best model's checkpoint is loaded.

    .. warning::
        The plugin checks the metric value, which is updated by the strategy
        during the evaluation. This means that you must ensure that the
        evaluation is called frequently enough during the training loop.

        For example, if you set `patience=1`, you must also set `eval_every=1`
        in the `BaseTemplate`, otherwise the metric won't be updated after
        every epoch/iteration. Similarly, `peval_mode` must have the same
        value.

    """


    def __init__(
        self,
        patience: int,
        margin: float = .0,
    ):
        """Init.

        :param patience: Number of epochs to wait before stopping the training.
        
        :param margin: a minimal margin of improvements required to be 
            considered best than a previous one. It should be an float, the 
            default value is 0. That means that any improvement is considered 
            better.
        """
        super().__init__()
        
        self.patience = patience
        self.margin = margin

    def before_training(self, strategy, **kwargs):
        self.best_state = None
        self.best_step = 0
        self.best_loss = np.inf      

    def _update_best(self, strategy):
        res = strategy.evaluator.get_last_metrics()
        names = [k for k in res.keys() if k.startswith('Loss_Epoch/train_phase')]
        if len(names) == 0:
            return None
        epoch_loss = res.get(names[-1])
        if epoch_loss + self.margin < self.best_loss:
            self.best_loss = strategy.loss
            self.best_step = self._get_strategy_counter(strategy)
            self.best_state = deepcopy(strategy.model.state_dict())
        
    def after_training_epoch(self, strategy, **kwargs):
        
        self._update_best(strategy)
        curr_step = self._get_strategy_counter(strategy)
        if curr_step - self.best_step >= self.patience:
            strategy.model.load_state_dict(self.best_state)
            print("Early stopping triggered!!!")
            strategy.stop_training()

    def _get_strategy_counter(self, strategy):
        
        return strategy.clock.train_exp_epochs
        