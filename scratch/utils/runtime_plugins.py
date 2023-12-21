import numpy as np
import torch
from avalanche.training.templates import SupervisedTemplate
from avalanche.core import SupervisedPlugin
from sklearn.metrics import classification_report, confusion_matrix


class ClassPrecisionPlugin(SupervisedPlugin):

    def __init__(self, num_classes):
        super().__init__()

        self.confusion_matrices = []
        self.classifications = []
        self.predictions = torch.empty((0), dtype=torch.int32)
        self.true = torch.empty((0), dtype=torch.int32)

        self.num_classes = num_classes
        self.mistakes = np.zeros((num_classes))
        self.total = np.zeros((num_classes))
        self.history = []

    def after_eval_forward(self, strategy: "SupervisedTemplate", **kwargs):
        super().after_eval_forward(strategy, **kwargs)

        with torch.no_grad():
            # Adds mistakes to array

            self.predictions = torch.cat(
                (self.predictions, torch.argmax(strategy.mb_output, dim=1)))
            self.true = torch.cat((self.true, strategy.mbatch[1]))

    def after_eval(self, strategy: "SupervisedTemplate", **kwargs):

        # store
        # print(classification_report(self.true, self.predictions, output_dict=True))

        self.classifications.append(classification_report(
            self.true, self.predictions, output_dict=True))
        self.confusion_matrices.append(
            confusion_matrix(self.true, self.predictions))
        self.predictions = torch.empty((0), dtype=torch.int32)
        self.true = torch.empty((0), dtype=torch.int32)
