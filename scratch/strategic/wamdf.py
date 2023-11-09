import torch
from avalanche.training.templates import SupervisedTemplate
from avalanche.core import SupervisedPlugin

class WAMDFPlugin(SupervisedPlugin):

    def __init__(self):
        super().__init__()
        self.old_classes = []
        self.current_classes = []

    def update_classes(self, new_classes):
      self.old_classes = self.old_classes + self.current_classes
      self.current_classes = new_classes

    def before_forward(self, strategy: "SupervisedTemplate", **kwargs):
        # quero mandar classes da experiencia
        self.update_classes(strategy.experience.classes_in_this_experience)

        with torch.no_grad():
          #Set bias to zero
          strategy.model.state_dict()[list(strategy.model.state_dict().keys())[-1]].data.copy_(torch.zeros_like(strategy.model.state_dict()[list(strategy.model.state_dict().keys())[-1]].data))

    def after_update(self, strategy: "SupervisedTemplate", **kwargs):

      if(len(self.old_classes) > 0)  :
        norm_old = 0
        norm_new = 0

        with torch.no_grad():
          for i in self.old_classes:
            norm_old += torch.linalg.vector_norm(strategy.model.state_dict()[list(strategy.model.state_dict().keys())[-2]].data[i, :])/len(self.old_classes)

          for i in self.current_classes:
            norm_new += torch.linalg.vector_norm(strategy.model.state_dict()[list(strategy.model.state_dict().keys())[-2]].data[i, :])/len(self.current_classes)

          for i in self.current_classes:
            strategy.model.state_dict()[list(strategy.model.state_dict().keys())[-2]].data[i, :].copy_(strategy.model.state_dict()[list(strategy.model.state_dict().keys())[-2]][i, :] * (norm_old/norm_new))

          #Set bias to zero
          strategy.model.state_dict()[list(strategy.model.state_dict().keys())[-1]].data.copy_(torch.zeros_like(strategy.model.state_dict()[list(strategy.model.state_dict().keys())[-1]].data))
