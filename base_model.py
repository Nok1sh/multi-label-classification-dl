import torchvision
import torch
import torchvision.models as models

from torch import nn
from torchinfo import summary


class ResNetModel():
    def __init__(self, num_cls=16) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = models.resnet18(weights="DEFAULT").to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.fc = nn.Linear(512, num_cls)
    
    @property
    def info(self):
        return summary(self.model, input_size=(1, 3, 224, 224))
    
    @property
    def get_model(self):
        return self.model
    
    @property
    def check_requires_grad(self):
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                print(name)
                for i, param in enumerate(layer.parameters()):
                    if i == 0:
                        print(f"weights.requires_grad = {param.requires_grad}")
                    else:
                        print(f"bias.requires_grad = {param.requires_grad}", end="\n")
    
    @property
    def unfreeze_last_layer(self):
        for param in self.model.layer4.parameters():
            param.requires_grad = True

