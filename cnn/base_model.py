import torch
import torchvision.models as models

from torch import nn
from torchinfo import summary
from torchvision.transforms import v2


class ResNetModel(nn.Module):

    TRANSFORM = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToTensor(),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    def __init__(self, num_cls=16) -> None:
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.num_cls = num_cls

        self.model = models.resnet18(weights="DEFAULT").to(device)

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.fc = nn.Linear(512, self.num_cls)
    
    @property
    def info(self):
        return summary(self.model, input_size=(1, 3, 224, 224))
    
    def forward(self, x):
        return self.model(x)
    
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
    
    def save_checkpoint(self, state, epoch):
        torch.save(state, f"checkpoints/checkpoint_{epoch+1}.pth")
    
    @classmethod
    def load_checkpoint(cls, epoch, optimizer=None, scheduler=None):
        """
        checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "num_classes": model.num_cls,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": {}
        }
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(f"checkpoints/checkpoint_{epoch}.pth", map_location=device)

        model = cls(
            num_cls = checkpoint["num_cls"]
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Metrics this checkpoint: {checkpoint["metrics"]}")

        return model

    
    def predict_proba(self, x):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.eval()
        with torch.no_grad():
            x = x.to(device)
            self.to(device)
            pred = self(x)
            return torch.sigmoid(pred)
        
    def predict(self, x, threshold = 0.5):
        pred = self.predict_proba(x)
        return (pred > threshold).float()


