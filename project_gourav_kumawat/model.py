import torch.nn as nn
import torchvision.models as models

from config import num_classes, hidden_units, dropout_rate, device


class SkinLesionClassifier(nn.Module):
    def __init__(self):
        super(SkinLesionClassifier, self).__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for param in backbone.parameters():
            param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, num_classes)
        )

        self.network = backbone

    def forward(self, x):
        return self.network(x)


def get_model():
    model = SkinLesionClassifier().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters   : {total - trainable:,}")
    return model
