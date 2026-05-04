import torch.nn as nn

class BirdSetClassifier(nn.Module):
    def __init__(self, num_classes, input_dim=256): # Example: HSN has 66 species
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim) # Normalizes the SAM latents
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.bn(x)
        return self.classifier(x)
