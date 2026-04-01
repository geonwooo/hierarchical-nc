import torch
import torch.nn as nn

__all__ = [
    'FCNorm',
]


# for LDAM Loss
class FCNorm(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FCNorm, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = self.fc(input)
        return x

