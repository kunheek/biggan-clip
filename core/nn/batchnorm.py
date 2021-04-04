import torch.nn as nn
from core.nn.utils import spectral_norm


class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, condition_dim, spectral_norm_=False):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, eps=1e-4, momentum=0.1, affine=False)
        
        self.gain = nn.Linear(condition_dim, num_features)
        self.bias = nn.Linear(condition_dim, num_features)
        if spectral_norm_:
            self.gain = spectral_norm(self.gain, eps=1e-6)
            self.bias = spectral_norm(self.bias, eps=1e-6)

    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(-1, self.num_features, 1, 1)
        bias = self.bias(y).view(-1, self.num_features, 1, 1)
        out = self.bn(x)
        return out * gain + bias
