import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):

    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.glu(input)