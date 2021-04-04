from functools import partial

import torch.nn as nn


spectral_norm = partial(nn.utils.spectral_norm, eps=1e-6)