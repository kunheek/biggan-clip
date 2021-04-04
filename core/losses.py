import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def contrastive_loss(query, key, t=0.0):
    # joint multimodal embedding
    q_norm = F.normalize(query, dim=1)
    k_norm = F.normalize(key, dim=1)

    # scaled pairwise cosine similarity
    logits = torch.mm(q_norm, k_norm.t()) * np.exp(t)

    # loss function
    labels = torch.arange(0, logits.size(0), device=logits.get_device())
    loss_q = F.cross_entropy(logits, labels, reduction='mean')
    loss_k = F.cross_entropy(logits.t(), labels, reduction='mean')
    loss = (loss_q+loss_k) * 0.5
    return loss
