#!/usr/bin/env python
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3

import numpy as np
from scipy import linalg


class FIDCalculator:

    def __init__(self, val_npz=None, batch_size=50, device="cuda"):
        # Create Inception V3 network.
        self.inception = inception_v3(pretrained=True, aux_logits=False)
        self.inception.dropout = nn.Identity()
        self.inception.fc = nn.Identity()
        self.inception.eval()
        for param in self.inception.parameters():
            param.requires_grad = False

        if val_npz is not None:
            with np.load(val_npz) as f:
                self.ref_mu, self.ref_sigma = f['mu'], f['sigma']
        
        self.batch_size = batch_size
        self.device = device
    
    def calculate_activations(self, tensor_images):
        num_images = tensor_images.size(0)
        batch_size = self.batch_size
        if batch_size > num_images:
            print(('Warning: batch size is bigger than the data size. '
                'Setting batch size to data size'))
            batch_size = num_images

        pred_arr = np.empty((tensor_images.size(0), 2048))

        start_idx = 0
        self.inception.to(self.device)
        for _ in range(ceil(tensor_images.size(0)/batch_size)):
            end_idx = min(start_idx+batch_size, num_images)

            batch = tensor_images[start_idx:end_idx].to(self.device)
            with torch.no_grad():
                batch = batch.clone()
                batch = F.interpolate(batch, size=(299, 299),
                                      mode='bilinear', align_corners=True)
                pred = self.inception(batch)

            pred = pred.cpu().numpy()
            pred_arr[start_idx:end_idx] = pred

            start_idx = end_idx
        self.inception.to(torch.device("cpu"))

        mu = np.mean(pred, axis=0)
        sigma = np.cov(pred, rowvar=False)
        return mu, sigma

    def calculate_fid(self, tensor_images, eps=1e-6):
        target_mu, target_sigma = self.calculate_activations(tensor_images)

        mu1 = np.atleast_1d(self.ref_mu)
        mu2 = np.atleast_1d(target_mu)

        sigma1 = np.atleast_2d(self.ref_sigma)
        sigma2 = np.atleast_2d(target_sigma)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
        