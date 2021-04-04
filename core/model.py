from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d

from core.nn import SelfAttention2d
from core.nn import ConditionalBatchNorm2d as CBN
from core.nn.utils import spectral_norm


def gen_arch(image_size, channel, attention=[64]):
    assert image_size in [64, 128, 256]

    if image_size == 256:
        in_channels = [16, 16, 8, 8, 4, 2]
        out_channels = [16, 8, 8, 4, 2, 1]
        resolution = [8, 16, 32, 64, 128, 256]
        depth = 6
    elif image_size == 128:
        in_channels = [16, 16, 8, 4, 2]
        out_channels = [16, 8, 4, 2, 1]
        resolution = [8, 16, 32, 64, 128]
        depth = 5
    elif image_size == 64:
        in_channels = [16, 16, 8, 4]
        out_channels = [16, 8, 4, 2]
        resolution = [8, 16, 32, 64]
        depth = 4

    return {
        'in_channels': [i * channel for i in in_channels],
        'out_channels': [i * channel for i in out_channels],
        'upsample': [True] * depth,
        'resolution': resolution,
        'attention': [res in attention for res in resolution],
        'depth': depth
    }


class Generator(nn.Module):

    def __init__(self,
                 image_size,
                 channels,
                 noise_dim,
                 embedding_dim=512,
                 activation='relu',
                 batchnorm='CBN',
                 concat_z_to_emb=False,
                 self_attention=[],
                 spectral_norm_=False,
                 **kwargs):
        super(Generator, self).__init__()
        arch = gen_arch(image_size, channels, attention=self_attention)
        init_size = arch['in_channels'][0] * (4**2)
        self.concat_z_to_emb = concat_z_to_emb
        condition_dim = noise_dim+embedding_dim if concat_z_to_emb else embedding_dim

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise NotImplementedError

        self.linear = nn.Linear(noise_dim, init_size)
        self.blocks = nn.ModuleList(
            [GBlock(arch['in_channels'][i],
                    arch['out_channels'][i],
                    condition_dim,
                    arch['upsample'][i],
                    activation=self.activation,
                    batchnorm=batchnorm,
                    self_attention=arch['attention'][i],
                    spectral_norm_=spectral_norm_) for i in range(arch['depth'])]
        )
        self.conv = nn.Conv2d(arch['out_channels'][-1], 3, 3, 1, 1)

        if spectral_norm_:
            self.linear = spectral_norm(self.linear)
            self.conv = spectral_norm(self.conv)

    def forward(self, z, embedding):
        h = self.linear(z)
        h = h.view(z.size(0), -1, 4, 4)

        if self.concat_z_to_emb:
            condition = torch.cat([z, embedding], dim=1)
        else:
            condition = embedding
        for block in self.blocks:
            h = block(h, condition)

        h = self.activation(h)
        h = self.conv(h)
        out = torch.tanh(h)
        return out


class GBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 condition_dim,
                 upsample=True,
                 activation=nn.ReLU(inplace=True),
                 batchnorm='cbn',
                 self_attention=False,
                 spectral_norm_=False,
                 **kwargs):
        super(GBlock, self).__init__()
        self.upsample = upsample
        self.activation = activation
        self.self_attn = self_attention

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)

        # Batchnorm layers
        batchnorm = batchnorm.lower()
        if batchnorm == 'bn':
            bn = BatchNorm2d
        elif batchnorm in ['cbn', 'scbn']:
            bn = partial(CBN, condition_dim=condition_dim, spectral_norm_=spectral_norm_)
        else:
            raise NotImplementedError     
        self.bn1 = bn(in_channels)
        self.bn2 = bn(out_channels)

        if self.self_attn:
            self.attn_layer = SelfAttention2d(out_channels, out_channels, return_attn=False)

        if spectral_norm_:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            if self.learnable_sc:
                self.conv_sc = spectral_norm(self.conv_sc)
            if self_attention:
                self.attn_layer.query = spectral_norm(self.attn_layer.query)
                self.attn_layer.key = spectral_norm(self.attn_layer.key)
                self.attn_layer.value = spectral_norm(self.attn_layer.value)

    def forward(self, x, y=None):
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = F.interpolate(h, scale_factor=2)
            x = F.interpolate(x, scale_factor=2)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        h = h + x
        if self.self_attn:
            h = self.attn_layer(h)
        return h



def disc_arch(image_size, channel, attention):
    assert image_size in [64, 128, 256]

    if image_size == 256:
        in_channels = [1, 2, 4, 8, 8, 16]
        out_channels = [1, 2, 4, 8, 8, 16, 16]
        resolution = [128, 64, 32, 16, 8, 4, 4]
        depth = 7
    elif image_size == 128:
        in_channels = [1, 2, 4, 8, 16]
        out_channels = [1, 2, 4, 8, 16, 16]
        resolution = [64, 32, 16, 8, 4, 4]
        depth = 6
    elif image_size == 64:
        in_channels = [1, 2, 4, 8]
        out_channels = [1, 2, 4, 8, 16]
        resolution = [32, 16, 8, 4, 4]
        depth = 5

    return {
        'in_channels': [3] + [item * channel for item in in_channels],
        'out_channels': [item * channel for item in out_channels],
        'downsample': [True] * (depth-1) + [False],
        'resolution': resolution,
        'attention': [res in attention for res in resolution],
        'depth': depth
    }


class Discriminator(nn.Module):
    def __init__(self,
                 image_size,
                 channels,
                 embedding_dim,
                 hypersphere_dim=512,
                 activation='relu',
                 self_attention=[],
                 spectral_norm_=False,
                 **kwargs):
        super(Discriminator, self).__init__()
        arch = disc_arch(image_size, channels, self_attention)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise NotImplementedError

        self.blocks = nn.ModuleList(
            [DBlock(arch['in_channels'][i],
                    arch['out_channels'][i],
                    arch['downsample'][i],
                    self.activation,
                    arch['attention'][i],
                    spectral_norm_=spectral_norm_) for i in range(arch['depth'])]
        )
        self.linear = nn.Linear(arch['out_channels'][-1], 1)
        self.nonlinear_i = nn.Linear(arch['out_channels'][-1], hypersphere_dim)
        self.nonlinear_t = nn.Linear(embedding_dim, hypersphere_dim)

        if spectral_norm_:
            self.linear = spectral_norm(self.linear)
            self.nonlinear_i = spectral_norm(self.nonlinear_i)
            self.nonlinear_t = spectral_norm(self.nonlinear_t)

    def forward(self, x, embedding=None):
        h = x
        for block in self.blocks:
            h = block(h)

        h = self.activation(h)
        h = torch.sum(h, [2, 3])  # global sum-pooling
        logit = self.linear(h)
        if embedding is not None:
            return logit, self.nonlinear_i(h), self.nonlinear_t(embedding)
        else:
            return logit


class DBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample=True,
                 activation=nn.ReLU(inplace=True),
                 self_attention=False,
                 spectral_norm_=False):
        super(DBlock, self).__init__()
        self.downsample = downsample
        self.activation = activation
        self.self_attn = self_attention

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.learnable_shortcut = True if (in_channels != out_channels) or downsample else False
        if self.learnable_shortcut:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        if self.self_attn:
            self.attn_layer = SelfAttention2d(out_channels, out_channels)
        
        if spectral_norm_:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            if self.learnable_shortcut:
                self.conv_sc = spectral_norm(self.conv_sc)
            if self.self_attn:
                self.attn_layer.query = spectral_norm(self.attn_layer.query)
                self.attn_layer.key = spectral_norm(self.attn_layer.key)
                self.attn_layer.value = spectral_norm(self.attn_layer.value)

    def shortcut(self, x):
        if self.learnable_shortcut:
            x = self.conv_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, kernel_size=2)
        return x

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = F.avg_pool2d(h, kernel_size=2)
        h += self.shortcut(x)
        if self.self_attn:
            h = self.attn_layer(h)
            h = self.activation(h)
        return h
