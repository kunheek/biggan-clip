import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.datasets import CocoCaptions

import numpy as np
import clip


class Tokenize:
    """CLIP tokenizer."""
    def __init__(self, context_length, num_captions=5):
        self.context_length = context_length
        self.num_captions = num_captions
    
    def __call__(self, captions):
        caption = str(captions[np.random.randint(0, self.num_captions)])
        return clip.tokenize([caption], context_length=self.context_length)


class Coco14(CocoCaptions):
    def __init__(self, root_dir, split, transform=None, target_transform=None):
        root = os.path.join(root_dir, 'coco14', f"{split}2014")
        annFile = os.path.join(root_dir, f'coco14/annotations/captions_{split}2014.json')

        if target_transform is None:
            target_transform = Tokenize(context_length=77, num_captions=5)
        super().__init__(root, annFile, transform, target_transform)
        
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, target


def coco14_collate(batch):
    images, tokens = zip(*batch)

    images = torch.stack(images)
    tokens = torch.cat(tokens, dim=0)
    return images, tokens
