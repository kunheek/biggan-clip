import random

import torch
from torchvision import transforms

import core.train_functions
from .model import Generator, Discriminator
from .data import Coco14, coco14_collate


def build_networks(args):
    netG = Generator(image_size=args.image_size,
                     channels=args.G_ch,
                     noise_dim=args.nz,
                     embedding_dim=512,
                     activation=args.G_act,
                     spectral_norm_=args.G_SN,
                     batchnorm=args.G_bn,
                     concat_z_to_emb=args.concat_z_to_emb,
                     attention=args.G_attn)
    netD = Discriminator(image_size=args.image_size,
                         channels=args.D_ch,
                         embedding_dim=512,
                         activation=args.D_act,
                         spectral_norm_=args.D_SN,
                         attention=args.D_attn)
    return netG, netD


def build_dataloader(args, split='train', sample_size=None):
    assert split in ['train', 'val']
    if split == 'train':
        transform = transforms.Compose([
                transforms.Resize(int(args.image_size * 76 / 64)),
                transforms.RandomCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(args.image_size * 76 / 64)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
            )

    if args.dataset == 'coco14':
        dataset = Coco14(root_dir=args.root_dir, split=split, transform=transform)
    else:
        raise NotImplementedError

    if split == 'val' and sample_size is not None:
        subset = [random.randint(0, len(dataset)-1) for _ in range(sample_size)]
        dataset = torch.utils.data.Subset(dataset, subset)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=coco14_collate, pin_memory=True, drop_last=True)
    return dataloader
