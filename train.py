#!/usr/bin/env python
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import clip
from tqdm import tqdm

import core
from core.metrics.fid_score import FIDCalculator
from core.utils import TrainHelper


def parse_args():
    parser = ArgumentParser(description='Train the BigGAN-CLIP Networks.')
    parser.add_argument('--manual_seed', type=int, help='manual seed')
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--max_epoch', default=None, type=int)
    parser.add_argument('--gpus', nargs='+', type=int, default=[])
    parser.add_argument('--num_D_steps', type=int, default=1,
                        help='Number of D steps per G step (default: %(default)s)')

    # Bookkeping stuff
    parser.add_argument('--exp_name', default=None, type=str,
                        help='Experiment name (where outputs will be saved).'
                             '(default: %(default)s')
    parser.add_argument('--log_every', type=int, default=100,
                        help="Log every X iterations. (default: %(default)s")
    parser.add_argument('--save_every', type=int, default=2000,
                        help='Save every X iterations. (default: %(default)s)')
    parser.add_argument('--save_top', type=int, default=5,
                        help='Save top X models. (default: %(default)s)')

    # Dataset/Dataloader stuff
    parser.add_argument('--root_dir', default='./datasets', type=str)
    parser.add_argument('--dataset', default='coco14', type=str, choices=['coco14'])
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', dest='workers', default=4, type=int)

    # Model stuff
    parser.add_argument('--noise_dim', dest='nz', default=128, type=int,
                        help='Noise dimensionallity (default: %(default)s)')
    parser.add_argument('--G_ch', default=64, type=int,
                        help='Channel multiplier for Generator. (default: %(default)s)')
    parser.add_argument('--D_ch', default=64, type=int,
                        help='Channel multiplier for Discriminator. (default: %(default)s)')
    parser.add_argument('--G_SN', action='store_true', default=False,
                        help='If True, apply spectral normalization to G. (default: %(default)s)')
    parser.add_argument('--D_SN', action='store_true', default=False,
                        help='If True, apply spectral normalization to D. (default: %(default)s)')
    parser.add_argument('--G_attn', nargs='+', default=[64], type=int,
                        help='Resolution to apply self-attention. (default: %(default)s)')
    parser.add_argument('--D_attn', nargs='+', default=[64], type=int,
                        help='Resolution to apply self-attention. (default: %(default)s)')
    parser.add_argument('--G_act', default='relu', type=str, choices=['relu', 'leaky_relu'],
                        help='Generator activation function. (default: %(default)s)')
    parser.add_argument('--D_act', default='relu', type=str, choices=['relu', 'leaky_relu'],
                        help='Discriminator activation function. (default: %(default)s)')
    parser.add_argument('--G_bn', default='CBN', type=str,
                        choices=['BN', 'CBN'],
                        help='Condition method. (default: %(default)s)')
    parser.add_argument('--concat_z_to_emb', default=False, type=bool,
                        help='If true, concat noise to the condition. (default: %(default)s)')

    # contrastive loss
    parser.add_argument('--lambdaC', default=1.0, type=float,
                        help='Weight for contrastive loss. (default: %(default)s')
    parser.add_argument('--temperature', dest='temp', default=0.0, type=float)
    
    # Optimizer stuff
    parser.add_argument('--G_lr', default=0.0001, type=float)
    parser.add_argument('--D_lr', default=0.0004, type=float)
    parser.add_argument('--G_b1', default=0.5, type=float)
    parser.add_argument('--G_b2', default=0.999, type=float)
    parser.add_argument('--D_b1', default=0.5, type=float)
    parser.add_argument('--D_b2', default=0.999, type=float)
    return parser.parse_args()


def train(args):
    # Create networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG, netD = core.build_networks(args)
    netG.cuda()
    netD.cuda()
    text_encoder, _ = clip.load("ViT-B/32", device)

    # Create dataloaders
    train_loader = core.build_dataloader(args, split='train')
    val_loader = core.build_dataloader(args, split='val', sample_size=5000)

    # Logging stuff.
    helper = TrainHelper(args)
    helper.log_params(netG, netD, text_encoder)

    #
    if len(args.gpus) > 1:
        netG = torch.nn.DataParallel(netG)
        netD = torch.nn.DataParallel(netD)
    
    # Define optimizers.
    optG = torch.optim.Adam(netG.parameters(), lr=args.G_lr, betas=(args.G_b1, args.G_b2))
    optD = torch.optim.Adam(netD.parameters(), lr=args.D_lr, betas=(args.D_b1, args.D_b2))

    # To observe progress.
    data_dir = os.path.join(args.root_dir, args.dataset)
    fid_calculator = FIDCalculator(os.path.join(data_dir, f"{args.dataset}_val.npz"))
    fixed_noise = torch.randn(min(16, args.batch_size), args.nz, device=device)
    ref_imgs, fixed_emb = None, None
 
    # Define train step.
    trainD = core.train_functions.trainD
    trainG = core.train_functions.trainG

    if args.max_epoch is not None:
        args.max_iter = args.max_epoch * len(train_loader)
    epoch = 0
    pbar = tqdm(range(1, args.max_iter+1))
    for step in pbar:
        # Discriminator step.
        for _ in range(args.num_D_steps):
            try:
                image, text = next(d_iter)
            except (StopIteration, UnboundLocalError):
                d_iter = iter(train_loader)
                image, text = next(d_iter)
                epoch += 1
            
            image = image.cuda()
            text = text.cuda()
            noise = torch.randn(args.batch_size, args.nz, device=device)

            with torch.no_grad():
                embedding = text_encoder.encode_text(text)
                embedding = embedding.float()

                if fixed_emb is None:
                    fixed_emb = embedding[:min(16, args.batch_size)]
                    ref_imgs = image[:min(16, args.batch_size)]
                    helper.save_image(ref_imgs, filename="reference_images.png", normalize=True)

            errD, errD_log = trainD(netG, netD, image, embedding, noise, args)
            
            # Update Discriminator.
            # optG.zero_grad(set_to_none=True)
            optD.zero_grad(set_to_none=True)
            errD.backward()
            optD.step()

        # Generator step.
        errG, errG_log = trainG(netG, netD, embedding, noise, args)

        # Update Generator.
        # optD.zero_grad(set_to_none=True)
        optG.zero_grad(set_to_none=True)
        errG.backward()
        optG.step()

        pbar.set_description(f"errD: {errD_log['L_adv']:.2f}, errG: {errG_log['L_adv']:.2f}")
        if step % args.log_every == 0:
            netG.eval()
            with torch.no_grad():
                samples = netG(fixed_noise, fixed_emb)
            netG.train()

            helper.save_image(samples, step=step, normalize=True)
            helper.log(step, train=True,
                       D_L_adv=errD_log['L_adv'], D_L_contra=errD_log['L_contra'],
                       G_L_adv=errG_log['L_adv'], G_L_contra=errG_log['L_contra'])

        if step % args.save_every == 0:
            netG.eval()
            fake_images = []
            for (image, text) in val_loader:
                noise = torch.randn(image.size(0), args.nz, device=device)
                text = text.cuda()
                with torch.no_grad():
                    embedding = text_encoder.encode_text(text)
                    embedding = embedding.float()
                    fake = netG(noise, embedding)
                fake_images.append(fake.cpu())
            netG.train()
            fake_images = torch.cat(fake_images, dim=0)
            
            fid_value = fid_calculator.calculate_fid(fake_images)
            helper.update_checkpoint(netG, netD, fid_value)
            helper.log(step, train=False, FID=fid_value)


def set_seed_all(seed):
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    args = parse_args()
    print('Using config:')
    print(args)

    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    print("random seed:", args.manual_seed)
    set_seed_all(args.manual_seed)
    torch.backends.cudnn.benchmark = True
    
    train(args)
