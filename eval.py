#!/usr/bin/env python
import os
import json
import random
from argparse import ArgumentParser, Namespace
from pprint import pprint

import numpy as np
import clip

import torch
from torchvision.utils import save_image

import core


def parse_args():
    parser = ArgumentParser(description='Evaluate T2IGAN network')
    parser.add_argument('--expr', default=None, type=str)
    parser.add_argument('--candidate', default=None, type=int)
    parser.add_argument('--manual_seed', type=int, help='manual seed')
    parser.add_argument('--cuda', default=True, type=bool)

    # data loader
    parser.add_argument('--root_dir', default='./datasets', type=str)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--num_workers', dest='workers', default=4, type=int)
    return parser.parse_args()


@torch.no_grad()
def sample_images(netG, dataloader, args):
    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    text_encoder, _ = clip.load("ViT-B/32", device)

    # Build and load the generator
    state_dict = torch.load('%s/checkpoints/net_%d.pth' % (args.expr, args.candidate))
    print(state_dict.keys())
    print(f"FID value: {state_dict['FID']:.2f}")
    netG.load_state_dict(state_dict['netG_state_dict'])
    netG.eval()
    netG.to(device)

    batch_size = args.batch_size
    cnt = 0
    os.makedirs(os.path.join(args.expr, "real"), exist_ok=True)
    os.makedirs(os.path.join(args.expr, "fake"), exist_ok=True)
    while cnt < 30000:
        for _, (image, text) in enumerate(dataloader, 0):
            image = image.cuda()
            text = text.cuda()
            
            embedding = text_encoder.encode_text(text)

            noise = torch.randn(batch_size, args.nz, device=device)
            noise = torch.clamp(noise, min=-1.0, max=1.0)
            fake_image = netG(noise, embedding)
            fake_image = (fake_image+1.0) * 0.5
            for j in range(batch_size):
                save_image(image[j], '%s/real/%05d.png' % (args.expr, cnt))
                save_image(fake_image[j], '%s/fake/%05d.png' % (args.expr, cnt))
                
                cnt += 1
                if cnt >= 30000:
                    break
                if cnt % 1000 == 0:
                    print(f"{cnt}/30000")
            if cnt >= 30000:
                    break


if __name__ == "__main__":
    args = parse_args()

    if args.manual_seed is None:
        args.manual_seed = 100
    print("random seed:", args.manual_seed)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.backends.cudnn.benchmark = True
    
    with open(os.path.join(args.expr, "args.json"), "r") as f:
        json_args = json.load(f)
    input_args = vars(args)
    # Override json_args with input_args
    for key in input_args.keys():
        json_args[key] = input_args[key]
    pprint(json_args)
    args = Namespace(**json_args)
    
    netG, netD = core.build_networks(args)
    dataloader = core.build_dataloader(args, split='val')

    sample_images(netG, dataloader, args)  # generate images for the whole valid dataset
