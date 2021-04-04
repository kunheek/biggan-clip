import os
import json
import logging
from datetime import datetime

import torch
from torchvision.utils import save_image

logger = logging.getLogger(__name__)


class TrainHelper:

    def __init__(self, args):
        if args.exp_name == None:
            now = datetime.now().strftime("%m%d_%H:%M:%S")
            args.exp_name = f"{args.dataset}_{now}"

        self.max_iter = args.max_iter
        self.expr_dir = os.path.join("expr", args.exp_name)
        self.sample_dir = os.path.join(self.expr_dir, "samples")
        self.ckpt_dir = os.path.join(self.expr_dir, "checkpoints")
        self._make_dirs()

        with open(os.path.join(self.expr_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        logging.basicConfig(filename=os.path.join(self.expr_dir, "train.log"),
                            filemode='a',
                            format='[%(asctime)s][%(levelname)s]%(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

        # Checkpoints
        self.states = []
        for i in range(args.save_top):
            state = {'name': str(i), 'FID': 99999.0}
            self.states.append(state)
    
    def _make_dirs(self):
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
    
    def log_params(self, netG, netD, text_encoder):
        def count_model_params(model):
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return n / 1000000
        num_g_params = count_model_params(netG)
        num_d_params = count_model_params(netD)
        num_t_params = count_model_params(text_encoder)
        total_params = num_g_params + num_d_params + num_t_params
        logger.info("#Param:")
        logger.info(f"Generator: {num_g_params:.2f}M")
        logger.info(f"Discriminator: {num_d_params:.2f}M")
        logger.info(f"Text encoder: {num_t_params:.2f}M")
        logger.info(f"Total: {total_params:.2f}M")

    def update_checkpoint(self, netG, netD, fid_value, inception_score=None):
        if fid_value > self.states[-1]['FID']:
            return

        self.states[-1]['FID'] = fid_value
        name = self.states[-1]['name']
        torch.save({
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'FID': fid_value
        }, os.path.join(self.ckpt_dir, f"net_{name}.pth"))
        self.states = sorted(self.states, key=lambda i: i['FID'], reverse=False)

    def save_image(self, tensor_images, step=0, filename=None, *args, **kwargs):
        if filename is None:
            filename = os.path.join(self.sample_dir, f"step_{step:06d}.png")
        else:
            filename = os.path.join(self.sample_dir, filename)
        save_image(tensor_images, filename, *args, **kwargs)
    
    def log(self, step, train=True, **items):
        msg = "[TRAIN]" if train else "[VAL]"
        msg += f"[{step}/{self.max_iter}]"
        for key, value in items.items():
            msg += f" {key}: {value:.3f}"
        logger.info(msg)
            