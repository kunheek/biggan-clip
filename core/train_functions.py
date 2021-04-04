import torch
import torch.nn as nn
import torch.nn.functional as F

from core.losses import contrastive_loss


def trainD(netG, netD, images, embedding, noise, args):
    logit_real, image_vector, text_vector = netD(images, embedding)
    errD_real = F.relu(1.0 - logit_real, inplace=True).mean()
    loss_contra = contrastive_loss(image_vector, text_vector)

    fake_images = netG(noise, embedding)
    logit_fake = netD(fake_images.detach())
    errD_fake = F.relu(1.0 + logit_fake, inplace=True).mean()

    loss_adv = errD_real + errD_fake
    errD = loss_adv + args.lambdaC*loss_contra

    log = {
        'L_adv': loss_adv.item(),
        'L_contra': loss_contra.item()
    }
    return errD, log


def trainG(netG, netD, embedding, noise, args):
    fake_image = netG(noise, embedding)

    out, image_vector, text_vector = netD(fake_image, embedding)
    loss_adv = -out.mean()
    loss_contra = contrastive_loss(image_vector, text_vector)
    errG = loss_adv + args.lambdaC*loss_contra

    log = {
        'L_adv': loss_adv.item(),
        'L_contra': loss_contra.item()
    }
    return errG, log
