# inspired from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
# standart library modules

# 3rd party modules
import numpy as np

# torch modules
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# my modules
from models.BaseGAN import BaseGAN


class Trainer:
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=5, print_every=50,
                 use_cuda=False):
        self.GAN = BaseGAN(discriminator, generator)
        self.G = self.GAN.get_generator()
        self.G_opt = gen_optimizer
        self.D = self.GAN.get_discriminator()
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
