# inspired from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
# standart library modules

# 3rd party modules

# torch modules
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# my modules
from models.BaseGAN import BaseGAN
from models.losses.BaseLoss import wassertein_loss
from utils import BasicUtils as utls_basic
from utils.BasicUtils import Parameters, device, require_net_update, numpy_to_var, calc_gradient_penalty, \
    prevent_net_update, cuda
from utils.logger import logger
from models.utils.visualization.visualization import plot_loss


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

    def _critic_train_iteration(self, data):
        """ """
        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        data = Variable(data)
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.data[0])

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.D_opt.step()

        # Record loss
        self.losses['D'].append(d_loss.data[0])