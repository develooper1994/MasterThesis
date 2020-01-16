# TODO: It is an abstaction layer for WaveGAN-generator
import torch
from torch import nn as nn
from torch.nn import functional as F

from models.layers.BaseLayers import Transpose1dLayer


class WaveGANGenerator(nn.Module):
    """
    WaveGAN Generator Network
    """

    def __init__(self, model_size=64, ngpus=1, num_out_channels=1,
                 latent_dim=100, post_proc_filt_len=512,
                 verbose=False, upsample=True):
        """
        Generator network initialized
        :param model_size: Size of model for flat input

            default=64
        :param ngpus: number of gpu

            default=1
        :param num_out_channels: number of output channels

            default=1
        :param latent_dim: First(linear) layer input dimension

            default=100
        :param post_proc_filt_len: post processing to lenght of filter

            default=512
        :param verbose: verbose output.
            prints size of each layer output for debugging.

            default=False
        :param upsample: upsample or same size? applying deconvolution
            If True: size getting bigger
            else: size remain same at output

            default=True
        """
        super(WaveGANGenerator, self).__init__()
        self.ngpus = ngpus
        self.model_size = model_size  # d
        self.num_channels = num_out_channels  # c
        self.latent_di = latent_dim
        self.post_proc_filt_len = post_proc_filt_len
        self.verbose = verbose
        # "Dense" is the same meaning as fully connection.
        self.fc1 = nn.Linear(latent_dim, 256 * model_size)

        stride = 4
        if upsample:
            stride = 1
            upsample = 4
        self.deconv_1 = Transpose1dLayer(16 * model_size, 8 * model_size, 25, stride, upsample=upsample)
        self.deconv_2 = Transpose1dLayer(8 * model_size, 4 * model_size, 25, stride, upsample=upsample)
        self.deconv_3 = Transpose1dLayer(4 * model_size, 2 * model_size, 25, stride, upsample=upsample)
        self.deconv_4 = Transpose1dLayer(2 * model_size, model_size, 25, stride, upsample=upsample)
        self.deconv_5 = Transpose1dLayer(model_size, num_out_channels, 25, stride, upsample=upsample)

        if post_proc_filt_len:
            self.ppfilter1 = nn.Conv1d(num_out_channels, num_out_channels, post_proc_filt_len)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        """
        Forward pass of Generator
        :param x: input of previous layer
        :return: output of Generator Network
        """
        # TODO: Try to convert linear to conv layer
        # TODO: Experiment with BatchNorm1d Layer
        # Try to DCGAN first and than Parallel WaveGAN, Progressive Growing and EfficientNet approaches.
        x = self.fc1(x).view(-1, 16 * self.model_size, 16)  # format for deconv layers.
        x = F.relu(x)
        # TODO try layer visualization
        self.print_shape(x)

        x = F.relu(self.deconv_1(x))
        self.print_shape(x)

        x = F.relu(self.deconv_2(x))
        self.print_shape(x)

        x = F.relu(self.deconv_3(x))
        self.print_shape(x)

        x = F.relu(self.deconv_4(x))
        self.print_shape(x)

        return torch.tanh(self.deconv_5(x))

    def print_shape(self, x):
        if self.verbose:
            print(x.shape)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))
