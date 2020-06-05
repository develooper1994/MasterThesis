# TODO: It is an abstaction layer for WaveGAN-generator

import torch
from torch import nn as nn
from torch.nn import functional as F

from not_functional.config import noise_latent_dim
from not_functional.models.architectures.layers.BaseLayers import Transpose1dLayer


class WaveGANGenerator(nn.Module):
    """
    WaveGAN Generator Network
    """

    def __init__(self, model_size: int = 64, num_channels: int = 1,
                     verbose: bool = False, upsample=True, slice_len: int = 16384,
                     use_batch_norm: bool = False) -> None:
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
        assert slice_len in [16384, 32768, 65536]  # used to predict longer utterances

        self.model_size = model_size  # d
        self.verbose = verbose
        self.use_batch_norm = use_batch_norm

        self.dim_mul = 16 if slice_len == 16384 else 32

        self.fc1 = nn.Linear(noise_latent_dim, 4 * 4 * model_size * self.dim_mul)
        self.bn1 = nn.BatchNorm1d(num_features=model_size * self.dim_mul)

        stride: int = 4
        if upsample:
            stride = 1
            upsample = 4

        deconv_layers = [
            Transpose1dLayer(self.dim_mul * model_size, (self.dim_mul * model_size) // 2, 25, stride, upsample=upsample,
                             use_batch_norm=use_batch_norm),
            Transpose1dLayer((self.dim_mul * model_size) // 2, (self.dim_mul * model_size) // 4, 25, stride,
                             upsample=upsample, use_batch_norm=use_batch_norm),
            Transpose1dLayer((self.dim_mul * model_size) // 4, (self.dim_mul * model_size) // 8, 25, stride,
                             upsample=upsample, use_batch_norm=use_batch_norm),
            Transpose1dLayer((self.dim_mul * model_size) // 8, (self.dim_mul * model_size) // 16, 25, stride,
                             upsample=upsample, use_batch_norm=use_batch_norm),
        ]

        if slice_len == 16384:
            deconv_layers.append(
                Transpose1dLayer((self.dim_mul * model_size) // 16, num_channels, 25, stride, upsample=upsample))
        elif slice_len == 32768:
            deconv_layers += [
                Transpose1dLayer((self.dim_mul * model_size) // 16, model_size, 25, stride, upsample=upsample,
                                 use_batch_norm=use_batch_norm)
                , Transpose1dLayer(model_size, num_channels, 25, 2, upsample=upsample)
            ]
        elif slice_len == 65536:
            deconv_layers += [
                Transpose1dLayer((self.dim_mul * model_size) // 16, model_size, 25, stride, upsample=upsample,
                                 use_batch_norm=use_batch_norm)
                , Transpose1dLayer(model_size, num_channels, 25, stride, upsample=upsample)
            ]
        else:
            raise ValueError('slice_len {} value is not supported'.format(slice_len))

        self.deconv_list = nn.ModuleList(deconv_layers)
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
        x = self.fc1(x).view(-1, self.dim_mul * self.model_size, 16)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        self.print_shape(x)

        for deconv in self.deconv_list[:-1]:
            x = F.relu(deconv(x))
            self.print_shape(x)
        return torch.tanh(self.deconv_list[-1](x))

    def print_shape(self, x):
        if self.verbose:
            print(x.shape)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))

    def __repr__(self):
        return 'WaveGANGenerator'
