from typing import NoReturn

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from wavegan_pytorch.models.layers.layers import Transpose1dLayer, Conv1D
from wavegan_pytorch.params import *


class WaveGANGenerator(nn.Module):
    def __init__(self, model_size: int = 64, num_channels: int = 1,
                 verbose: bool = False, upsample = True, slice_len: int = 16384, use_batch_norm: bool = False) -> NoReturn:
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
        x = self.fc1(x).view(-1, self.dim_mul * self.model_size, 16)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        if self.verbose:
            print(x.shape)

        for deconv in self.deconv_list[:-1]:
            x = F.relu(deconv(x))
            if self.verbose:
                print(x.shape)
        return torch.tanh(self.deconv_list[-1](x))


class WaveGANDiscriminator(nn.Module):
    def __init__(self, model_size: int =64, ngpus: int = 1, num_channels: int = 1, shift_factor: int = 2,
                 alpha: float = 0.2, verbose: bool = False, slice_len: int = 16384, use_batch_norm: bool = False):
        super(WaveGANDiscriminator, self).__init__()
        assert slice_len in [16384, 32768, 65536]  # used to predict longer utterances

        self.model_size = model_size  # d
        self.ngpus = ngpus
        self.use_batch_norm = use_batch_norm
        self.num_channels = num_channels  # c
        self.shift_factor = shift_factor  # n
        self.alpha = alpha
        self.verbose = verbose

        conv_layers = [
            Conv1D(num_channels, model_size, 25, stride=4, padding=11, use_batch_norm=use_batch_norm, alpha=alpha,
                   shift_factor=shift_factor),
            Conv1D(model_size, 2 * model_size, 25, stride=4, padding=11, use_batch_norm=use_batch_norm, alpha=alpha,
                   shift_factor=shift_factor),
            Conv1D(2 * model_size, 4 * model_size, 25, stride=4, padding=11, use_batch_norm=use_batch_norm, alpha=alpha,
                   shift_factor=shift_factor),
            Conv1D(4 * model_size, 8 * model_size, 25, stride=4, padding=11, use_batch_norm=use_batch_norm, alpha=alpha,
                   shift_factor=shift_factor),
            Conv1D(8 * model_size, 16 * model_size, 25, stride=4, padding=11, use_batch_norm=use_batch_norm,
                   alpha=alpha, shift_factor=0 if slice_len == 16384 else shift_factor)
        ]
        self.fc_input_size = 256 * model_size
        if slice_len == 32768:
            conv_layers.append(
                Conv1D(16 * model_size, 32 * model_size, 25, stride=2, padding=11, use_batch_norm=use_batch_norm,
                       alpha=alpha, shift_factor=0)
            )
            self.fc_input_size = 480 * model_size
        elif slice_len == 65536:
            conv_layers.append(
                Conv1D(16 * model_size, 32 * model_size, 25, stride=4, padding=11, use_batch_norm=use_batch_norm,
                       alpha=alpha, shift_factor=0)
            )
            self.fc_input_size = 512 * model_size

        self.conv_layers = nn.ModuleList(conv_layers)

        self.fc1 = nn.Linear(self.fc_input_size, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            if self.verbose:
                print(x.shape)
        x = x.view(-1, self.fc_input_size)
        if self.verbose:
            print(x.shape)

        return self.fc1(x)


if __name__ == '__main__':

    for slice_len in [16384, 32768, 65536]:
        G = WaveGANGenerator(verbose=True, upsample=True, use_batch_norm=True, slice_len=slice_len)
        out = G(torch.randn(10, noise_latent_dim))
        print(out.shape)
        assert (out.shape == (10, 1, slice_len))
        print('==========================')

        D = WaveGANDiscriminator(verbose=True, use_batch_norm=True, slice_len=slice_len)
        out2 = D(torch.randn(10, 1, slice_len))
        print(out2.shape)
        assert (out2.shape == (10, 1))
        print('==========================')
