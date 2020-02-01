# TODO: It is an abstaction layer for WaveGAN-discriminator
from torch import nn as nn
from torch.nn import functional as F

from models.custom_transforms.custom_transforms import PhaseShuffle
from models.layers.BaseLayers import Conv1D


class WaveGANDiscriminator(nn.Module):
    def __init__(self, model_size: int =64, ngpus: int = 1, num_channels: int = 1, shift_factor: int = 2,
                 alpha: float = 0.2, verbose: bool = False, slice_len: int = 16384, use_batch_norm: bool = False) -> None:
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
            self.print_shape(x)
        x = x.view(-1, self.fc_input_size)
        self.print_shape(x)

        return self.fc1(x)

    def print_shape(self, x):
        if self.verbose:
            print(x.shape)

    def __repr__(self):
        return 'WaveGANDiscriminator'
