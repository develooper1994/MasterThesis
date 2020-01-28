from typing import NoReturn

from torch import nn as nn
from torch.nn import functional as F

from wavegan_pytorch.models.Trainers.custom_transforms.custom_transforms import PhaseShuffle


class Transpose1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1,
                 use_batch_norm=False) -> NoReturn:
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample
        reflection_pad = nn.ConstantPad1d(kernel_size // 2, value=0)
        conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        conv1d.weight.data.normal_(0.0, 0.02)
        Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        batch_norm = nn.BatchNorm1d(out_channels)
        operation_list = [reflection_pad, conv1d] if self.upsample else [Conv1dTrans]
        if use_batch_norm:
            operation_list.append(batch_norm)
        self.transpose_ops = nn.Sequential(*operation_list)

    def forward(self, x):
        if self.upsample:
            # recommended by wavgan paper to use nearest upsampling
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode='nearest')
        return self.transpose_ops(x)


class Conv1D(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, alpha=0.2, shift_factor=2, stride=4, padding=11,
                 use_batch_norm=False, drop_prob=0) -> NoReturn:
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm1d(output_channels)
        self.phase_shuffle = PhaseShuffle(shift_factor)
        self.alpha = alpha
        self.use_batch_norm = use_batch_norm
        self.use_phase_shuffle = shift_factor != 0
        self.use_drop = drop_prob > 0
        self.dropout = nn.Dropout2d(drop_prob)

    def forward(self, x):
        x = self.conv1d(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = F.leaky_relu(x, negative_slope=self.alpha)
        if self.use_phase_shuffle:
            print(x.shape)
            x = self.phase_shuffle(x)
            print(x.shape)
        if self.use_drop:
            x = self.dropout(x)
        return x