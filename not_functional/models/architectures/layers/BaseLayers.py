from typing import NoReturn

from torch import nn as nn
from torch.nn import functional as F

from not_functional.models.custom_transforms import PhaseShuffle


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class WaveResBlock(nn.Module):
    def __init__(self, ic, oc, last_act='leakyrelu', resolution_type='upsample'):
        super(WaveResBlock, self).__init__()
        self.last_act = last_act
        if resolution_type == 'upsample':
            self.resolution = UpSample1D()
        elif resolution_type == 'downsample':
            self.resolution = DownSample1D()
        self.conv1 = nn.Conv1d(ic, ic, 25, 1, 12)
        self.conv2 = nn.Conv1d(ic, ic, 25, 1, 12)
        self.conv3 = nn.Conv1d(ic, oc, 25, 1, 12)
        self.norm1 = nn.InstanceNorm1d(ic)
        self.norm2 = nn.InstanceNorm1d(ic)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.resolution(out + x)
        if self.last_act == 'sigmoid':
            out = self.sigmoid(self.conv3(out))
        elif self.last_act == 'tanh':
            out = self.tanh(self.conv3(out))
        elif self.last_act == 'leakyrelu':
            out = self.relu(self.conv3(out))
        return out


class UpSample1D(nn.Module):
    def __init__(self):
        super(UpSample1D, self).__init__()
        self.scale_factor = 4

    def forward(self, x):
        return F.interpolate(x, None, self.scale_factor, 'linear', align_corners=True)


class DownSample1D(nn.Module):
    def __init__(self):
        super(DownSample1D, self).__init__()
        self.scale_factor = 4

    def forward(self, x):
        return F.avg_pool1d(x, self.scale_factor)


class Transpose1dLayer(nn.Module):
    """
    Package of all 1d Convolution Transpose Layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1,
                 use_batch_norm=False) -> NoReturn:
        """
        Initialize 1d Convolution Transpose Layer package

        -*-*-*Convolution Transpose summary*-*-*-
        There isn't a direct back-process to convolution like deconvolution but there is a technique to retrieve most of
        the information back called "Convolution Transpose". Apply convolution with bigger kernel than image but it is
        actually not. Whenever use convolution, it produce a non-square matrix to multiply with data.
        Apply pseudo inverse to the non-square matrix and multiply with data. This method reverse at least size of data.
        -*-*-*-*-*-*-
        :param in_channels: input convolution channel size
        :param out_channels: output convolution channel size
        :param kernel_size: size of kernel
        :param stride: step size of kernel
        :param padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of the input. Default: 0

            default=11
        :param upsample: upsample or same size? applying deconvolution
            If True: size getting bigger
            else: size remain same at output

            default=None
        :param output_padding (int or tuple, optional): Additional size added to one side
            of the output shape. Default: 0

            default=1
        :return: None
        """
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
        """
        Forward pass of 1d Convolution Transpose Layer package
        :param x: input from previous layer
        :return:
            If True: size getting bigger
            else: size remain same at output
        """
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
            # print(x.shape)
            x = self.phase_shuffle(x)
            # print(x.shape)
        if self.use_drop:
            x = self.dropout(x)
        return x