from torch import nn as nn


class Transpose1dLayer(nn.Module):
    """
    Package of all 1d Convolution Transpose Layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1) -> None:
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

        self.upsample_layer = nn.Upsample(scale_factor=upsample)
        reflection_pad = kernel_size // 2  # same padding?
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        """
        Forward pass of 1d Convolution Transpose Layer package
        :param x: input from previous layer
        :return:
            If True: size getting bigger
            else: size remain same at output
        """
        if self.upsample:
            return self.conv1d(self.reflection_pad(self.upsample_layer(x)))
        else:
            return self.Conv1dTrans(x)