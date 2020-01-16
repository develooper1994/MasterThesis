# TODO: It is an abstaction layer for WaveGAN-discriminator
from torch import nn as nn
from torch.nn import functional as F

from models.custom_transforms.custom_transforms import PhaseShuffle


class WaveGANDiscriminator(nn.Module):
    """
    WaveGAN Discriminators Network
    """
    def __init__(self, model_size=64, ngpus=1, num_in_channels=1, shift_factor=2,
                 alpha=0.2, verbose=False) -> None:
        """
        Initialize WaveGAN Discriminators Network
        :param model_size: Size of model for flat input

            default=64
        :param ngpus: number of gpu

            default=1
        :param num_in_channels: number of input channels

            default=1
        :param shift_factor: phase shuffling transform factor

            default=2
        :param alpha: negative(left) part slope of leakyrelu

            default=0.2
        :param verbose:verbose output.
            prints size of each layer output for debugging.

            default=False
        """
        super(WaveGANDiscriminator, self).__init__()
        self.model_size = model_size  # d
        self.ngpus = ngpus
        self.num_channels = num_in_channels  # c
        self.shift_factor = shift_factor  # n
        self.alpha = alpha
        self.verbose = verbose

        # Reduce data size for discriminate
        self.conv1 = nn.Conv1d(num_in_channels, model_size, 25, stride=4, padding=11)
        self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=4, padding=11)
        self.conv3 = nn.Conv1d(2 * model_size, 4 * model_size, 25, stride=4, padding=11)
        self.conv4 = nn.Conv1d(4 * model_size, 8 * model_size, 25, stride=4, padding=11)
        self.conv5 = nn.Conv1d(8 * model_size, 16 * model_size, 25, stride=4, padding=11)

        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)

        self.fc1 = nn.Linear(256 * model_size, 1)  # last layer

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)  # He initialization of weights

    def forward(self, x):
        """
        Forward pass of Discriminators Netowkr
        :param x: input from previous layer
        :return: output of Discriminators Network
        """
        # TODO: Experiment with BatchNorm1d Layer
        # Try to DCGAN first and than Parallel WaveGAN, Progressive Growing and EfficientNet approaches.
        x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha)
        self.print_shape(x)
        x = self.ps1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha)
        self.print_shape(x)
        x = self.ps2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha)
        self.print_shape(x)
        x = self.ps3(x)

        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha)
        self.print_shape(x)
        x = self.ps4(x)

        x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha)
        self.print_shape(x)

        # TODO: Try to convert linear to conv layer
        x = x.view(-1, 256 * self.model_size)
        self.print_shape(x)

        return self.fc1(x)

    def print_shape(self, x):
        if self.verbose:
            print(x.shape)