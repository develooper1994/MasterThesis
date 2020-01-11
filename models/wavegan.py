import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


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
        if self.verbose:
            print(x.shape)

        x = F.relu(self.deconv_1(x))
        if self.verbose:
            print(x.shape)

        x = F.relu(self.deconv_2(x))
        if self.verbose:
            print(x.shape)

        x = F.relu(self.deconv_3(x))
        if self.verbose:
            print(x.shape)

        x = F.relu(self.deconv_4(x))
        if self.verbose:
            print(x.shape)

        output = torch.tanh(self.deconv_5(x))
        return output


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """
    # Copied from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
    def __init__(self, shift_factor):
        """
        Initializes phase shuffling transform
        :param shift_factor: phase shuffling transform factor
        """
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        """
        Phase shuffling transform forward pass
        :param x: input of previous layer
        :return: phase shuffled signal
        """
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape,
                                                       x.shape)
        return x_shuffle


class PhaseRemove(nn.Module):
    # TODO: Not Implemented Yet.
    """
    Not Implemented Yet.
    """
    def __init__(self):
        super(PhaseRemove, self).__init__()

    def forward(self, x):
        pass


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
        if self.verbose:
            print(x.shape)
        x = self.ps1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps3(x)

        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps4(x)

        x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)

        # TODO: Try to convert linear to conv layer
        x = x.view(-1, 256 * self.model_size)
        if self.verbose:
            print(x.shape)

        return self.fc1(x)


"""
from torch.autograd import Variable
x = Variable(torch.randn(10, 100))
G = WaveGANGenerator(verbose=True, upsample=False)
out = G(x)
print(out.shape)
D = WaveGANDiscriminator(verbose=True)
out2 = D(out)
print(out2.shape)
"""
