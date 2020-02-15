import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.spectral_norm import spectral_norm

from config import device, opts

from typing import Tuple, List, Union, Optional


def build_norm_layer(norm_type, param=None, num_feats=None):
    if norm_type == 'bnorm':
        return nn.BatchNorm1d(num_feats)
    elif norm_type == 'snorm':
        spectral_norm(param)
        return None
    elif norm_type == 'vnorm':
        return VirtualBatchNorm1d(num_feats)
    elif norm_type is None:
        return None
    else:
        raise TypeError('Unrecognized norm type: ', norm_type)


def forward_norm(x: torch.Tensor, norm_layer) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if norm_layer is not None:
        return norm_layer(x)
    else:
        return x


# class OctConv(torch.nn.Module):
#     """
#     This module implements the OctConv paper https://arxiv.org/pdf/1904.05049v1.pdf
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, alpha_in=0.5, alpha_out=0.5):
#         super(OctConv, self).__init__()
#         self.alpha_in, self.alpha_out, self.kernel_size = alpha_in, alpha_out, kernel_size
#         self.H2H, self.L2L, self.H2L, self.L2H = None, None, None, None
#         self.stride = stride
#         self.padding = (kernel_size - stride) // 2
#
#         self.ch_in_lf = int(self.alpha_in*in_channels)
#         if not self.ch_in_lf > 0:
#             self.ch_in_lf = 1
#         self.ch_in_hf = in_channels - self.ch_in_lf
#
#         self.ch_out_lf = int(self.alpha_out*out_channels)
#         if not self.ch_out_lf > 0:
#             self.ch_in_lf = 1
#         self.ch_out_hf = out_channels - self.ch_out_lf
#
#         if not (alpha_in == 0.0 and alpha_out == 0.0):
#             self.L2L = torch.nn.Conv1d(in_channels=self.ch_in_lf, out_channels=self.ch_out_lf,
#                                        kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
#         if not (alpha_in == 0.0 and alpha_out == 1.0):
#             self.L2H = torch.nn.Conv1d(self.ch_in_lf,
#                                        out_channels - int(alpha_out * out_channels),
#                                        kernel_size, stride, kernel_size//2)
#         if not (alpha_in == 1.0 and alpha_out == 0.0):
#             self.H2L = torch.nn.Conv1d(in_channels - self.ch_in_lf,
#                                        int(alpha_out * out_channels),
#                                        kernel_size, stride, kernel_size//2)
#         if not (alpha_in == 1.0 and alpha_out == 1.0):
#             self.H2H = torch.nn.Conv1d(in_channels - int(alpha_in * in_channels),
#                                        out_channels - int(alpha_out * out_channels),
#                                        kernel_size, stride, kernel_size//2)
#         self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
#         self.avg_pool = partial(torch.nn.functional.avg_pool2d, kernel_size=kernel_size, stride=kernel_size)
#
#     def forward(self, x):
#         hf, lf = x
#         h2h, l2l, h2l, l2h = None, None, None, None
#         if self.H2H is not None:
#             h2h = self.H2H(hf)
#         if self.L2L is not None:
#             l2l = self.L2L(lf)
#         if self.H2L is not None:
#             h2l = self.H2L(self.avg_pool(hf))
#         if self.L2H is not None:
#             l2h = self.upsample(self.L2H(lf))
#         hf_, lf_ = 0, 0
#         for i in [h2h, l2h]:
#             if i is not None:
#                 hf_ = hf_ + i
#         for i in [l2l, h2l]:
#             if i is not None:
#                 lf_ = lf_ + i
#         return hf_, lf_

index_counter = 1

# Reducing Spatial Redundancy to speed up training.
class OctConv(nn.Module):
    def __init__(self, ninp, fmaps, kwidth, stride=1, bias=True, alphas=None):
        super(OctConv, self).__init__()

        # Get layer parameters
        if alphas is None:
            alphas = [0.5, 0.5]
        self.alpha_in, self.alpha_out = alphas
        assert 0 <= self.alpha_in <= 1 and 0 <= self.alpha_in <= 1, \
            "Alphas must be in interval [0, 1]"
        self.kernel_size = kwidth
        self.stride = stride // 4
        self.padding = (kwidth - stride) // 2
        self.bias = bias

        # Calculate the exact number of high/low frequency channels
        self.ch_in_lf = int(self.alpha_in * ninp)
        self.ch_in_hf = ninp - self.ch_in_lf
        self.ch_out_lf = int(self.alpha_out * fmaps)
        self.ch_out_hf = fmaps - self.ch_out_lf

        # Create convolutional and other modules necessary. Not all paths
        # will be created in call cases. So we check number of high/low freq
        # channels in input/output to determine which paths are present.
        # Example: First layer has alpha_in = 0, so hasLtoL and hasLtoH (bottom
        # two paths) will be false in this case.
        self.hasLtoL = self.hasLtoH = self.hasHtoL = self.hasHtoH = False
        if self.ch_in_lf and self.ch_out_lf:
            # Green path at bottom.
            self.hasLtoL = True
            self.conv_LtoL = nn.Conv1d(self.ch_in_lf, self.ch_out_lf, self.kernel_size, stride=self.stride,
                                       padding=self.padding, bias=self.bias)
        if self.ch_in_lf and self.ch_out_hf:
            # Red path at bottom.
            self.hasLtoH = True
            self.conv_LtoH = nn.Conv1d(self.ch_in_lf, self.ch_out_hf, self.kernel_size, stride=self.stride,
                                       padding=self.padding, bias=self.bias)
        if self.ch_in_hf and self.ch_out_lf:
            # Red path at top
            self.hasHtoL = True
            self.conv_HtoL = nn.Conv1d(self.ch_in_hf, self.ch_out_lf, self.kernel_size, stride=self.stride,
                                       padding=self.padding, bias=self.bias)
        if self.ch_in_hf and self.ch_out_hf:
            # Green path at top
            self.hasHtoH = True
            self.conv_HtoH = nn.Conv1d(self.ch_in_hf, self.ch_out_hf, self.kernel_size, stride=self.stride,
                                       padding=self.padding, bias=self.bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        # self.avg_pool = nn.AvgPool2d(2, 2)
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, input):
        global index_counter
        # Split input into high frequency and low frequency components
        fmap_w = input.shape[-1]
        # fmap_h = input.shape[-2]

        # We resize the high freqency components to the same size as the low
        # frequency component when sending out as output. So when bringing in as
        # input, we want to reshape it to have the original size as the intended
        # high frequnecy channel (if any high frequency component is available).
        input_hf = input
        if self.ch_in_lf:
            input_hf = input[:, :self.ch_in_hf * 2, :].reshape(-1, self.ch_in_hf, fmap_w * 2)
            input_lf = input[:, self.ch_in_hf * 2:, :]

            # Create all conditional branches
        LtoH = HtoH = LtoL = HtoL = 0.
        if self.hasLtoL:
            # Since, there is no change in spatial dimensions between input and
            # output, we use vanilla convolution
            self.conv_LtoL = nn.Conv1d(self.ch_in_lf // 2, self.ch_out_lf // 2, self.kernel_size, stride=self.stride,
                                       padding=self.padding, bias=self.bias).to(device)
            LtoL = self.conv_LtoL(input_lf)
            if LtoL.shape[-1] % 2:  # if last dim odd. cannot perform reshaping op. with odd dim.
                LtoL = F.pad(LtoL, [0, 1], "reflect")  # TODO: process my corrupt further calculations!!!
        if self.hasHtoH:
            # Since, there is no change in spatial dimensions between input and
            # output, we use vanilla convolution
            HtoH = self.conv_HtoH(input_hf)
            # We want the high freq channels and low freq channels to be
            # packed together such that the output has one dimension. This
            # enables octave convolution to be used as is with other layers
            # like Relu, elementwise etc. So, we fold the high-freq channels
            # to make its height and width same as the low-freq channels. So,
            # h = h/2 and w = w/2 since we are making h and w smaller by a
            # factor of 2, the number of channels increases by 4.
            # op_h = HtoH.shape[-2] //2
            # op_w = HtoH.shape[-1] // 2
            # HtoH = HtoH.reshape(-1, self.ch_out_hf * 4, op_h, op_w)
            if HtoH.shape[-1] % 2:  # if last dim odd. cannot perform reshaping op. with odd dim.
                HtoH = F.pad(HtoH, [0, 1], "reflect")  # TODO: process my corrupt further calculations!!!
            op_w = HtoH.shape[-1] // 2
            HtoH = HtoH.reshape(-1, self.ch_out_hf * 2, op_w)[..., :-1]  # remove padding
            # index_counter += 1

        if self.hasLtoH:
            # Since, the spatial dimension has to go up, we do
            # bilinear interpolation to increase the size of output
            # feature maps
            self.conv_LtoH = nn.Conv1d(self.ch_in_lf // 2, self.ch_out_lf // 2, self.kernel_size, stride=self.stride,
                                       padding=self.padding, bias=self.bias).to(device)
            # LtoH = F.interpolate(self.conv_LtoH(input_lf), scale_factor=2, mode='bilinear')  # bilinear nneds 4D input
            LtoH = self.upsample(self.conv_LtoH(input_lf))
            # We want the high freq channels and low freq channels to be
            # packed together such that the output has one dimension. This
            # enables octave convolution to be used as is with other layers
            # like Relu, elementwise etc. So, we fold the high-freq channels
            # to make its height and width same as the low-freq channels. So,
            # h = h/2 and w = w/2 since we are making h and w smaller by a
            # factor of 2, the number of channels increases by 4.
            # op_h = LtoH.shape[-2] //2
            op_w = LtoH.shape[-1] // 2
            if op_w % 4:
                LtoH = F.pad(LtoH, [0, 1], "reflect")
            if LtoH.shape[-1] % 2:  # if last dim odd. cannot perform reshaping op. with odd dim.
                LtoH = F.pad(LtoH, [0, 1], "reflect")  # TODO: process my corrupt further calculations!!!
            op_w = LtoH.shape[-1] // 4
            LtoH = LtoH.reshape(-1, self.ch_out_hf * 2, op_w)  # [..., :-index_counter + 2]
            if LtoH.shape[-1] * 2 == HtoH.shape[-1]:
                LtoH = self.upsample(LtoH)
            # LtoH = LtoH.reshape(-1, self.ch_out_hf * 2, op_w)
            # LtoH = LtoH.reshape(-1, self.ch_out_hf * 4, op_h, op_w)
        if self.hasHtoL:
            # Since, the spatial dimension has to go down here, we do
            # average pooling to reduce the height and width of output
            # feature maps by a factor of 2

            HtoL = self.avg_pool(self.conv_HtoL(input_hf))  # reduces spatial dimension
            if HtoL.shape[-1] % 2:
                HtoL = HtoL[..., :-1]

        # Elementwise addition of high and low freq branches to get the output
        out_hf = LtoH + HtoH
        out_lf = LtoL + HtoL

        # Since, not all paths are always present, we need to put a check
        # on how the output is generated. Example: the final convolution layer
        # will have alpha_out == 0, so no low freq. output channels,
        # so the layers returns just the high freq. components. If there are no
        # high freq component then we send out the low freq channels (we have it
        # just to have a general module even though this scenerio has not been
        # used by the authors). If both low and high freq components are present,
        # we concat them (we have already resized them to be of the same dimension)
        # and send them out.
        if self.ch_out_lf == 0:
            return out_hf
        if self.ch_out_hf == 0:
            return out_lf
        op = torch.cat([out_hf, out_lf], dim=1)
        return op


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


# TODO: refactor...
class VirtualBatchNorm1d(nn.Module):
    """
    Module for Virtual Batch Normalization.
    Implementation borrowed and modified from Rafael_Valle's code + help of SimonW from this discussion thread:
    https://discuss.pytorch.org/t/parameter-grad-of-conv-weight-is-none-after-virtual-batch-normalization/9036

    Virtual Batch Normalization Module as proposed in the paper
    `"Improved Techniques for Training GANs by Salimans et. al." <https://arxiv.org/abs/1805.08318>`_

    Performs Normalizes the features of a batch based on the statistics collected on a reference
    batch of samples that are chosen once and fixed from the start, as opposed to regular
    batch normalization that uses the statistics of the batch being normalized

    Virtual Batch Normalization requires that the size of the batch being normalized is at least
    a multiple of (and ideally equal to) the size of the reference batch. Keep this in mind while
    choosing the batch size in ```torch.utils.data.DataLoader``` or use ```drop_last=True```

    .. math:: y = \frac{x - \mathrm{E}[x_{ref}]}{\sqrt{\mathrm{Var}[x_{ref}] + \epsilon}} * \gamma + \beta

    where

    - :math:`x` : Batch Being Normalized
    - :math:`x_{ref}` : Reference Batch

    Args:
        in_features (int): Size of the input dimension to be normalized
        eps (float, optional): Value to be added to variance for numerical stability while normalizing
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super(VirtualBatchNorm1d, self).__init__()
        # batch statistics
        self.num_features = num_features
        self.eps = eps  # epsilon
        self.ref_mean = None
        self.ref_mean_sq = None

        # define gamma and beta parameters
        self.gamma = Parameter(torch.normal(mean=torch.ones(1, num_features, 1) - 0.1, std=0.02).type(torch.float))
        self.beta = Parameter(torch.zeros(1, num_features, 1))
        # self.gamma = Parameter(gamma)
        # self.beta = Parameter(torch.zeros(1, num_features, 1))

    def get_stats(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Calculates mean and mean square for given batch x.
        Args:
            x: tensor containing batch of activations
        Returns:
            mean: mean tensor over features
            mean_sq: squared mean tensor over features
        """
        mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x ** 2).mean(2, keepdim=True).mean(0, keepdim=True)
        return [mean, mean_sq]

    def forward(self, x: torch.Tensor, ref_mean: None, ref_mean_sq: None) -> List[torch.Tensor]:
        """
        Forward pass of virtual batch normalization.
        Virtual batch normalization require two forward passes
        for reference batch and train batch, respectively.
        The input parameter is_reference should indicate whether it is a forward pass
        for reference batch or not.
        Args:
            x: input tensor
            is_reference(bool): True if forwarding for reference batch
        Result:
            x: normalized batch tensor
            :param ref_mean_sq:
            :param ref_mean:
        """
        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            # reference mode - works just like batch norm
            out, mean, mean_sq = self.forward_first(x, mean, mean_sq)
        else:
            # calculate new mean and mean_sq
            out, mean, mean_sq = self.forward_head(x, mean, mean_sq, ref_mean, ref_mean_sq)
        return [out, mean, mean_sq]

    def forward_first(self, x: torch.Tensor, mean, mean_sq) -> List[torch.Tensor]:
        # reference mode - works just like batch norm
        mean = mean.clone().detach()
        mean_sq = mean_sq.clone().detach()
        out = self._normalize(x, mean, mean_sq)
        return [out, mean, mean_sq]

    def forward_head(self, x: torch.Tensor, mean, mean_sq, ref_mean, ref_mean_sq) -> List[torch.Tensor]:
        # calculate new mean and mean_sq
        batch_size = x.size(0)
        new_coeff = 1. / (batch_size + 1.)
        old_coeff = 1. - new_coeff
        mean = new_coeff * mean + old_coeff * ref_mean
        mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
        out = self._normalize(x, mean, mean_sq)
        return [out, mean, mean_sq]

    def _normalize(self, x: torch.Tensor, mean, mean_sq):
        """
        Normalize tensor x given the statistics.
        Args:
            x: input tensor
            mean: mean over features. it has size [1:num_features:]
            mean_sq: squared means over features.
        Result:
            x: normalized batch tensor
        """

        ## TODO: disabled runtime control for JIT
        if not torch.jit.is_scripting():
            self._normalize_runtime_control(x, mean, mean_sq)

        std = torch.sqrt(self.eps + mean_sq - mean ** 2)
        x = (x - mean) / std * self.gamma + self.beta
        return x

    def _normalize_runtime_control(self, x: torch.Tensor, mean: torch.Tensor, mean_sq: torch.Tensor) -> None:
        assert mean is not None
        assert mean_sq is not None
        assert len(x.size()) == 3  # specific for 1d VBN
        if mean.size(1) != self.num_features:
            raise Exception('Mean size not equal to number of featuers : given {}, expected {}'
                            .format(mean.size(1), self.num_features))
        if mean_sq.size(1) != self.num_features:
            raise Exception('Squared mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean_sq.size(1), self.num_features))

    def __repr__(self):
        return ('{name}(num_features={num_features}, eps={eps}'
                .format(name=self.__class__.__name__, **self.__dict__))


class ResBlock1D(nn.Module):
    def __init__(self, num_inputs, hidden_size, kwidth, dilation: int = 1, bias: bool = True, norm_type=None,
                 hid_act=nn.ReLU(inplace=True), out_act=None, skip_init=0):
        super(ResBlock1D, self).__init__()
        # first conv level to expand/compress features
        self.entry_conv = nn.Conv1d(num_inputs, hidden_size, 1, bias=bias)
        self.entry_norm = build_norm_layer(norm_type, self.entry_conv, hidden_size)
        self.entry_act = hid_act
        # second conv level to exploit temporal structure
        self.mid_conv = nn.Conv1d(hidden_size, hidden_size, kwidth, dilation=dilation, bias=bias)
        self.mid_norm = build_norm_layer(norm_type, self.mid_conv, hidden_size)
        self.mid_act = hid_act
        # third conv level to expand/compress features back
        self.exit_conv = nn.Conv1d(hidden_size, num_inputs, 1, bias=bias)
        self.exit_norm = build_norm_layer(norm_type, self.exit_conv, num_inputs)
        if out_act is None:
            out_act = hid_act
        self.exit_act = out_act
        self.kwidth = kwidth
        self.dilation = dilation
        # self.skip_alpha = nn.Parameter(torch.FloatTensor([skip_init]))
        self.skip_alpha = torch.FloatTensor([skip_init])

    def forward(self, x):
        # entry level
        h = self.entry_conv(x)
        h = forward_norm(h, self.entry_norm)
        h = self.entry_act(h)
        # mid level
        # first padding
        kw_2 = self.kwidth // 2
        P = kw_2 + kw_2 * (self.dilation - 1)
        h_p = F.pad(h, [P, P], mode='reflect')
        h = self.mid_conv(h_p)
        h = forward_norm(h, self.mid_norm)
        h = self.mid_act(h)
        # exit level
        h = self.exit_conv(h)
        h = forward_norm(h, self.exit_norm)
        # skip connection + exit_act
        return self.exit_act(self.skip_alpha * x + h)


class GConv1DBlock(nn.Module):
    def __init__(self, ninp, fmaps, kwidth, stride=1, bias=True, norm_type=None):
        super(GConv1DBlock, self).__init__()
        self.fmaps = fmaps
        self.conv = nn.Conv1d(ninp, fmaps, kwidth, stride=stride, bias=bias)  # vanilla conv layer

        # OctConv not work well
        # self.conv = OctConv(ninp, fmaps, kwidth, stride, bias)
        self.norm = build_norm_layer(norm_type, self.conv, fmaps)
        self.act = nn.PReLU(fmaps, init=0)
        self.kwidth = kwidth
        self.stride = stride

        P1 = int(self.kwidth // 2 - int(self.stride > 1))
        P2 = int(self.kwidth // 2)
        self.P: List[int, int] = [P1, P2]

    def forward(self, x: torch.Tensor, ret_linear: bool = False) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        x_p = F.pad(x, self.P, mode='reflect')
        a = self.conv(x_p)
        # a = forward_norm(a, self.norm)
        if self.norm is not None:
            a = self.norm(a)
        h = self.act(a)  # original
        # OctConv isn't work well
        # a_hf = a[:, :self.fmaps, :]
        # a_lf = a[:, self.fmaps:, :]
        # h_hf = self.act(a_hf)
        # h_lf = nn.PReLU(a.shape[-2]-self.fmaps, init=0).to(device)(a_lf)
        # h = torch.cat([h_hf, h_lf], dim=1)
        if ret_linear:
            # OctConv isn't work well
            # a = torch.cat([a_hf, a_lf], dim=1)
            return h, a
        else:
            return h, None


class VirtualGConv1DBlock(GConv1DBlock):
    def __init__(self, ninp, fmaps, kwidth, stride=1, bias=True, norm_type=None):
        super(VirtualGConv1DBlock, self).__init__(ninp, fmaps, kwidth, stride=stride, bias=bias, norm_type=norm_type)
        self.ref_input = torch.rand(ninp, fmaps).unsqueeze(0)
        # !!! full sinc-conv ValueError: autodetected range of [nan, nan] is not finite !!!
        # (Iter 8968) Batch 300 / 788(Epoch 12) d_real: nan, d_fake: nan, g_adv: nan, g_l1: nan l1_w: 100.00, btime: 4.3358 s, mbtime: 4.5765 s
        self.conv = nn.Conv1d(ninp, fmaps, kwidth, stride=stride, bias=bias)  # vanilla conv layer

        P1 = int(self.kwidth // 2 - int(self.stride > 1))
        P2 = int(self.kwidth // 2)
        self.P: List[int, int] = [P1, P2]

        self.norm = VirtualBatchNorm1d(fmaps)
        self.mean, self.mean_sq = torch.tensor(0), torch.tensor(0)

    def forward(self, x: torch.Tensor, ret_linear: bool = False) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        x_p = F.pad(x, self.P, mode='reflect')
        self.ref_input = x_p
        ref_a = self.conv(self.ref_input)
        # ref_a, mean, mean_sq = self.forward_norm(ref_a, self.norm, ref_mean=None, ref_mean_sq=None)
        mean_list  = self.norm.get_stats(ref_a)
        self.mean = mean_list[0]
        self.mean_sq = mean_list[1]
        forward_first_list = self.norm.forward_first(ref_a, mean=self.mean, mean_sq=self.mean_sq)
        ref_a = forward_first_list[0]
        mean_first = forward_first_list[1]
        mean_sq_first = forward_first_list[2]

        ref_h = self.act(ref_a)
        self.ref_input = ref_h
        a = self.conv(x_p)
        # a, _, _ = self.forward_norm(a, self.norm, ref_mean=mean, ref_mean_sq=mean_sq)
        a, _, _ = self.norm.forward_head(a, mean=self.mean, mean_sq=self.mean_sq, ref_mean=mean_first, ref_mean_sq=mean_sq_first)
        h = self.act(a)
        if ret_linear:
            return h, a
        else:
            return h, None

    # @torch.jit.export
    # def forward_norm(self, x: torch.Tensor, norm_layer, ref_mean, ref_mean_sq):
    #     if norm_layer is not None:
    #         return norm_layer(x, ref_mean=ref_mean, ref_mean_sq=ref_mean_sq)
    #     else:
    #         return x


class GDeconv1DBlock(nn.Module):
    def __init__(self, ninp, fmaps, kwidth, stride=4, bias=True, norm_type=None, act=None):
        super(GDeconv1DBlock, self).__init__()
        pad = max(0, (stride - kwidth) // -2)
        self.deconv = nn.ConvTranspose1d(ninp, fmaps, kwidth, stride=stride, padding=pad)
        self.norm = build_norm_layer(norm_type, self.deconv, fmaps)
        self.act = getattr(nn, act)() if act is not None else nn.PReLU(fmaps, init=0)
        self.kwidth = kwidth
        self.stride = stride

    def forward(self, x):
        h = self.deconv(x)
        if self.kwidth % 2 != 0:
            h = h[:, :, :-1]
        # h = forward_norm(h, self.norm)
        if self.norm is not None:
            h = self.norm(h)
        h = self.act(h)
        return h


class ResARModule(nn.Module):
    def __init__(self, ninp, fmaps, res_fmaps, kwidth, dilation, bias=True, norm_type=None, act=None):
        super(ResARModule, self).__init__()
        self.dil_conv = nn.Conv1d(ninp, fmaps, kwidth, dilation=dilation, bias=bias)
        self.act = getattr(nn, act)() if act is not None else nn.PReLU(fmaps, init=0)
        self.dil_norm = build_norm_layer(norm_type, self.dil_conv, fmaps)
        self.kwidth = kwidth
        self.dilation = dilation
        # skip 1x1 convolution
        self.conv_1x1_skip = nn.Conv1d(fmaps, ninp, 1, bias=bias)
        self.conv_1x1_skip_norm = build_norm_layer(norm_type, self.conv_1x1_skip, ninp)
        # residual 1x1 convolution
        self.conv_1x1_res = nn.Conv1d(fmaps, res_fmaps, 1, bias=bias)
        self.conv_1x1_res_norm = build_norm_layer(norm_type, self.conv_1x1_res, res_fmaps)

    def forward(self, x):
        kw__1 = self.kwidth - 1
        P = kw__1 + kw__1 * (self.dilation - 1)
        # causal padding
        x_p = F.pad(x, (P, 0))
        # dilated conv
        h = self.dil_conv(x_p)
        # normalization if applies
        h = forward_norm(h, self.dil_norm)
        # activation
        h = self.act(h)
        a = h
        # conv 1x1 to make residual connection
        h = self.conv_1x1_skip(h)
        # normalization if applies
        h = forward_norm(h, self.conv_1x1_skip_norm)
        # return with skip connection
        y = x + h
        # also return res connection (going to further net point directly)
        sh = self.conv_1x1_res(a)
        sh = forward_norm(sh, self.conv_1x1_res_norm)
        return y, sh


# Modified from https://github.com/mravanelli/SincNet
class SincConv(nn.Module):
    # SincNet as proposed in
    # https://arxiv.org/abs/1808.00158
    def __init__(self, N_filt, Filt_dim, fs, stride=1, padding=True):
        """

        :param N_filt: number of filters
        :param Filt_dim: filter kernel size
        :param fs: sampling frequency
        :param stride:
        :param bias:
        :param padding: (Default: True -> 'VALID')
        True -> 'VALID'
        False -> 'SAME'
        """
        super(SincConv, self).__init__()
        self.N_filt = torch.tensor(N_filt, dtype=torch.int).to(device)
        self.Filt_dim = torch.tensor(Filt_dim).to(device)
        self.fs = torch.tensor(fs, dtype=torch.float).to(device)
        self.stride = stride
        if padding:
            self.padding = False  # 'VALID'
        else:
            self.padding = True  # 'SAME'

        # Mel Initialization of the filterbanks
        low_freq_mel = torch.tensor(80, dtype=torch.float)
        high_freq_mel = (2595 * torch.log10(1 + (self.fs / 2) / 700))  # Convert Hz to Mel
        mel_points = torch.linspace(low_freq_mel, high_freq_mel, steps=self.N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        b1, b2 = torch.roll(f_cos, 1), torch.roll(f_cos, -1)
        b1[0], b2[-1] = 30, (fs / 2) - 100

        self.freq_scale = self.fs * 1.0
        # self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
        # self.filt_band = nn.Parameter(torch.from_numpy((b2 - b1) / self.freq_scale))
        self.filt_b1 = torch.abs(b1 / self.freq_scale)
        self.filt_band = torch.abs((b2 - b1) / self.freq_scale)

        self.filters = torch.zeros((self.N_filt, self.Filt_dim)).to(device)
        self.t_right = (torch.linspace(1, (self.Filt_dim - 1) / 2, steps=int((self.Filt_dim - 1) / 2)) / self.fs).to(
            device)
        self.ones = torch.ones(1).to(device)  # to prevent allocation on device in loop
        self.sinc_result = torch.zeros(self.Filt_dim).to(device)  # to prevent allocation on device in loop
        self.min_freq, self.min_band = 50.0, 50.0
        self.n = torch.linspace(0, self.Filt_dim, steps=self.Filt_dim)
        self.filt_beg_freq = self.filt_b1 + self.min_freq / self.freq_scale
        self.filt_end_freq = self.filt_beg_freq + (self.filt_band + self.min_band / self.freq_scale)
        # Filter window
        self.windowing = Windowing(self.n, self.N_filt, self.Filt_dim)
        self.window = self.windowing.get_window()

        # Filter window (hamming)
        self.band_pass = torch.zeros((1)).to(device)
        for i in range(self.N_filt):
            self.low_pass1 = 2 * self.filt_beg_freq[i].float() * \
                             self.get_filter(self.filt_beg_freq[i].float() * self.freq_scale, self.t_right)
            self.low_pass2 = 2 * self.filt_end_freq[i].float() * \
                             self.get_filter(self.filt_end_freq[i].float() * self.freq_scale, self.t_right)
            self.band_pass = (self.low_pass2 - self.low_pass1)
            self.band_pass = self.band_pass / torch.max(self.band_pass)

            self.filters[i, :] = self.band_pass * self.window  # convolution filters.

        P1 = int(self.Filt_dim // 2 - int(self.stride > 1))
        P2 = int(self.Filt_dim // 2)
        self.P: List[int, int] = [P1, P2]

    # all of them x dependent. no more optimization for speed!
    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.padding:
            x_p = F.pad(x, self.P, mode='reflect')
        else:
            x_p = x
        filter_size = x_p.shape[-2]
        return F.conv1d(x_p, self.filters.view(self.N_filt, 1, self.Filt_dim).repeat((1, filter_size, 1)), stride=self.stride)

    # !! DON'T REMOVE !!
    # def flip(self, x, dim):
    #     xsize = x.size()
    #     dim = x.dim() + dim if dim < 0 else dim
    #     x = x.view(-1, *xsize[dim:])
    #     x = torch.flip(x, [0, 1])
    #     # x = x.view(x.size(0), x.size(1), -1)[:, torch.arange(x.size(1) - 1, -1, -1).long(), :]
    #     # # x.unsqueeze_(2)
    #     # return x.view(xsize)
    #     return x.unsqueeze(2).view(xsize)

    def get_filter(self, band, t_right):
        y_right = self.filter(band, t_right)
        yr_s = y_right.shape[0]
        # y_left = self.flip(y_right, 0)

        # time domain. cannot remove one part of filter.
        # return torch.cat([torch.flip(y_right, [0]), self.ones, y_right])  # get_filter is a symetric function.

        # concatenate without copy
        self.sinc_result[:yr_s] = torch.flip(y_right, [0])
        self.sinc_result[yr_s:yr_s + 1] = self.ones
        self.sinc_result[yr_s + 1:] = y_right

        return self.sinc_result

    def filter(self, band, t_right):
        y_right = self.sinc(band, t_right)
        # if you change to somethingelse windowing changes
        # add more windowing functions like blackman. You can find from Mathwork or Wikipedia.
        return y_right

    @torch.jit.export
    def sinc(self, band, t_right):
        y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
        return y_right


class Windowing:
    # https://en.wikipedia.org/wiki/Window_function#A_list_of_window_functions
    # http://www.dspguide.com/ch16/2.htm
    # https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/utils/windows.py#L108-L142
    def __init__(self, n, N_filt, Filt_dim):
        self.n = n
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim

    def get_window(self, coefs=None):
        """
        https://en.wikipedia.org/wiki/Window_function#A_list_of_window_functions
        some windowing formulas to apply and learn from data.
        :param coefs: a subindex k coefficients for windowing (Default: [0.54, 0.46] for hamming)
        :return:
        """
        ## Filter window
        # http://www.dspguide.com/ch16/2.htm

        cosine_const2 = torch.cos(2 * math.pi * self.n / self.Filt_dim)
        if opts.windowing.lower() == "hamming":
            # (hamming)
            # window = self.windowing.hamming(self.Filt_dim).to(device)
            window = (0.54 - 0.46 * cosine_const2).float().to(device)
        elif opts.windowing.lower() == "hann":
            # window = self.windowing.hann(self.Filt_dim).to(device)
            window = 0.5 * (1 - cosine_const2)
        elif opts.windowing.lower() == "blackman":
            # window = self.windowing.hann(self.Filt_dim).to(device)
            window = (0.42 - 0.5 * cosine_const2 + 0.08 * torch.cos(4 * math.pi * self.n / self.Filt_dim)).to(device)
        elif opts.windowing.lower() == "blackman_harris":
            window = self.blackman_harris(self.Filt_dim).to(device)
        elif opts.windowing.lower() == "rectangular":
            window = self.rectangular()
            # window = torch.tensor(1).to(device)
        elif opts.windowing.lower() == "triangular":
            window = self.triangular().to(device)
            # window = (1 - torch.abs((self.n - self.N_filt / 2) / self.N_filt / 2)).to(device)
        else:
            # !!! SLLOOOOOOWWW !!!
            window = self.generalized_cosine(coefs)
        return window

    @torch.jit.export
    def rectangular(self):
        return torch.tensor(1)

    @torch.jit.export
    def triangular(self):
        return 1 - torch.abs((self.n - self.N_filt / 2) / self.N_filt / 2)

    ## Cosine-sum windows
    @torch.jit.export
    def generalized_cosine(self, coefs):
        """
            The generalized cosine family of window functions.
            Notes
            -----
            The generalized cosine window is a simple weighted sum of cosine terms.
            For :math:`n \in \{0, \ldots, \\text{window_len} \}`:
            .. math::
                \\text{GCW}(n) = \sum_{k=0}^K (-1)^k a_k \cos\left(\\frac{2 \pi k n}{\\text{window_len}}\\right)
            Parameters
            ----------
            window_len : int
                The length of the window in samples. Should be equal to the
                `frame_width` if applying to a windowed signal.
            coefs: list of floats
                The :math:`a_k` coefficient values
            symmetric : bool
                If False, create a 'periodic' window that can be used in with an FFT /
                in spectral analysis.  If True, generate a symmetric window that can be
                used in, e.g., filter design. Default is False.
            Returns
            -------
            window : :py:class:`ndarray <numpy.ndarray>` of shape `(window_len,)`
                The window
            """
        return torch.sum(
            [((-1) ** k) * a * torch.cos(2 * math.pi * k * self.n / self.N_filt) for k, a in enumerate(coefs)],
            axis=0)

    @torch.jit.export
    def hann(self, window_len):
        """
            The Hann window.
            Notes
            -----
            The Hann window is an instance of the more general class of cosine-sum
            windows where `K=1` and :math:`a_0` = 0.5. Unlike the Hamming window, the
            end points of the Hann window touch zero.
            .. math::
                \\text{hann}(n) = 0.5 - 0.5 \cos\left(\\frac{2 \pi n}{\\text{window_len} - 1}\\right)
            Parameters
            ----------
            window_len : int
                The length of the window in samples. Should be equal to the
                `frame_width` if applying to a windowed signal.
            symmetric : bool
                If False, create a 'periodic' window that can be used in with an FFT /
                in spectral analysis.  If True, generate a symmetric window that can be
                used in, e.g., filter design. Default is False.
            Returns
            -------
            window : :py:class:`ndarray <numpy.ndarray>` of shape `(window_len,)`
                The window
            """
        return self.generalized_cosine(window_len, [0.5, 0.5])

    @torch.jit.export
    def hamming(self, window_len):
        """
            The Hamming window.
            Notes
            -----
            The Hamming window is an instance of the more general class of cosine-sum
            windows where `K=1` and :math:`a_0 = 0.54`. Coefficients selected to
            minimize the magnitude of the nearest side-lobe in the frequency response.
            .. math::
                \\text{hamming}(n) = 0.54 -
                    0.46 \cos\left(\\frac{2 \pi n}{\\text{window_len} - 1}\\right)
            Parameters
            ----------
            window_len : int
                The length of the window in samples. Should be equal to the
                `frame_width` if applying to a windowed signal.
            symmetric : bool
                If False, create a 'periodic' window that can be used in with an FFT /
                in spectral analysis.  If True, generate a symmetric window that can be
                used in, e.g., filter design. Default is False.
            Returns
            -------
            window : :py:class:`ndarray <numpy.ndarray>` of shape `(window_len,)`
                The window
            """
        return self.generalized_cosine(window_len, [0.54, 1 - 0.54])

    @torch.jit.export
    def blackman(self, window_len):
        # (0.42 - 0.5 * cosine_const2 + 0.08 * torch.cos(4 * math.pi * self.n / self.Filt_dim)).to(device)
        return self.generalized_cosine(window_len, [0.42659, 0.49656, 0.076849, 0.01168])

    @torch.jit.export
    def nuttall(self, window_len):
        return self.generalized_cosine(window_len, [0.3555768, 0.487396, 0.144232, 0.012604])

    @torch.jit.export
    def blackman_nuttall(self, window_len):
        return self.generalized_cosine(window_len, [0.3635819, 0.04891775, 0.0165995, 0.0106411])

    @torch.jit.export
    def blackman_harris(self, window_len):
        """
            The Blackman-Harris window.
            Notes
            -----
            The Blackman-Harris window is an instance of the more general class of
            cosine-sum windows where `K=3`. Additional coefficients extend the Hamming
            window to further minimize the magnitude of the nearest side-lobe in the
            frequency response.
            .. math::
                \\text{bh}(n) = a_0 - a_1 \cos\left(\\frac{2 \pi n}{N}\\right) +
                    a_2 \cos\left(\\frac{4 \pi n }{N}\\right) -
                        a_3 \cos\left(\\frac{6 \pi n}{N}\\right)
            where `N` = `window_len` - 1, :math:`a_0` = 0.35875, :math:`a_1` = 0.48829,
            :math:`a_2` = 0.14128, and :math:`a_3` = 0.01168.
            Parameters
            ----------
            window_len : int
                The length of the window in samples. Should be equal to the
                `frame_width` if applying to a windowed signal.
            symmetric : bool
                If False, create a 'periodic' window that can be used in with an FFT /
                in spectral analysis.  If True, generate a symmetric window that can be
                used in, e.g., filter design. Default is False.
            Returns
            -------
            window : :py:class:`ndarray <numpy.ndarray>` of shape `(window_len,)`
                The window
            """
        return self.generalized_cosine(window_len, [0.35875, 0.48829, 0.14128, 0.01168])

    @torch.jit.export
    def flat_top(self, window_len):
        return self.generalized_cosine(window_len, [0.21557895, 0.041663158, 0.277263158, 0.083578947, 0.006947368])


## Not used
class CombFilter(nn.Module):
    def __init__(self, ninputs, fmaps, L):
        super(CombFilter, self).__init__()
        self.L = L
        self.filt = nn.Conv1d(ninputs, fmaps, 2, dilation=L, bias=False)
        r_init_weight = torch.ones(ninputs * fmaps, 2)
        r_init_weight[:, 0] = torch.rand(r_init_weight.size(0))
        self.filt.weight.data = r_init_weight.view(fmaps, ninputs, 2)

    def forward(self, x):
        x_p = F.pad(x, [self.L, 0])
        return self.filt(x_p)


class PostProcessingCombNet(nn.Module):
    def __init__(self, ninputs, fmaps, L=None):
        super(PostProcessingCombNet, self).__init__()
        if L is None:
            L = [4, 8, 16, 32]
        filts = nn.ModuleList()
        for l in L:
            filt = CombFilter(ninputs, fmaps // len(L), l)
            filts.append(filt)
        self.filts = filts
        self.W = nn.Linear(fmaps, 1, bias=False)

    def forward(self, x):
        hs = []
        for filt in self.filts:
            h = filt(x)
            hs.append(h)
            # print('Comb h: ', h.size())
        hs = torch.cat(hs, dim=1)
        # print('hs size: ', hs.size())
        return self.W(hs.transpose(1, 2)).transpose(1, 2)


if __name__ == '__main__':
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    # 800 samples @ 16kHz is 50ms
    T = 800
    # n = 20 z time-samples per frame
    n = 20
    zgen = ZGen(n, T // n, 
                z_amp=0.5)
    all_z = None
    for t in range(0, 200, 5):
        time_idx = torch.LongTensor([t])
        z_ten = zgen(time_idx)
        print(z_ten.size())
        z_ten = z_ten.squeeze()
        if all_z is None:
            all_z = z_ten
        else:
            all_z = np.concatenate((all_z, z_ten), axis=1)
    N = 20
    for k in range(N):
        plt.subplot(N, 1, k + 1)
        plt.plot(all_z[k, :], label=k)
        plt.ylabel(k)
    plt.show()

    # ResBlock
    resblock = ResBlock1D(40, 100, 5, dilation=8)
    print(resblock)
    z = z_ten.unsqueeze(0)
    print('Z size: ', z.size())
    y = resblock(z)
    print('Y size: ', y.size())

    x = torch.randn(1, 1, 16) 
    deconv = GDeconv1DBlock(1, 1, 31)
    y = deconv(x)
    print('x: {} -> y: {} deconv'.format(x.size(),
                                         y.size()))
    conv = GConv1DBlock(1, 1, 31, stride=4)
    z = conv(y)
    print('y: {} -> z: {} conv'.format(y.size(),
                                       z.size()))
    """
    x = torch.randn(1, 1, 16384)
    sincnet = SincConv(1024, 251, 16000, padding='SAME')
    y = sincnet(x)
    print('y size: ', y.size())
