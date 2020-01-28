import torch
from torch import nn as nn
from torch.nn import functional as F


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

    # TODO: tensorboard throws an error because can't pytorch jit trace numpy or python objects like "regular int"
    def forward(self, x):
        """
        Phase shuffling transform forward pass
        :param x: input of previous layer
        :return: phase shuffled signal
        """
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = torch.zeros(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        # k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor  # Original and correct.
        # k_list = torch.Tensor(x.shape[0].item()).random_(0, 2 * self.shift_factor + 1) - self.shift_factor  # tb.add_graph
        k_list = k_list.numpy().astype(int)
        # k_list = k_list.type(torch.int)

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
        # TODO: second iteration idxs takes [1, 2] like 2 argument. Solve it.
        # debugging cuda is dangerous. deletes original values while accessing with index
        for k, idxs in k_map.items():
            # for idxs in idxs:  # just an idea. If it don't work, remove it.
                # idx = [idx]
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][:, :, :-k], [k, 0], mode='reflect')
            else:
                x_shuffle[idxs] = F.pad(x[idxs][:, :, -k:], [0, -k], mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape,
                                                           x.shape)
        return x_shuffle

    def apply_phaseshuffle(self, x):
        # https://github.com/fromme0528/pytorch-WaveGAN/blob/597e9eb9d6ca8dd1eed3aa630fee318ff7791ee7/wavegan.py#L90
        (batch, n_channel, x_len) = x.shape
        r = torch.zeros(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        pad_l = torch.max(r, 0)
        pad_r = torch.max(-r, 0)
        phase_start = pad_r

        padding = F.pad()
        # padding = nn.ReflectionPad1d((pad_l, pad_r, 0, 0))
        #    print("phase : ", r)
        # print("pad_l, pad_r, phase_start, x_len", pad_l, pad_r, phase_start, x_len)
        # print("x.shape", x.shape)

        for x_ in x:
            ch_, len_ = x_.shape
            x_ = x_.reshape(1, 1, ch_, len_)
            x_ = padding(x_)
            x_ = x_[:, :, :, phase_start:phase_start + len_]
            x_ = x_.reshape(ch_, len_)

        return x_

class PhaseRemove(nn.Module):
    # TODO: Not Implemented Yet.
    """
    Not Implemented Yet.
    """

    def __init__(self):
        super(PhaseRemove, self).__init__()

    def forward(self, x):
        pass
