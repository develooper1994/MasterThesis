from typing import NoReturn

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
    def __init__(self, shift_factor) -> NoReturn:
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
