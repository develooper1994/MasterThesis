import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from not_functional.config import opts, device
from segan.datasets import SEDataset, SEH5Dataset, collate_fn
from segan.models import SEGAN, WSEGAN, AEWSEGAN


def main(opts, memory_pin):
    torch.cuda.empty_cache()
    # select device to work on 
    global va_dset
    # device = torch.device("cuda" if torch.cuda.is_available() and not opts.no_cuda else "cpu")
    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    # if device.type:
    #     torch.cuda.manual_seed_all(opts.seed)
    # create SEGAN model
    if opts.wsegan:
        segan = WSEGAN(opts)
    elif opts.aewsegan:
        segan = AEWSEGAN(opts)
    else:
        segan = SEGAN(opts)
    segan = segan.to(device)
    # possibly load pre-trained sections of networks G or D
    print('Total model parameters: ', segan.get_n_params())
    if opts.g_pretrained_ckpt is not None:
        segan.G.load_pretrained(opts.g_pretrained_ckpt, True)
    if opts.d_pretrained_ckpt is not None:
        segan.D.load_pretrained(opts.d_pretrained_ckpt, True)
    # create Dataset(s) and Dataloader(s)
    if opts.h5:
        # H5 Dataset with processed speech chunks
        if opts.h5_data_root is None:
            raise ValueError('Please specify an H5 data root')
        dset = SEH5Dataset(opts.h5_data_root, split='train', preemph=opts.preemph,
                           verbose=True, random_scale=opts.random_scale)
    else:
        # Directory Dataset from raw wav files
        dset = SEDataset(opts.clean_trainset, opts.noisy_trainset, opts.preemph, do_cache=True,
                         cache_dir=opts.cache_dir, split='train', stride=opts.data_stride,
                         slice_size=opts.slice_size, max_samples=opts.max_samples,
                         verbose=True, slice_workers=opts.slice_workers,
                         preemph_norm=opts.preemph_norm, random_scale=opts.random_scale)
    if memory_pin:
        dloader = DataLoader(dset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers,
                             pin_memory=True, collate_fn=collate_fn)
    else:
        dloader = DataLoader(dset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers,
                             collate_fn=collate_fn)
    if opts.clean_valset is not None:
        if opts.h5:
            dset = SEH5Dataset(opts.h5_data_root, split='valid', preemph=opts.preemph, verbose=True)
        else:
            va_dset = SEDataset(opts.clean_valset, opts.noisy_valset, opts.preemph, do_cache=True,
                                cache_dir=opts.cache_dir, split='valid', stride=opts.data_stride,
                                slice_size=opts.slice_size, max_samples=opts.max_samples, verbose=True,
                                slice_workers=opts.slice_workers, preemph_norm=opts.preemph_norm)
        if memory_pin:
            va_dloader = DataLoader(va_dset, batch_size=300, shuffle=False, num_workers=opts.num_workers,
                                    pin_memory=True, collate_fn=collate_fn)
        else:
            va_dloader = DataLoader(va_dset, batch_size=300, shuffle=False, num_workers=opts.num_workers,
                                    collate_fn=collate_fn)
    else:
        va_dloader = None
    criterion = nn.MSELoss()
    segan.train(opts, dloader, criterion, opts.l1_weight, opts.l1_dec_step, opts.l1_dec_epoch, opts.save_freq,
                va_dloader=va_dloader)


if __name__ == '__main__':
    print('Parsed arguments: ', opts)
    main(opts, True)