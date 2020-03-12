## Python standart imports
import json
import glob
import os
import timeit
import argparse
import random

## outisde of standart imports
import torch
import numpy as np
import h5py
from scipy.io import wavfile

import matplotlib
matplotlib.use('Agg')

## my modules
from segan.models import *
from Denoiser.segan import normalize_wave_minmax, pre_emphasize, SEGAN, WSEGAN, AEWSEGAN


class ArgParser(object):
    def __init__(self, args):
        self.preemph = None
        for k, v in args.items():
            setattr(self, k, v)


def main(opts):
    assert opts.cfg_file is not None
    assert opts.test_files is not None
    assert opts.g_pretrained_ckpt is not None

    with open(opts.cfg_file, 'r') as cfg_f:
        args = ArgParser(json.load(cfg_f))
        print('Loaded train config: ')
        print(json.dumps(vars(args), indent=2))
    args.cuda = opts.cuda
    if hasattr(args, 'wsegan') and args.wsegan:
        segan = WSEGAN(args)
    elif hasattr(args, 'aesegan') and args.wsegan:
        segan = AEWSEGAN(args)
    else:
        segan = SEGAN(args)
    segan.G.load_pretrained(opts.g_pretrained_ckpt, True)
    if opts.cuda:
        segan.cuda()
    segan.G.eval()
    if opts.h5:
        with h5py.File(opts.test_files[0], 'r') as f:
            twavs = f['data'][:]
    else:
        # process every wav in the test_files
        if len(opts.test_files) == 1:
            # assume we read directory
            twavs = glob.glob(os.path.join(opts.test_files[0], '*.wav'))
        else:
            # assume we have list of files in input
            twavs = opts.test_files
    print('Cleaning {} wavs'.format(len(twavs)))  # last correctly runned line
    beg_t = timeit.default_timer()
    for t_i, twav in enumerate(twavs, start=1):
        if opts.h5:
            tbname = 'tfile_{}.wav'.format(t_i)
            wav = twav
            twav = tbname
        else:
            tbname = os.path.basename(twav)
            rate, wav = wavfile.read(twav)  # TODO! error.
            wav = normalize_wave_minmax(wav)
        wav = pre_emphasize(wav, args.preemph)
        pwav = torch.FloatTensor(wav).view(1, 1, -1)
        if opts.cuda:
            pwav = pwav.cuda()
        g_wav, g_c = segan.generate(pwav)
        out_path = os.path.join(opts.synthesis_path,
                                tbname)
        if opts.soundfile:
            sf.write(out_path, g_wav, 16000)
        else:
            wavfile.write(out_path, 16000, g_wav)
        end_t = timeit.default_timer()
        print('Cleaned {}/{}: {} in {} s'.format(t_i, len(twavs), twav,
                                                 end_t - beg_t))
        beg_t = timeit.default_timer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--g_pretrained_ckpt', type=str, default=None)
    parser.add_argument('--test_files', type=str, nargs='+', default=None)
    parser.add_argument('--h5', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=2020,
                        help="Random seed (Def: 111).")  # 111
    parser.add_argument('--synthesis_path', type=str, default='segan_samples',
                        help='Path to save output samples (Def: segan_samples).')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--soundfile', action='store_true', default=False)
    parser.add_argument('--cfg_file', type=str, default=None)

    opts = parser.parse_args()

    if not os.path.exists(opts.synthesis_path):
        os.makedirs(opts.synthesis_path)

    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed_all(opts.seed)

    main(opts)
