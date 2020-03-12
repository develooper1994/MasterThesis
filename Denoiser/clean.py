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


class ArgParser(object):
    def __init__(self, args):
        self.preemph = None
        for k, v in args.items():
            setattr(self, k, v)


def main(options):
    assert options.cfg_file is not None
    assert options.test_files is not None
    assert options.g_pretrained_ckpt is not None

    with open(options.cfg_file, 'r') as cfg_f:
        args = ArgParser(json.load(cfg_f))
        print('Loaded train config: ')
        print(json.dumps(vars(args), indent=2))
    args.cuda = options.cuda
    if hasattr(args, 'wsegan') and args.wsegan:
        segan = WSEGAN(args)
    elif hasattr(args, 'aesegan') and args.wsegan:
        segan = AEWSEGAN(args)
    else:
        segan = SEGAN(args)
    segan.G.load_pretrained(options.g_pretrained_ckpt, True)
    if options.cuda:
        segan.cuda()
    segan.G.eval()
    if options.h5:
        with h5py.File(options.test_files[0], 'r') as f:
            twavs = f['data'][:]
    else:
        # process every wav in the test_files
        # if len(options.test_files) == 1:
        if os.file.isdir(options.test_files):
            # assume we read directory
            twavs = glob.glob(os.path.join(options.test_files[0], '*.wav'))
        else:
            # assume we have list of files in input
            twavs = options.test_files
    print('Cleaning {} wavs'.format(len(twavs)))  # last correctly runned line
    beg_t = timeit.default_timer()
    for t_i, twav in enumerate(twavs, start=1):
        if options.h5:
            tbname = 'tfile_{}.wav'.format(t_i)
            wav = twav
            twav = tbname
        else:
            tbname = os.path.basename(twav)
            rate, wav = wavfile.read(twav)  # TODO! error.
            wav = normalize_wave_minmax(wav)
        wav = pre_emphasize(wav, args.preemph)
        pwav = torch.FloatTensor(wav).view(1, 1, -1)
        if options.cuda:
            pwav = pwav.cuda()
        g_wav, g_c = segan.generate(pwav)
        out_path = os.path.join(options.synthesis_path,
                                tbname)
        if options.soundfile:
            sf.write(out_path, g_wav, 16000)
        else:
            wavfile.write(out_path, 16000, g_wav)
        end_t = timeit.default_timer()
        print('Cleaned {}/{}: {} in {} s'.format(t_i, len(twavs), twav,
                                                 end_t - beg_t))
        beg_t = timeit.default_timer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--g_pretrained_ckpt', type=str, default="/home/selcukcaglar08/MasterThesis/Denoiser/ckpt_segan_sinc+")  # None
    parser.add_argument('--test_files', type=str, nargs='+', default="/home/selcukcaglar08/full_audio_dataset/DS_10283_2791/noisy_testset_wav")  # None
    parser.add_argument('--h5', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=2020,
                        help="Random seed (Def: 2020).")  # 111
    parser.add_argument('--synthesis_path', type=str, default='segan_samples',
                        help='Path to save output samples (Def: segan_samples).')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--soundfile', action='store_true', default=False)  # False
    parser.add_argument('--cfg_file', type=str, default="ckpt_segan_sinc+/train.opts")  # None

    options = parser.parse_args()

    if not os.path.exists(options.synthesis_path):
        os.makedirs(options.synthesis_path)

    # seed initialization
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    if options.cuda:
        torch.cuda.manual_seed_all(options.seed)

    main(options)
