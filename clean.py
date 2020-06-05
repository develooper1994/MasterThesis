## Python standart imports
import json
import glob
import os
import timeit
import argparse
import random
import subprocess

## outisde of standart imports
import torch
import numpy as np
import h5py
from scipy.io import wavfile
import soundfile as sf

import matplotlib
matplotlib.use('Agg')

## my modules
# from segan.models import *
from segan.models.model import SEGAN, WSEGAN, AEWSEGAN
from segan.datasets.se_dataset import normalize_wave_minmax, pre_emphasize


def get_input_sampling(file):
    # I don't want to only support "16khz=16000hz" sampling rate.
    # please install sox software. it is only compitable with linux and macosx.
    result = subprocess.run(['soxi', file], stdout=subprocess.PIPE)
    result_stdout = result.stdout.decode('utf-8')
    result_stdout = result_stdout.splitlines()[1:-1]
    result_stdout_sampling_str = str(result_stdout[2]).split(':')[-1]
    sampling = int(result_stdout_sampling_str[1:])
    return sampling


def check_input_sampling(file):
    sampling = get_input_sampling(file)
    return sampling == 16000  # check sampling is valid.


def downsample(wave, sampling):
    k = sampling % 16000  # check sampling is valid.
    if not k:  # downsample if sampling is not 16khz
        wave = wave[::k]
    return wave


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
        #     twavs = glob.glob(os.path.join(options.test_files[0], '*.wav'))
        if os.path.isdir(options.test_files):
            # assume we read directory
            twavs = glob.glob(os.path.join(options.test_files, '*.wav'))
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
        write_sampling = options.write_sampling  # 16000
        if options.soundfile:
            sf.write(out_path, g_wav, write_sampling)
        else:
            wavfile.write(out_path, write_sampling, g_wav)
        end_t = timeit.default_timer()
        print('Cleaned {}/{}: {} in {} s'.format(t_i, len(twavs), twav,
                                                 end_t - beg_t))
        beg_t = timeit.default_timer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--g_pretrained_ckpt', type=str, default="/home/selcuk/PycharmProjects/MasterThesis/Denoiser/ckpt_wsegan_misalign/weights_EOE_G-Generator-130680.ckpt")  # None
    parser.add_argument('--test_files', type=str, nargs='+', default="/home/selcuk/.pytorch/DS_10283_2791/noisy_testset_wav/")  # None
    parser.add_argument('--h5', action='store_true', default=False,
                        help="""If you have h5 file type, check it "True' (Def: False)""")
    parser.add_argument('--seed', type=int, default=2020,
                        help="Random seed (Def: 2020).")  # 111
    parser.add_argument('--synthesis_path', type=str, default='wsegan_samples',
                        help='Path to save output samples (Def: segan_samples).')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Do you want to use cuda while cleaning audio? (Def: True)')
    parser.add_argument('--soundfile', action='store_true', default=True,
                        help='Do you want soundfile Python packes? (Def: True)')  # False
    parser.add_argument('--cfg_file', type=str, default="ckpt_wsegan_misalign/train.opts",
                        help="""train.opts configuration file inside of the same '--save_path' folder""")  # None
    parser.add_argument('--write_sampling', type=int, default=48000,
                        help='Specifying audio recodering sampling rate (Def: 48000)')  # 16000. Normally it only supports 16khz but my test wavs 48khz.

    options = parser.parse_args()

    if not os.path.exists(options.synthesis_path):
        os.makedirs(options.synthesis_path)

    # save opts
    with open(os.path.join(options.synthesis_path, 'clean.opts'), 'w') as cfg_f:
        cfg_f.write(json.dumps(vars(options), indent=2))
    print('Parsed arguments: ', json.dumps(vars(options), indent=2))

    # seed initialization
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    if options.cuda:
        torch.cuda.manual_seed_all(options.seed)

    main(options)
