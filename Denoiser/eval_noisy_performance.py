import argparse
import json
import timeit
import glob
import os
from collections import namedtuple

import librosa
import numpy as np

from segan.utils import *


# eval expanded noisy testset with composite metrics
# NOISY_TEST_PATH = 'data/expanded_segan1_additive/noisy_testset'

def main(opts):
    NOISY_TEST_PATH = opts.test_wavs
    CLEAN_TEST_PATH = opts.clean_wavs

    noisy_wavs = glob.glob(os.path.join(NOISY_TEST_PATH, '*.wav'))
    metrics = {'csig': [], 'cbak': [], 'covl': [], 'pesq':[], 'ssnr':[]}
    timings = []
    # out_log = open('eval_noisy.log', 'w')
    out_log = open(opts.logfile, 'w')
    out_log.write('FILE CSIG CBAK COVL PESQ SSNR\n')
    for n_i, noisy_wav in enumerate(noisy_wavs, start=1):
        bname = os.path.splitext(os.path.basename(noisy_wav))[0]
        clean_wav = os.path.join(CLEAN_TEST_PATH, bname + '.wav')
        noisy, rate = librosa.load(noisy_wav, 16000)
        clean, rate = librosa.load(clean_wav, 16000)
        # rate, noisy = wavfile.read(noisy_wav)
        # rate, clean = wavfile.read(clean_wav)
        beg_t = timeit.default_timer()
        csig, cbak, covl, pesq, ssnr = CompositeEval(clean, noisy, True)
        # if nan, assign 0
        if np.isnan(csig):
            csig = 0
        if np.isnan(cbak):
            cbak = 0
        if np.isnan(covl):
            covl = 0
        if np.isnan(pesq):
            pesq = 0
        if np.isnan(ssnr):
            ssnr = 0
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        metrics['csig'].append(csig)
        metrics['cbak'].append(cbak)
        metrics['covl'].append(covl)
        metrics['pesq'].append(pesq)
        metrics['ssnr'].append(ssnr)
        out_log.write('{} '.format(bname + '.wav') + ' '*(len(bname)-4) +
                      '{:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3}\n'.format(csig, cbak, covl,
                                                                      pesq, ssnr))
        print('Processed {}/{} wav, CSIG:{:.3f} CBAK:{:.3f} COVL:{:.3f} '
              'PESQ:{:.3f} SSNR:{:.3f} '
              'total time: {:.2f} seconds, mproc: {:.2f}'
              ' seconds'.format(n_i, len(noisy_wavs), csig, cbak, covl,
                                pesq, ssnr,
                                np.sum(timings), np.mean(timings)))

    csig_mean = np.mean(metrics['csig'])
    cbak_mean = np.mean(metrics['cbak'])
    covl_mean = np.mean(metrics['covl'])
    pesq_mean = np.mean(metrics['pesq'])
    ssnr_mean = np.mean(metrics['ssnr'])

    print('mean Csig: ', csig_mean)
    print('mean Cbak: ', cbak_mean)
    print('mean Covl: ', covl_mean)
    print('mean pesq: ', pesq_mean)
    print('mean ssnr: ', ssnr_mean)

    out_log.write('\n' + '-*'*11 + ' Mean of statics ' + '-*'*11 + '\n')
    out_log.write('{}\n{:.3f} {:.3f} {:.3f} {:.3f} {:.3}\n\n\n'.format('CSIG  CBAK  COVL  PESQ  SSNR',
                                                                  csig_mean, cbak_mean, covl_mean,
                                                                  pesq_mean, ssnr_mean))
    out_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_wavs', type=str, default="/home/selcuk/.pytorch/DS_10283_2791/noisy_testset_wav/")  # None
    parser.add_argument('--clean_wavs', type=str, default="/home/selcuk/Desktop/All Thesis Results/segan+/segan+_pretrained_samples/")  # None
    parser.add_argument('--logfile', type=str, default="statistics_wsegan_samples.txt")

    opts = parser.parse_args()

    assert opts.test_wavs is not None
    assert opts.clean_wavs is not None
    assert opts.logfile is not None

    # load opts
    with open('wsegan_samples/clean.opts', 'r') as cfg_f:
        clean_opts = json.load(cfg_f, object_hook=lambda d: namedtuple('Namespace', d.keys())(*d.values()))

    main(opts)
