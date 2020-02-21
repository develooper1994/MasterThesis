import argparse
import json
import os
from collections import namedtuple

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default="seganv1_ckpt",  # seganv1_ckpt
                    help="Path to save models (Def: seganv1_ckpt).")
parser.add_argument('--d_pretrained_ckpt', type=str, default=None,
                    help='Path to ckpt file to pre-load in training (Def: None).')
parser.add_argument('--g_pretrained_ckpt', type=str, default=None,
                    help='Path to ckpt file to pre-load in training (Def: None).')
parser.add_argument('--cache_dir', type=str, default='data_cache')
parser.add_argument('--clean_trainset', type=str,
                    default='/home/selcuk/.pytorch/DS_10283_2791/clean_trainset_56spk_wav/')
parser.add_argument('--noisy_trainset', type=str,
                    default='/home/selcuk/.pytorch/DS_10283_2791/noisy_trainset_56spk_wav/')
parser.add_argument('--clean_valset', type=str,
                    default=None)  # '/home/selcuk/.pytorch/DS_10283_2791/clean_testset_wav/')
parser.add_argument('--noisy_valset', type=str,
                    default=None)  # '/home/selcuk/.pytorch/DS_10283_2791/noisy_testset_wav/')
parser.add_argument('--h5_data_root', type=str, default=None,
                    help='H5 data root dir (Def: None). The files will be found by split name {train, valid, test}.h5')
parser.add_argument('--h5', action='store_true', default=False,
                    help='Activate H5 dataset mode (Def: False).')
parser.add_argument('--data_stride', type=float, default=0.5,
                    help='Stride in seconds for data read')
parser.add_argument('--seed', type=int, default=2020,  # 111
                    help="Random seed (Def: 111).")
parser.add_argument('--epoch', type=int, default=100)  # 100
parser.add_argument('--patience', type=int, default=100,
                    help='If validation path is set, there are denoising evaluations running for which '
                         'COVL, CSIG, CBAK, PESQ and SSNR are computed. Patience is number of validation '
                         'epochs to wait til breakining train loop. This is an unstable and slow process though, so we'
                         'avoid patience by setting it high atm (Def: 100).')
parser.add_argument('--batch_size', type=int, default=136)  # 136  # 130  # 115  # 100  # 50
parser.add_argument('--save_freq', type=int, default=50,
                    help="Batch save freq (Def: 50).")
parser.add_argument('--slice_size', type=int, default=16384)
parser.add_argument('--opt', type=str, default='adam')  # rmsprop
parser.add_argument('--l1_dec_epoch', type=int, default=100)
parser.add_argument('--l1_weight', type=float, default=100,
                    help='L1 regularization weight (Def. 100). ')
parser.add_argument('--l1_dec_step', type=float, default=1e-5,
                    help='L1 regularization decay factor by batch (Def: 1e-5).')
parser.add_argument('--g_lr', type=float, default=0.00005,
                    help='Generator learning rate (Def: 0.00005).')
parser.add_argument('--d_lr', type=float, default=0.00005,
                    help='Discriminator learning rate (Def: 0.0005).')
parser.add_argument('--preemph', type=float, default=0.95,
                    help='Wav preemphasis factor (Def: 0.95).')
parser.add_argument('--max_samples', type=int, default=None,
                    help='Max num of samples to train (Def: None).')
parser.add_argument('--eval_workers', type=int, default=2)  # 2
parser.add_argument('--slice_workers', type=int, default=1)  # 4
parser.add_argument('--num_workers', type=int, default=1,  # 4
                    help='DataLoader number of workers (Def: 1).')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA even if device is available')
parser.add_argument('--random_scale', type=float, nargs='+', default=[1],
                    help='Apply randomly a scaling factor in list to the (clean, noisy) pair')
parser.add_argument('--no_train_gen', action='store_true', default=False,
                    help='Do NOT generate wav samples during training')
parser.add_argument('--preemph_norm', action='store_true', default=False,
                    help='Inverts old  norm + preemph order in data loading, so denorm has to respect this aswell')
parser.add_argument('--wsegan', action='store_true', default=False)  # False
parser.add_argument('--aewsegan', action='store_true', default=False)
parser.add_argument('--vanilla_gan', action='store_true', default=False)
parser.add_argument('--no_bias', action='store_true', default=False,
                    help='Disable all biases in Generator')
parser.add_argument('--n_fft', type=int, default=2048)
parser.add_argument('--reg_loss', type=str, default='l1_loss',
                    help='Regression loss (l1_loss or mse_loss) in the output of G (Def: l1_loss)')

# Skip connections options for G
parser.add_argument('--skip_merge', type=int, default=0,
                    help='merging mode to connect skip connections (Def: 0)'
                    '0) concat'
                    '1) sum')
parser.add_argument('--skip_type', type=int, default=0,
                    help='Type of skip connection: (Def: 0)\n' \
                         '0) alpha: learn a vector of channels to multiply elementwise. \n' \
                         '1) conv: learn conv kernels of size 11 to learn complex responses in the shuttle.\n' \
                         '2) constant: with alpha value, set values to not learnable, just fixed.\n(Def: alpha)')
parser.add_argument('--skip_init', type=int, default=1,
                    help='Way to init skip connections (Def: 1). 0: zero | 1: one | 2: randn')
parser.add_argument('--skip_kwidth', type=int, default=11)

# Generator parameters
parser.add_argument('--gkwidth', type=int, default=31)
parser.add_argument('--genc_fmaps', type=int, nargs='+', default=[64, 128, 256, 512, 1024],
                    help='Number of G encoder feature maps, (Def: [64, 128, 256, 512, 1024]).')
parser.add_argument('--genc_poolings', type=int, nargs='+', default=[4, 4, 4, 4, 4],
                    help='G encoder poolings')
parser.add_argument('--z_dim', type=int, default=1024)
parser.add_argument('--gdec_fmaps', type=int, nargs='+', default=None)
parser.add_argument('--gdec_poolings', type=int, nargs='+', default=None,
                    help='Optional dec poolings. Defaults to None so that encoder poolings are mirrored.')
parser.add_argument('--gdec_kwidth', type=int,
                    default=None)
parser.add_argument('--gnorm_type', type=str, default=None,
                    help='Normalization to be used in G. Can be: (0) none, (1) snorm, (2) bnorm (Def: None).')
parser.add_argument('--no_z', action='store_true', default=False)
parser.add_argument('--no_skip', action='store_true', default=False)
parser.add_argument('--pow_weight', type=float, default=0.001)
parser.add_argument('--misalign_pair', action='store_true', default=False)
parser.add_argument('--interf_pair', action='store_true', default=False)

# Discriminator parameters
parser.add_argument('--denc_fmaps', type=int, nargs='+', default=[64, 128, 256, 512, 1024],
                    help='Number of D encoder feature maps, (Def: [64, 128, 256, 512, 1024]')
parser.add_argument('--dpool_type', type=str, default='none',
                    help='conv/none/gmax/gavg (Def: none)')
parser.add_argument('--dpool_slen', type=int, default=16,
                    help='Dimension of last conv D layer time axis prior to classifier real/fake (Def: 16)')
parser.add_argument('--dkwidth', type=int, default=None,
                    help='Disc kwidth (Def: None), None is gkwidth.')
parser.add_argument('--denc_poolings', type=int, nargs='+', default=[4, 4, 4, 4, 4],
                    help='(Def: [4, 4, 4, 4, 4])')
parser.add_argument('--dnorm_type', type=str, default='vnorm',  # bnorm
                    help='Normalization to be used in D. Can be: (0) none, (1) snorm, (2) bnorm, (3) vnorm (Def: bnorm).')
parser.add_argument('--phase_shift', type=int, default=5)
parser.add_argument('--sinc_conv', action='store_true', default=True)  # False
parser.add_argument('--windowing', type=str, default='hamming',  # hamming
                    help='Windowing to be used with Sinc_conv Can be: (1) rectangular, (2) triangular, (3) hann or '
                    '(4) hann (5) hamming (6) blackman (7) nuttall (8) blackman_nuttall '
                    '(9) blackman_harris (10) flat_top  (Def: hamming).')

opts = parser.parse_args()
opts.bias = not opts.no_bias

if not os.path.exists(opts.save_path):
    os.makedirs(opts.save_path)

# save opts
with open(os.path.join(opts.save_path, 'train.opts'), 'w') as cfg_f:
    cfg_f.write(json.dumps(vars(opts), indent=2))
print('Parsed arguments: ', json.dumps(vars(opts), indent=2))

# # load opts
# with open(os.path.join(opts.save_path, 'train.opts'), 'r') as cfg_f:
#     opts = json.load(cfg_f, object_hook=lambda d: namedtuple('Namespace', d.keys())(*d.values()))

device = torch.device("cuda" if torch.cuda.is_available() and not opts.no_cuda else "cpu")
# device = torch.device("cpu")
print(f"Training device: {device}")
