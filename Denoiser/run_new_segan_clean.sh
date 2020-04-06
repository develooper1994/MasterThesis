#!/bin/bash
### TODO: !!! Still Problematic !!!

CKPT_PATH="/home/selcukcaglar08/MasterThesis/Denoiser/ckpt_new_segan"
CKPT_SINC_PATH="/home/selcukcaglar08/MasterThesis/Denoiser/ckpt_new_segan_sinc"
CKPT_WSEGAN_PATH="/home/selcuk/PycharmProjects/MasterThesis/Denoiser/ckpt_wsegan_misalign/"

# please specify the path to your G model checkpoint
# as in weights_G-EOE_<iter>.ckpt
G_PRETRAINED_CKPT="new_segan_generator.ckpt"
G_PRETRAINED_CKPT="weights_EOE_G-Generator-56101.ckpt"
G_PRETRAINED_WSEGAN_CKPT="weights_EOE_G-Generator-130680.ckpt"

# please specify the path to your folder containing
# noisy test files, each wav in there will be processed
#TEST_FILES_PATH="data_veu4/expanded_segan1_additive/noisy_testset/"
TEST_FILES_PATH=$noisy_test_wav

# please specify the output folder where cleaned files
# will be saved
SAVE_PATH="synth_segan_sinc+"

#python3 -u clean.py --g_pretrained_ckpt $CKPT_PATH/$G_PRETRAINED_CKPT \
#                    --test_files $TEST_FILES_PATH  \
#                    --cfg_file $CKPT_PATH/train.opts \
#                    --synthesis_path $SAVE_PATH --soundfile


python3 -u clean.py --g_pretrained_ckpt $CKPT_SINC_PATH/$G_PRETRAINED_CKPT \
                    --test_files $TEST_FILES_PATH \
                    --cfg_file $CKPT_SINC_PATH/train.opts \
                    --synthesis_path $SAVE_PATH --soundfile

#python3 -u clean.py $CKPT_SINC_PATH/$G_PRETRAINED_CKPT $TEST_FILES_PATH False 2020 segan_samples False False train.opts