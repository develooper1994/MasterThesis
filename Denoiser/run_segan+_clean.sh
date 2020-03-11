#!/bin/bash

CKPT_PATH="/home/selcukcaglar08/MasterThesis/Denoiser/ckpt_segan+"

# please specify the path to your G model checkpoint
# as in weights_G-EOE_<iter>.ckpt
G_PRETRAINED_CKPT="segan+_generator.ckpt"

# please specify the path to your folder containing
# noisy test files, each wav in there will be processed
#TEST_FILES_PATH="data_veu4/expanded_segan1_additive/noisy_testset/"
TEST_FILES_PATH=$noisy_test_wav

# please specify the output folder where cleaned files
# will be saved
SAVE_PATH="synth_segan_sinc+"

python3 -u clean.py --g_pretrained_ckpt $CKPT_PATH/$G_PRETRAINED_CKPT \
                    --test_files $TEST_FILES_PATH  \
                    --cfg_file $CKPT_PATH/train.opts \
                    --synthesis_path $SAVE_PATH --soundfile

python3 -u clean.py --g_pretrained_ckpt ckpt_segan_sinc+/weights_EOE_G-Generator-56101.ckpt \
                    --test_files /home/selcukcaglar08/full_audio_dataset/DS_10283_2791/noisy_testset_wav \
                    --cfg_file ckpt_segan_sinc+/train.opts \
                    --synthesis_path $SAVE_PATH \
                    --soundfile