#!/bin/bash


#python3 -u train.py --save_path ckpt_segan+ \
#       --clean_trainset data_veu4/expanded_segan1_additive/clean_trainset \
#       --noisy_trainset data_veu4/expanded_segan1_additive/noisy_trainset \
#       --cache_dir data_tmp --no_train_gen --batch_size 300 --no_bias

# --g_pretrained_ckpt $pretrained is experimental to speed up training
# --clean_valset $clean_test_wav\
# --noisy_valset $noisy_test_wav \
python3 -u train.py --save_path ckpt_segan_sinc+ \
        --clean_trainset $clean_train56spk_wav \
        --noisy_trainset $noisy_train56spk_wav \
        --cache_dir data_cache --no_train_gen --batch_size 300 --no_bias