#!/bin/bash


#python3 -u train.py --save_path ckpt_new_segan \
#                     --clean_valset $clean_test_wav\
#                     --noisy_valset $noisy_test_wav \

# --g_pretrained_ckpt $$pretrained is experimental to speed up training
python3 -u eval_noisy_performance.py \
        --clean_valset $clean_test_wav\
        --noisy_valset $noisy_test_wav \
        --logfile "log.txt"