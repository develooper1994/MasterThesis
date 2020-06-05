#!/bin/bash
### TODO: !!! Still Problematic !!!

#python3 -u train.py --save_path ckpt_wsegan_misalign \
#        --clean_trainset data_veu4/silent/clean_trainset_M4 \
#        --noisy_trainset data_veu4/silent/whisper_trainset_M4 \
#        --cache_dir data_silent_cache --no_train_gen --batch_size 150  \
#        --wsegan --gnorm_type snorm --dnorm_type snorm --opt adam \
#        --data_stride 0.05 --misalign_pair

# --batch_size 150
# --clean_valset $clean_test_wav\
# --noisy_valset $noisy_test_wav \
python3 -u train.py --save_path ckpt_aewsegan_misalign \
        --clean_trainset $clean_train56spk_wav \
        --noisy_trainset $noisy_train56spk_wav \
        --cache_dir data_silent_cache --no_train_gen --batch_size 200  \
        --aewsegan --gnorm_type snorm --dnorm_type snorm --opt adam \
        --data_stride 0.05 --misalign_pair