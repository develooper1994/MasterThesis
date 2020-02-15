#!/bin/bash


#python -u train.py --save_path ckpt_segan+ \
#	--clean_trainset data_veu4/expanded_segan1_additive/clean_trainset \
#	--noisy_trainset data_veu4/expanded_segan1_additive/noisy_trainset \
#	--cache_dir data_tmp --no_train_gen --batch_size 300 --no_bias


python -u train.py --save_path ckpt_segan+ \
	--clean_trainset /home/selcuk/.pytorch/DS_10283_2791/clean_trainset_56spk_wav/ \
	--noisy_trainset /home/selcuk/.pytorch/DS_10283_2791/noisy_trainset_56spk_wav/ \
	--cache_dir data_cache --no_train_gen --batch_size 300 --no_bias