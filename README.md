# This repository is not a full package yet.

# ! TODO: Title
PyTorch implementation of [!TODO: link]()

Before running, make sure you have the `sc09` dataset, and put that dataset into 
config.py DATASET_NAME variable with full path and filename

If you want to output different directory, change config.py OUTPUT_PATH variable with full path and filename

## Quick Start:
1.Installation
```
sudo apt-get install libav-tools
```

2.Download dataset
* `sc09`: [sc09 raw WAV files](http://deepyeti.ucsd.edu/cdonahue/sc09.tar.gz), utterances of spoken english words '0'-'9'
* `piano`: [Piano raw WAV files](http://deepyeti.ucsd.edu/cdonahue/mancini_piano.tar.gz)

3.Run

For `sc09` task, **make sure `sc09` dataset under your current project filepath befor run your code.**
```
$ python train.py
```

#### Training time
* For `SC09` dataset, <don't know yet> takes nearly <don't know yet> to get reasonable result.
* For `piano` piano dataset, <don't know yet> takes <don't know yet> to get reasonable result.
* Increase the `BATCH_SIZE` from 10 to 32 or 64 can acquire shorter per-epoch time on multiple-GPU but slower gradient descent learning rate.

## Results
Generated  ! TODO: soundcloud

Generated  ! TODO: soundcloud

## Contributions
! TODO