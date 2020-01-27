from collections import OrderedDict

# EmTraining
EPOCHS = 1  # 1 # 10 # 180
# NOTE!!!: Bigger BATCH_SIZE; faster training and slower gradient degradation
BATCH_SIZE = 150  # 193 # 190 # 175 # 150 # 100 # 80 # 64 # 10

# signal processing configurations.
# NOTE: DON'T CHANGE IF YOU DON'T REALLLLYYYY NEED
WINDOW_LENGHT = 16384
FS = 16000

SAMPLE_EVERY = 1  # Generate audio samples every 1 epoch.
# it is also input size of discriminator
SAMPLE_NUM = int(FS/16)  # Generate 10 samples every sample generation.

# Data
DATASET_NAME = "/home/selcuk/.pytorch/sc09/"  # 'sc09/'
OUTPUT_PATH = "output/"

# Model(Network)
MODEL = "wavegan"
# if you want to change optimizers look at models.optimizers.BaseOptimizer.py



# experiments
params = OrderedDict(
        # epochs=[EPOCHS],
        lr=[.01, .001],
        batch_size=[BATCH_SIZE, 100, 1000],
        shuffle=[True, False],
        # dataset_name=[DATASET_NAME],
        # output_name=[OUTPUT_PATH],
        # window_lenght=[WINDOW_LENGHT],
        # sample_every=[SAMPLE_EVERY],
        # sample_num=[SAMPLE_NUM],
        # Model(Network)
        MODEL=["wavegan"],
    )