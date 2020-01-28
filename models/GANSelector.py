# TODO: It is an abstaction layer for different types of GAN
# standart module imports
import datetime
import time
from typing import NoReturn

# classical machine learning imports

# other 3rd party libraries

# my modules
import torch
import torchaudio
from torch.utils.tensorboard import SummaryWriter

from models.DefaultTrainBuilder import RunBuilder, DefaultRunManager, DefaultTrainBuilder
# all trainers
from models.Trainers.DefaultTrainer import DefaultTrainer, epochs
# collect pair to "architectures" and import all models from "architectures"
from models.architectures.WaveGAN import WaveGAN


# utilities for all architectures
from models.utils.BasicUtils import numpy_to_var, device, torch_image_to_numpy_image, rgb2gray


class RunManager(DefaultRunManager):
    def __init__(self, GAN, loader):
        super(RunManager, self).__init__(GAN, loader)

    @torch.no_grad()
    def begin_run(self, run):
        """
        Configures and gives a start the one experiment.
        @param run: Represents the one experiment. Information comes from RunBuilder class.
        @param network: Pytorch Your neural network class
        @param loader: Pytorch dataloader.
        @return: None
        :param GAN: Generative Adverserial Network inside of architecture folder
        """
        self.run.start_time = time.time()

        self.run.params = run
        self.run.count += 1

        # TODO: Implment Tensorboard visualization in  models.utils.visualization
        # one batch data
        # TODO: label info not completed
        waveforms_labels = next(self.loader)  # next(iter(self.loader))
        # waveforms, labels = numpy_to_var(waveforms_labels['X']), waveforms_labels['Y']
        waveforms, labels = torch.from_numpy(waveforms_labels['X']).to(device), waveforms_labels['Y']
        # torch.set_default_tensor_type('torch.cuda.FloatTensor') # fixes "RuntimeError: expected device cuda:0 but got device cpu" error
        # specgram = torchaudio.transforms.Spectrogram()(torchaudio.transforms.AmplitudeToDB()(waveforms[0].cpu()))  # I don't it will write with iterator
        specgram = torchaudio.transforms.Spectrogram()(waveforms[0].cpu())  # I don't it will write with iterator
        # grid = torchvision.utils.make_grid(specgrams)

        # # Tensorboard configuration
        self.tb = SummaryWriter(comment=f'-{run}')
        self.tb.add_image('sample image', specgram.unsqueeze(0).permute((0, 2, 1)))
        # TODO: !!!RuntimeError: hasSpecialCase INTERNAL ASSERT FAILED at /opt/conda/conda-bld/pytorch_1580112455885/work/torch/csrc/jit/passes/alias_analysis.cpp:299, please report a bug to PyTorch. We don't have an op for aten::to but it isn't a special case. (analyzeImpl at /opt/conda/conda-bld/pytorch_1580112455885/work/torch/csrc/jit/passes/alias_analysis.cpp:299)
        # TODO: I couldn't solve it. Almost everyone has this issue and it haven't solve yet.
        # self.tb.add_graph(self.D, waveforms.unsqueeze(1), True)  # waveforms.unsqueeze(1)
        # self.tb.add_graph(self.G, waveforms.unsqueeze(1))  # waveforms.unsqueeze(1)

    def end_run(self):
        super(RunManager, self).end_run()

    def begin_epoch(self) -> NoReturn:
        super(RunManager, self).begin_epoch()

    def end_epoch(self):
        super(RunManager, self).end_epoch()

    def track_loss(self, loss):
        super(RunManager, self).track_loss(loss)

    def track_num_correct(self, preds, labels):
        super(RunManager, self).track_num_correct(preds, labels)

    def _get_num_correct(self, preds, labels):
        super(RunManager, self)._get_num_correct(preds, labels)

    def save(self, filename):
        super(RunManager, self).save(filename)


class GANSelector(DefaultTrainBuilder):
    m: DefaultRunManager

    # def __init__(self, netD, netG) -> None:
    def __init__(self, GAN, data_loader, epochs=1) -> None:
        super().__init__()
        global gan_type
        self.epochs = epochs
        self.data_loader = data_loader
        # self.BATCH_NUM, self.train_iter, self.valid_iter, self.test_iter = self.dataloader.split_manage_data(arguments,
        #                                                                                                   batch_size)

        # select GAN
        if GAN == "wavegan":
            gan_type = WaveGAN()
            self.netD, self.netG = gan_type.netD, gan_type.netG  # WaveGANDiscriminator, WaveGANGenerator
            self.optimizerD, self.optimizerG = gan_type.optimizerD, gan_type.optimizerG  # wave_gan_utils.optimizers(arguments)

        # TODO add "segan" and "segan+(plus)" options
        else:
            print("I don't know your GAN. Make your own")
        self.set_nets(self.netD, self.netG)
        self.base_trainer = DefaultTrainer(self.netG, self.netD, self.optimizerG, self.optimizerD, gan_type,
                                           self.data_loader)
        self.train_iter = self.base_trainer.train_iter
        self.valid_iter = self.base_trainer.valid_iter
        self.test_iter = self.base_trainer.test_iter
        self.dataset = [self.train_iter, self.valid_iter, self.test_iter]

        # self.m = RunManager(gan_type, data_loader)  # m indicates manager
        # don't want to consume train_iter
        self.m = RunManager(gan_type, self.test_iter)  # m indicates manager


    def batches(self, **kwargs) -> NoReturn:
        self.base_trainer.train_gan_one_batch(kwargs)
        self.m.end_epoch()

    def all_epochs(self, loader) -> NoReturn:
        start = time.time()
        for epoch in range(1, epochs + 1):
            self.m.begin_epoch()
            # one batch
            self.base_trainer.train_one_epoch(epoch, start)

    def train(self):
        # self.base_trainer.train()
        self.all_epochs(self.train_iter)

    def experiments(self, train_set, runs) -> NoReturn:
        # start experiments with parameters
        for run in runs:
            ## Experiments Start ##

            self.m.begin_run(run)
            self.all_epochs(self.train_iter)
            ## Experiments End ##
            self.m.end_run()

    def wrap_experiments(self,  params, data_sets, epochs: int, validation=False) -> NoReturn:
        # super(GANSelector, self).wrap_experiments(params, data_sets, epochs, validation=False)
        if validation:
            validation_set = data_sets[1]
            test_set = data_sets[2]
        else:
            test_set = data_sets[1]

        train_set = data_sets[0]
        # get all runs from params using RunBuilder class
        runs = RunBuilder.get_runs(params)
        self.epochs = epochs
        self.number_of_experiments = len(runs) * epochs
        print(f"number of experiments or rows: {self.number_of_experiments}")

        # start experiments with parameters
        self.experiments(train_set, runs)

        # when all runs are done, save results to files
        time = datetime.datetime.now()
        # TODO! add time stamp like that. datetime.datetime.now().strftime("%c"). !OSError: [Errno 22] Invalid argument!
        self.m.save('results')
        print("End of all training experiments")

    def train_experiments(self, params):
        self.wrap_experiments(params=params, data_sets=self.dataset, epochs=self.epochs, validation=False)

    def set_discriminator(self, netD):
        self.netD = netD

    def get_discriminator(self):
        return self.netD

    def set_generator(self, netG):
        self.netG = netG

    def get_generator(self):
        return self.netG

    def set_nets(self, netD, netG):
        self.set_generator(netD)
        self.set_generator(netG)

    def get_nets(self):
        return self.get_discriminator(), self.get_generator()
