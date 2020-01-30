# standart module imports
import datetime
import json
import os
import time
from collections import OrderedDict, namedtuple
from itertools import product  # cartesian product
from typing import List, Any, NoReturn, Union

import pandas as pd
# classical machine learning imports
import torch
import torchaudio
from IPython.display import display, clear_output
from pandas import DataFrame
from torch.utils.tensorboard import SummaryWriter

# my modules
from config import target_signals_dir, batch_size, generator_batch_size_factor, device
from models.DataLoader.AudioDataset import AudioDataset
from models.DataLoader.DataLoader import WavDataLoader
from models.Trainers.DefaultTrainer import DefaultTrainer
from models.Trainers.DefaultTrainer import audio_dir, output_dir
from models.Trainers.WaveganTrainer import WaveGan_GP
from models.architectures.WaveGAN import WaveGAN
from models.utils.BasicUtils import visualize_loss, latent_space_interpolation, sample_noise


# other 3rd party libraries


class RunBuilder:
    """
    Takes all parameters with variable names and values and insert into a list
    """

    @staticmethod
    def get_runs(params) -> List[Any]:
        """
        Takes parameters and makes first it dictionary.
        Second appends all values or keys of variable in to a list.
        @param params: takes an OrderedDict to try all experiments in one loop and also to show results on any report.
        @return: list of values or keys of variable

        """
        Run = namedtuple('Run', params.keys())

        return [Run(*v) for v in product(*params.values())]


class Epoch:
    """
    Refactored epoch variables
    """
    count: int
    loss: int
    num_correct: int

    def __init__(self, count=0, loss=0, num_correct=0, start_time=0):
        self.count = count
        self.loss = loss
        self.num_correct = num_correct
        self.start_time = start_time


class Run:
    """
    Refactored run variables
    """
    count: int
    data: List[Any]

    def __init__(self, params: Union[None, None, None] = None, count: int = 0, data=None,
                 start_time: Union[None, None, None] = None) -> None:
        if data is None:
            data = []
        self.params = params
        self.count = count
        self.data = data
        self.start_time = start_time


class DefaultRunManager:
    """
    Controls all learning process functions and utilities like visualization and taking checkpoint
    """
    run: Run

    # TODO! make a evaluation method with testloader
    # TODO! split dataset into train, validation(dev) and test
    def __init__(self, discriminator, generator, loader):
        # tracking every epoch count, loss, accuracy, time

        self.netG = generator
        self.netD = discriminator
        self.epoch = Epoch()

        # tracking every run count, run data, hyper-params used, time
        self.run = Run()

        # record model, loader and TensorBoard
        # self.network = None
        self.loader = loader
        self.tb = None

    # record the count, hyper-param, model, loader of each run
    # record sample images and network graph to TensorBoard
    # TODO! refactor begin_run and end_run functions
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
        waveforms, labels = next(iter(self.loader))
        specgrams = torchaudio.transforms.Spectrogram()(waveforms.cpu())  # I don't it will write with iterator
        # grid = torchvision.utils.make_grid(specgrams)

        # # Tensorboard configuration
        self.tb = SummaryWriter(comment=f'-{run}')  # MOST TIME CONSUMING PART. don't remove or change.
        self.tb.add_image('images', specgrams[0])
        # TODO: RuntimeError: size mismatch,
        self.tb.add_graph(self.netD, waveforms)
        # (noise_latent_dim, 4 * 4 * model_size * self.dim_mul)
        fix_noise = sample_noise(batch_size * generator_batch_size_factor).to(device)
        self.tb.add_graph(self.netG, fix_noise)

    # when run ends, close TensorBoard, zero epoch count
    def end_run(self):
        """
        Takes nothing
        Concludes one experiment and closes Tensorboard session.
        @return: None
        """
        self.tb.flush()
        self.tb.close()
        self.epoch.count = 0
        visualize_loss(self.base_trainer.g_cost, self.base_trainer.valid_g_cost, 'Train', 'Val', 'Negative Critic Loss')
        latent_space_interpolation(self.base_trainer.generator, n_samples=5)

    # zero epoch count, loss, accuracy,
    # TODO! refactor begin_epoch and end_epoch functions
    # @torch.no_grad()
    def begin_epoch(self) -> NoReturn:
        """
        Takes nothing
        Configures and gives a start the one epoch experiment.
        @return: None
        """
        self.epoch.start_time = time.time()

        self.epoch.count += 1
        self.epoch.loss = 0
        self.epoch.num_correct = 0

    def end_epoch(self):
        """
        Takes nothing
        1) Measures taken time to complete.
        2) Calculates loss and accuracy for each epoch.
        3) Adds scalar plots to Tensorboard.
        4) Makes results and pandas frame and print in nice way.

        Concludes one experiment.
        @return: None
        """
        # calculate epoch duration and run duration(accumulate)
        epoch_duration = time.time() - self.epoch.start_time
        run_duration = time.time() - self.run.start_time

        # record epoch loss and accuracy
        loss = self.epoch.loss / len(self.loader.dataset)
        accuracy = self.epoch.num_correct / len(self.loader.dataset)

        # Record epoch loss and accuracy to TensorBoard
        self.tb.add_scalar('Loss', loss, self.epoch.count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch.count)

        # Record discriminator params to TensorBoard
        for name, param in self.netD.named_parameters():
            self.tb.add_histogram(f'{name} discriminator', param, self.epoch.count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch.count)
        # Record generator params to TensorBoard
        for name, param in self.netG.named_parameters():
            self.tb.add_histogram(f'{name} generator', param, self.epoch.count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch.count)

        # Write into 'results' (OrderedDict) for all run related data
        results = OrderedDict()
        results["run"] = self.run.count
        results["epoch"] = self.epoch.count
        results["loss"] = loss
        results["accuracy"] = accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration

        # Record hyper-params into 'results'
        for k, v in self.run.params.asdict().items(): results[k] = v
        self.run.data.append(results)
        df: DataFrame = pd.DataFrame.from_dict(self.run.data, orient='columns')

        # display epoch information and show progress
        clear_output(wait=True)
        display(df)

    # accumulate loss of batch into entire epoch loss
    # @torch.no_grad()
    def track_loss(self, loss):
        """
        Tracks loss function for loss for each epoch
        @param loss: loss function
        @return: None
        """
        # multiply batch size so variety of batch sizes can be compared
        self.epoch.loss += loss.item() * self.loader.batch_size

    # accumulate number of corrects of batch into entire epoch num_correct
    # @torch.no_grad()
    def track_num_correct(self, preds, labels):
        """
        Takes predictions and labels
        Tracks num correct predictions with respect to actual labels for each epoch
        @param preds: correct predictions
        @type preds: torch.tensor
        @param labels: actual labels
        @return: None
        """
        self.epoch.num_correct += self._get_num_correct(preds, labels)

    # @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        """
        Takes predictions and labels
        Counts num correct predictions with respect to actual labels for each epoch
        @param preds: correct predictions
        @type preds: torch.tensor
        @param labels: actual labels
        @return: number of corrections
        """
        return preds.argmax(dim=1).eq(labels).sum().item()

    # save end results of all runs into csv, json for further analysis
    @torch.no_grad()
    def save(self, filename):
        """
        Creates .csv file with pandas and a .json dump file with json.
        @param filename: a filename string
        @return: None
        """

        pd.DataFrame.from_dict(
            self.run.data,
            orient='columns',
        ).to_csv(f'{filename}.csv')

        with open(f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run.data, f, ensure_ascii=False, indent=4)


class DefaultTrainBuilder:
    number_of_experiments: Union[int, Any]

    def __init__(self, GAN, data_loader, epochs=1) -> NoReturn:
        self.epochs = epochs
        self.GAN = GAN
        self.data_loader = data_loader

        self.number_of_experiments = 0
        self.network = None
        # self.m = DefaultRunManager(self.network)  # m indicates manager
        self.optimizerD, self.optimizerG = None, None
        self.epochs = 1

        print(f"using {GAN} trainer")
        if GAN == "wavegan":
            gan_type: WaveGAN = WaveGAN()
            self.netD, self.netG = gan_type.discriminator, gan_type.generator  # WaveGANDiscriminator, WaveGANGenerator
            self.optimizerD, self.optimizerG = gan_type.optimizerD, gan_type.optimizerG  # wave_gan_utils.optimizers(arguments)
            self.set_nets(self.netD, self.netG)
            self.data_loader = AudioDataset(input_dir=audio_dir, output_dir=output_dir)
            self.base_trainer = DefaultTrainer(self.netG, self.netD, self.optimizerG, self.optimizerD, gan_type,
                                               self.data_loader)
            self.train_iter = self.base_trainer.train_iter
            self.valid_iter = self.base_trainer.valid_iter
            self.test_iter = self.base_trainer.test_iter
            self.dataset = [self.train_iter, self.valid_iter, self.test_iter]
        elif GAN in ["wavegan-gp", "wavegan_gp", "wavegangp"]:
            self.train_iter: WavDataLoader = WavDataLoader(os.path.join(target_signals_dir, 'train'))
            self.valid_iter: WavDataLoader = WavDataLoader(os.path.join(target_signals_dir, 'valid'))
            self.test_iter: WavDataLoader = WavDataLoader(os.path.join(target_signals_dir, 'test'))
            self.dataset = [self.train_iter, self.valid_iter, self.test_iter]

            gan_type: WaveGan_GP = WaveGan_GP(self.train_iter, self.valid_iter)
            self.netD, self.netG = gan_type.discriminator, gan_type.generator
            self.set_nets(self.netD, self.netG)
            self.base_trainer = gan_type
            # self.base_trainer.train()
            # visualize_loss(self.base_trainer.g_cost, self.base_trainer.valid_g_cost, 'Train', 'Val', 'Negative Critic Loss')
            # latent_space_interpolation(self.base_trainer.generator, n_samples=5)

        else:
            print("I don't know your GAN. Make your own")

        self.m = DefaultRunManager(gan_type.discriminator, gan_type.generator, self.test_iter)  # m indicates manager

    # TODO! keras like evaluate function to measure accuracy
    def evaluate(self):
        # evaluate neural network on all test data set.
        pass

    # TODO! keras like compile and train. Complete keras compile
    def compile(self):
        # compile hyperparameters, train, validation and test sets
        pass

    # TODO! keras like compile and train. Complete keras fit
    def fit(self, data_sets, run=RunBuilder.get_runs(OrderedDict(lr=[.01], batch_size=[1000], shuffle=[True]))[0],
            validation=False):
        """
        It is just a simple trainer
        @param data_sets: Pytorch dataloader, train_set, validation_set(dev_set), test_set
            train_set: Pytorch dataloader. dataset for training phase
            validation_set: Pytorch dataloader. dataset for validation phase
            test_set: Pytorch dataloader.  dataset for testing phase

        @param run: hyperparameters
        example: Run(lr=0.01, batch_size=100, shuffle=True)
        default: RunBuilder.get_runs(OrderedDict(lr=[.01], batch_size=[1000], shuffle=[True]))[0]

        @return: None
        :param validation:
        """

        # if validation:
        #     validation_set = data_sets[1]
        #     test_set = data_sets[2]
        # else:
        #     test_set = data_sets[1]
        # train_set = data_sets[0]

        # RunBuilder.get_runs(OrderedDict(lr=[.01], batch_size=[1000], shuffle=[True]))[0]
        # # if params changes, following line of code should reflect the changes too
        # self.network = Network()  # !!!if network hyperparameters changes each time then it should inside of the loop!!!
        # # Just an idea: call __init__ and than __forward__
        # loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size,
        #                                      num_workers=1)  # num_workers=1 is
        # # good for small data
        # self.optimizer = optim.Adam(self.network.parameters(), lr=run.lr)

        # self.m.begin_run(run, self.network, loader)
        # self.all_epochs(loader)
        # self.m.end_run()
        pass

    # # DefaultTrainer train
    # def all_epochs(self, loader) -> NoReturn:
    #     start = time.time()
    #     for epoch in range(1, epochs + 1):
    #         self.m.begin_epoch()
    #         # one batch
    #         if self.GAN == "wavegan":
    #             self.base_trainer.train_one_epoch(epoch, start)
    #         elif self.GAN in ["wavegan-gp", "wavegan_gp", "wavegangp"]:
    #             self.base_trainer.train_one_epoch()
    #         else:
    #             print("I don't know your GAN. Make your own")

    def train(self):
        self.base_trainer.train()
        # self.all_epochs(self.train_iter)

    def experiments(self, runs) -> NoReturn:
        # start experiments with parameters
        for run in runs:
            # if params changes, following line of code should reflect the changes too
            # self.network = Network()  # !!!if network hyperparameters changes each time then it should inside of the loop!!!
            # # Just an idea: call __init__ and than __forward__
            # loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size,
            #                                      num_workers=1)  # num_workers=1 is
            # # good for small data
            # self.optimizer = optim.Adam(self.network.parameters(), lr=run.lr)

            # self.m.begin_run(run, self.network, loader)
            # self.all_epochs(loader)

            ## Experiments Start ##
            self.m.begin_run(run)
            # self.all_epochs(self.train_iter)
            self.base_trainer.train()
            ## Experiments End ##
            self.m.end_run()

    def wrap_experiments(self, params, epochs: int) -> NoReturn:
        """
        Put all to gather
        @param params: hyperparameters for experiment
        @param data_sets:
            data_sets[0] -> Train set
            data_sets[1] -> Validation(dev) set
            data_sets[2] -> Test set
        @param epochs: number of iterations.
        @param validation: If there is a validation change it to True.
            Default False.
        @return: None
        """
        # get all runs from params using RunBuilder class
        runs = RunBuilder.get_runs(params)
        self.epochs = epochs
        self.number_of_experiments = len(runs) * epochs
        print(f"number of experiments or rows: {self.number_of_experiments}")

        # start experiments with parameters
        self.experiments(runs)

        # when all runs are done, save results to files
        time = datetime.datetime.now()
        # TODO! add time stamp like that. datetime.datetime.now().strftime("%c"). !OSError: [Errno 22] Invalid argument!
        self.m.save('results')
        print("End of all training experiments")

    def train_experiments(self, params):
        self.wrap_experiments(params=params, epochs=self.epochs)

    ######################
    # Getters and Stetters
    ######################
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
