# standart module imports
import datetime
import json
import os
import time
from collections import OrderedDict
from typing import List, Any, NoReturn, Union

import pandas as pd
# classical machine learning imports
import torch
import torchaudio
from IPython.display import display, clear_output
from pandas import DataFrame
from torch.utils.tensorboard import SummaryWriter

# my modules
from config import target_signals_dir, batch_size, generator_batch_size_factor, device, n_iterations, \
    progress_bar_step_iter_size, RunBuilder, Runs
from models.DataLoader.AudioDataset import AudioDataset
from models.DataLoader.DataLoader import WavDataLoader
from models.Trainers.DefaultTrainer import DefaultTrainer
from models.Trainers.DefaultTrainer import audio_dir, output_dir
from models.Trainers.WaveganTrainer import WaveGan_GP
from models.architectures.WaveGAN import WaveGAN
from models.utils.BasicUtils import visualize_loss, latent_space_interpolation, sample_noise


# other 3rd party libraries


class Epoch:
    """
    Refactored epoch variables
    """
    count: int
    loss: int
    num_correct: int

    def __init__(self, count=n_iterations // progress_bar_step_iter_size, loss_list=None, num_correct=0, start_time=0):
        if loss_list is None:
            loss_list = []
        self.count = count
        self.loss_list = loss_list
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


# TODO: Implement for all gan types
class DefaultRunManager:
    """
    Controls all learning process functions and utilities like visualization and taking checkpoint
    """
    run: Run

    # def __init__(self, loader, discriminator, generator):
    def __init__(self, loader, **networks):
        # tracking every epoch count, loss, accuracy, time

        self.base_trainer = None
        self.networks_names = list(networks.keys())
        self.networks = list(networks.values())
        self.numer_networks = len(self.networks)
        self.epoch = Epoch()

        # tracking every run count, run data, hyper-params used, time
        self.run = Run()

        # record model, loader and TensorBoard
        # self.network = None
        self.loader = loader
        self.tb = None

    # record the count, hyper-param, model, loader of each run
    # record sample images and network graph to TensorBoard
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
        waveforms, labels = next(iter(self.loader))
        specgrams = torchaudio.transforms.Spectrogram()(waveforms.cpu())  # I don't it will write with iterator
        # # Tensorboard configuration
        tensorboard_file_name = f"target_signals_dir:{target_signals_dir}_n_iterations:{run.n_iterations}" \
        f"_lr_d:{run.lr_d}_lr_g:{run.lr_g}_beta1:{run.beta1}_beta2:{run.beta2}_decay_lr:{run.decay_lr}" \
        f"_n_critic:{run.n_critic}_p_coeff:{run.p_coeff}_noise_latent_dim:{run.noise_latent_dim}" \
        f"_model_capacity_size:{run.model_capacity_size}"  # Tensorboard throws File name too long for full name.
        self.tb = SummaryWriter(comment=f'-{tensorboard_file_name}')  # MOST TIME CONSUMING PART. don't remove or change.
        self.tb.add_image('images', specgrams[0])
        # TODO: Second graph overrides.
        fix_noise = sample_noise(batch_size * generator_batch_size_factor).to(device)
        for net_name, net in zip(self.networks_names, self.networks):
            graphs_input = fix_noise if net_name is 'generator' else waveforms
            self.tb.add_graph(net.module, graphs_input)  # jit can't trace dataparallel

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
        visualize_loss(y_label='generator', legends=['Train', 'Val'],
                       losses=[self.base_trainer.g_cost, self.base_trainer.valid_g_cost])
        latent_space_interpolation(self.base_trainer.generator, n_samples=5)

    # zero epoch count, loss, accuracy,
    # TODO! refactor begin_epoch and end_epoch functions
    @torch.no_grad()
    def begin_epoch(self) -> None:
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
        # loss = self.epoch.loss / len(self.loader.dataset)
        # lossD = self.epoch.loss / len(self.loader)
        # lossG = self.epoch.lossG / len(self.loader)
        #
        # # Record epoch loss and accuracy to TensorBoard
        # self.tb.add_scalar('Loss_D', lossD, self.epoch.count)
        # self.tb.add_scalar('Loss_G', lossG, self.epoch.count)
        for count, loss in enumerate(self.epoch.loss_list):
            loss = loss / len(self.loader)

            # Record epoch loss and accuracy to TensorBoard
            self.tb.add_scalar(f'Loss_{self.networks[count]}', loss, self.epoch.count)  # self.networks_names[count]...

        # accuracy = self.epoch.num_correct / len(self.loader.dataset)
        accuracy = self.epoch.num_correct / len(self.loader)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch.count)

        # Record discriminator and generator or something else params to TensorBoard
        self.histogram_tensorboard(self.networks, self.tb)

        # Write into 'results' (OrderedDict) for all run related data
        results: OrderedDict[Union[str, Any], Union[Union[int, float], Any]] = OrderedDict()
        results["run"] = self.run.count
        results["epoch"] = self.epoch.count
        results["loss"] = loss
        results["accuracy"] = accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration

        # Record hyper-params into 'results'
        for k, v in self.run.params._asdict().items(): results[k] = v
        self.run.data.append(results)
        # no need to display data
        df: DataFrame = pd.DataFrame.from_dict(self.run.data, orient='columns')

        # display epoch information and show progress
        clear_output(wait=True)
        display(df)

    def histogram_tensorboard(self, network, tensorboard_sessions) -> None:
        """
        Projects histogram to the tensorboard.
        :param tensorboard_sessions: Logging Tensorboard session object.
        :param network: Articial Neural Network created by Pytorch.
        :return: None
        """
        for net_name, net in zip(self.networks_names, self.networks):
            for name, param in net.named_parameters():
                tensorboard_sessions.add_histogram(f'{name} {net_name}', param, self.epoch.count)
                if param.grad is None:
                    tensorboard_sessions.add_histogram(f'{name}.grad', 0, self.epoch.count)
                    Warning(f"!!! {network} Zero gradient occured. Caution for UNDERFITTING !!!")
                else:
                    tensorboard_sessions.add_histogram(f'{name}.grad', param.grad, self.epoch.count)

    # accumulate loss of batch into entire epoch loss
    @torch.no_grad()
    def track_loss(self, loss):
        """
        Tracks loss function for loss for each epoch
        @param loss: loss function
        @return: None
        """
        # multiply batch size so variety of batch sizes can be compared
        self.epoch.loss += loss.item() * self.loader.batch_size

    # accumulate number of corrects of batch into entire epoch num_correct
    @torch.no_grad()
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

    @torch.no_grad()
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

    def __init__(self, GAN: str, data_loader, epochs=1) -> None:
        """

        :type GAN: str
        """
        self.epochs = epochs
        self.GAN = GAN
        self.data_loader = data_loader

        self.number_of_experiments = 0
        self.network = None
        # self.m = DefaultRunManager(self.network)  # m indicates manager
        self._discriminator, self._generator = None, None
        self.optimizerD, self.optimizerG = None, None

        GAN = GAN.lower()
        print(f"using {GAN} trainer")
        # if GAN == "wavegan":
        #     gan_type: WaveGAN = WaveGAN()
        #     self.discriminator, self.generator = gan_type.discriminator, gan_type.generator  # WaveGANDiscriminator, WaveGANGenerator
        #     self.optimizerD, self.optimizerG = gan_type.optimizerD, gan_type.optimizerG  # wave_gan_utils.optimizers(arguments)
        #     # self.set_nets(self.discriminator, self.generator)
        #     self.data_loader = AudioDataset(input_dir=audio_dir, output_dir=output_dir)
        #     self.base_trainer = DefaultTrainer(self.generator, self.discriminator, self.optimizerG, self.optimizerD, gan_type,
        #                                        self.data_loader)
        #     manager = DefaultRunManager(self.test_iter, discriminator=gan_type.discriminator,
        #                                 generator=gan_type.generator)
        #     gan_type.manager = manager
        #
        #     self.train_iter = self.base_trainer.train_iter
        #     self.valid_iter = self.base_trainer.valid_iter
        #     self.test_iter = self.base_trainer.test_iter
        #     self.dataset = [self.train_iter, self.valid_iter, self.test_iter]
        # elif GAN in ["wavegan-gp", "wavegan_gp", "wavegangp"]:

        if GAN in ["wavegan-gp", "wavegan_gp", "wavegangp"]:
            self.train_iter: WavDataLoader = WavDataLoader(os.path.join(target_signals_dir, 'train'))
            self.valid_iter: WavDataLoader = WavDataLoader(os.path.join(target_signals_dir, 'valid'))
            self.test_iter: WavDataLoader = WavDataLoader(os.path.join(target_signals_dir, 'test'))
            self.dataset = [self.train_iter, self.valid_iter, self.test_iter]

            gan_type: WaveGan_GP = WaveGan_GP(self.train_iter, self.valid_iter)
            self.base_trainer = gan_type
            manager = DefaultRunManager(self.test_iter, discriminator=gan_type.discriminator,
                                        generator=gan_type.generator)
            gan_type.manager = manager
            self.discriminator, self.generator = gan_type.discriminator, gan_type.generator
            # self.set_nets(self.discriminator, self.generator)
        else:
            print("I don't know your GAN. Make your own")

        self.m = manager  # m indicates manager

    # TODO! keras like evaluate function to measure accuracy
    def evaluate(self):
        # evaluate neural network on all test data set.
        pass

    # TODO! keras like compile and train. Complete keras compile
    def compile(self):
        # compile hyperparameters, train, validation and test sets
        pass

    # TODO! keras like compile and train. Complete keras fit
    def fit(self, data_sets, run=Runs,
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

    def train(self):
        self.base_trainer.train()

    def experiments(self, runs=Runs) -> NoReturn:
        # start experiments with parameters
        for run in runs:
            ## Experiments Start ##
            self.m.begin_run(run)
            # self.all_epochs(self.train_iter)
            self.base_trainer.hyperparameters = run
            self.base_trainer.train()
            ## Experiments End ##
            self.m.end_run()

    def wrap_experiments(self) -> NoReturn:
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
        runs = Runs
        self.number_of_experiments = len(runs) * self.epochs
        print(f"number of experiments or rows: {self.number_of_experiments}")

        # start experiments with parameters
        self.experiments(runs)

        # when all runs are done, save results to files
        # TODO! add time stamp like that. datetime.datetime.now().strftime("%c"). !OSError: [Errno 22] Invalid argument!
        self.m.save('results')
        print("End of all training experiments")

    def train_experiments(self):
        self.wrap_experiments()

    ######################
    # Getters and Stetters
    ######################

    @property
    def discriminator(self):
        return self._discriminator

    @discriminator.setter
    def discriminator(self, net):
        self._discriminator = net

    @discriminator.deleter
    def discriminator(self):
        del self._discriminator

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, net):
        self._generator = net

    @generator.deleter
    def generator(self):
        del self._generator

    # def set_discriminator(self, discriminator):
    #     self.discriminator = discriminator
    #
    # def get_discriminator(self):
    #     return self.discriminator
    #
    # def set_generator(self, generator):
    #     self.generator = generator
    #
    # def get_generator(self):
    #     return self.generator
    #
    # def set_nets(self, discriminator, generator):
    #     self.set_generator(discriminator)
    #     self.set_generator(generator)
    #
    # def get_nets(self):
    #     return self.get_discriminator(), self.get_generator()
