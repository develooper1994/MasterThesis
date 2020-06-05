# Standart library
from typing import NoReturn

# torch modules
import torch

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)  # On by default, leave it here for clarity

from not_functional.models.utils import BasicUtils as utls_basic
from not_functional.models.utils.visualization.visualization import plot_loss


class TrainingUtility:

    @staticmethod
    def last_touch(GAN, D, G, output_dir, D_costs_train, D_wasses_train,
    D_costs_valid, D_wasses_valid, G_costs) -> NoReturn:
        """
        Last process to end up training. Do it what do you want. Visualization, early stopping, checkpoint of models,
        calculate metrics, ...
        :return: None
        """
        utls_basic.save_models(output_dir, D, G)
        GAN.Logger.save_loss_curve()
        # Plot loss curve.
        plot_loss(D_costs_train, D_wasses_train,
                  D_costs_valid, D_wasses_valid, G_costs, output_dir)
