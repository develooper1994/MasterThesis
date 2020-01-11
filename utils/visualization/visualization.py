# TODO: Implement tensorboard, matplotlib or something else visualization
# *** Collect all tool abstraction to the utils.
# First matplotlib
# second tensorboard

import os

import tensorboard
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

# TODO: Implement tensorboard visualization
def plot_loss(D_cost_train, D_wass_train, D_cost_valid, D_wass_valid,
              G_cost, save_path) -> None:
    """
    Visualize Discriminators and Generator with respect to cost and Wasserstein(metric) loss using Matplotlib
    :param D_cost_train: Discriminators train cost
    :param D_wass_train: Discriminators train Wasserstein cost
    :param D_cost_valid: Discriminators validation cost
    :param D_wass_valid: Discriminators validation Wasserstein cost
    :param G_cost: Generator cost
    :param save_path: Image path. Save plot as image.
    :return: None
    """
    assert len(D_cost_train) == len(D_wass_train) == len(D_cost_valid) == len(D_wass_valid) == len(G_cost)

    save_path = os.path.join(save_path, "loss_curve.png")

    x = range(len(D_cost_train))

    y1 = D_cost_train
    y2 = D_wass_train
    y3 = D_cost_valid
    y4 = D_wass_valid
    y5 = G_cost

    plt.plot(x, y1, label='D_loss_train')
    plt.plot(x, y2, label='D_wass_train')
    plt.plot(x, y3, label='D_loss_valid')
    plt.plot(x, y4, label='D_wass_valid')
    plt.plot(x, y5, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)