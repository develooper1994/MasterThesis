# TODO: Implement tensorboard, matplotlib or something else visualization
# *** Collect all tool abstraction to the utils.
# First matplotlib
# second tensorboard

# standart modules
import os

# torch
import torch
import torchvision
from torch import nn

import tensorboard
import numpy as np
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


def plot_conv2d_weights(net):
    for i, module in enumerate(net.modules()):
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module_name = "Conv" + str(i)
            weights = module.weight
            weights = weights.reshape(-1).detach().cpu().numpy()
            print("{} bias: ".format(module_name), module.bias)  # Bias to zero
            plt.hist(weights)
            plt.title(module_name)
            plt.show()


## inspect the data, parameters and NN(Neural Network)
def imshow(img):
    img_shape = img.shape
    print("img_shape: {}".format(img.shape))
    img = img / 2 + 0.5  # unnormalize
    trans_img = torch_image_to_numpy_image(img)
    print("img_shape: {}".format(img.shape))
    plt.imshow(trans_img)  # numpy and torch dimension orders are different so that need to change dims.
    # torch dim order: CHW(channel, height, width)
    plt.show()
    # print("image size: {}".format( np.transpose(npimg, (1, 2, 0)).size ) )


def inspect_data(loader: torch.utils.data.dataloader.DataLoader):
    # get some random training images
    loader_type = loader.dataset.train
    if loader_type:
        images, labels = get_one_iter(trainloader)
    else:
        images, labels = get_one_iter(testloader)
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print('classes: ', ''.join('%5s' % classes[labels[j]] for j in range(batch_size)))


def inspect_one_data(images):
    image = images[0, ...]
    img = np.squeeze(image)
    img = torch_image_to_numpy_image(img)  # PIL or numpy image format
    img = rgb2gray(img)  # gray scale

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][y] != 0 else 0
            ax.annotate(str(val), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'black')
    plt.show()
