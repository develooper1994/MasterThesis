from torch import optim


# def optimizers(arguments, netD, netG):
#     lr = arguments['learning-rate']
#     beta_one = arguments['beta-one']
#     beta_two = arguments['beta-two']
#     optimizerD = optim.Adam(netD.parameters(), lr=lr,
#                             betas=(beta_one, beta_two))
#     optimizerG = optim.Adam(netG.parameters(), lr=lr,
#                             betas=(beta_one, beta_two))
#     return optimizerD, optimizerG


def optimizers(arguments, **networks):
    networks_names = list(networks.keys())
    networks = list(networks.values())

    lr = arguments['learning-rate']  # lr_g, lr_d, lr_e
    lr_g, lr_d, lr_e = lr
    beta_one = arguments['beta-one']
    beta_two = arguments['beta-two']

    return [
        optim.Adam(net.parameters(), lr=lr_d, betas=(beta_one, beta_two))  # discriminator optimizer
        if net_name.lower() == "discriminator"
        else optim.Adam(net.parameters(), lr=lr_g, betas=(beta_one, beta_two))  # generator optimizer
        for net_name, net in zip(networks_names, networks)
    ]
