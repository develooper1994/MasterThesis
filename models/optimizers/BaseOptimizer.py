from torch import optim


def optimizers(netD, netG, arguments):
    lr = arguments['learning-rate']
    beta_one = arguments['beta-one']
    beta_two = arguments['beta-two']
    optimizerD = optim.Adam(netD.parameters(), lr=lr,
                            betas=(beta_one, beta_two))
    optimizerG = optim.Adam(netG.parameters(), lr=lr,
                            betas=(beta_one, beta_two))
    return optimizerD, optimizerG
