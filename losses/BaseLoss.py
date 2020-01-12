# TODO: Implement 2 type Losses. Wasserstein loss is important


def wassertein_loss(D_fake, D_real, gradient_penalty):
    D_cost_train = D_fake - D_real + gradient_penalty
    D_wass_train = D_real - D_fake
    return D_cost_train, D_wass_train