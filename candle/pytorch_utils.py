from __future__ import absolute_import

from typing import Dict

import torch
import torch.nn
import torch.nn.functional as F
import torch.nn.init
import torch.optim

from .helper_utils import set_seed as set_seed_defaultUtils


def set_pytorch_threads():  # for compatibility
    pass


def set_pytorch_seed(seed):
    """Set the random number seed to the desired value

    Parameters
    ----------
    seed : integer
        Random number seed.
    """

    set_seed_defaultUtils(seed)
    torch.manual_seed(seed)


def get_pytorch_function(name: str):
    mapping = {}

    # loss
    mapping["mse"] = torch.nn.MSELoss()
    mapping["binary_crossentropy"] = torch.nn.BCELoss()
    mapping["categorical_crossentropy"] = torch.nn.CrossEntropyLoss()
    mapping["smoothL1"] = torch.nn.SmoothL1Loss()

    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No pytorch function found for "{}"'.format(name))

    return mapped


def build_pytorch_activation(activation: str):

    # activation
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation == "tanh":
        return torch.nn.Tanh()


def build_pytorch_optimizer(
    model, optimizer: str, lr: float, kerasDefaults: Dict, trainable_only: bool = True
):
    if trainable_only:
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        params = model.parameters()

    # schedule = optimizers.optimizer.Schedule() # constant lr (equivalent to default Keras setting)

    if optimizer == "sgd":
        return torch.optim.GradientDescentMomentum(
            params,
            lr=lr,
            momentum_coef=kerasDefaults["momentum_sgd"],
            nesterov=kerasDefaults["nesterov_sgd"],
        )

    elif optimizer == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=kerasDefaults["rho"],
            eps=kerasDefaults["epsilon"],
        )

    elif optimizer == "adagrad":
        return torch.optim.Adagrad(
            model.parameters(), lr=lr, eps=kerasDefaults["epsilon"]
        )

    elif optimizer == "adadelta":
        return torch.optim.Adadelta(
            params, eps=kerasDefaults["epsilon"], rho=kerasDefaults["rho"]
        )

    elif optimizer == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=[kerasDefaults["beta_1"], kerasDefaults["beta_2"]],
            eps=kerasDefaults["epsilon"],
        )


def pytorch_initialize(weights, initializer, kerasDefaults, seed=None, constant=0.0):

    if initializer == "constant":
        return torch.nn.init.constant_(weights, val=constant)

    elif initializer == "uniform":
        return torch.nn.init.uniform(
            weights,
            a=kerasDefaults["minval_uniform"],
            b=kerasDefaults["maxval_uniform"],
        )

    elif initializer == "normal":
        return torch.nn.init.normal(
            weights,
            mean=kerasDefaults["mean_normal"],
            std=kerasDefaults["stddev_normal"],
        )

    elif initializer == "glorot_normal":  # not quite Xavier
        return torch.nn.init.xavier_normal(weights)

    elif initializer == "glorot_uniform":
        return torch.nn.init.xavier_uniform_(weights)

    elif initializer == "he_normal":
        return torch.nn.init.kaiming_uniform(weights)


def pytorch_xent(y_true, y_pred):
    return F.cross_entropy(y_pred, y_true)


def pytorch_mse(y_true, y_pred):
    return F.mse_loss(y_pred, y_true)
