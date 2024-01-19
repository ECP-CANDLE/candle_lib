from __future__ import print_function

import logging
import os

import pandas as pd
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

logger = logging.getLogger(__name__)
candle.set_parallelism_threads()

additional_definitions = [
    {"name": "latent_dim", "type": int, "help": "latent dimensions"},
    {
        "name": "model",
        "default": "ae",
        "choices": ["ae", "vae", "cvae"],
        "help": "model to use: ae,vae,cvae",
    },
    {
        "name": "use_landmark_genes",
        "type": candle.str2bool,
        "default": False,
        "help": "use the 978 landmark genes from LINCS (L1000) as expression features",
    },
    {
        "name": "residual",
        "type": candle.str2bool,
        "default": False,
        "help": "add skip connections to the layers",
    },
    {
        "name": "reduce_lr",
        "type": candle.str2bool,
        "default": False,
        "help": "reduce learning rate on plateau",
    },
    {
        "name": "warmup_lr",
        "type": candle.str2bool,
        "default": False,
        "help": "gradually increase learning rate on start",
    },
    {"name": "base_lr", "type": float, "help": "base learning rate"},
    {
        "name": "epsilon_std",
        "type": float,
        "help": "epsilon std for sampling latent noise",
    },
    {
        "name": "cp",
        "type": candle.str2bool,
        "default": False,
        "help": "checkpoint models with best val_loss",
    },
    {
        "name": "tb",
        "type": candle.str2bool,
        "default": False,
        "help": "use tensorboard",
    },
    {
        "name": "tsne",
        "type": candle.str2bool,
        "default": False,
        "help": "generate tsne plot of the latent representation",
    },
    {
        "name": "alpha_dropout",
        "type": candle.str2bool,
        "default": False,
        "help": "use the AlphaDropout layer from keras instead of regular Dropout",
    },
]

required = [
    "epochs",
    "batch_size",
    "learning_rate"
]


class BenchmarkMNIST(candle.Benchmark):
    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


