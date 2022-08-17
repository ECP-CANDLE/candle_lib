
"""
CKPT KERAS UTILS.

CANDLE checkpoint/restart utilities for Keras

"""

# Python imports
from typing import Dict

# TensorFlow imports
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import Model

# CANDLE imports
from .ckpt_utils import CandleCkpt


class MultiGPUCheckpoint(ModelCheckpoint):
    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model


class CandleCkptKeras(Callback, CandleCkpt):
    """
    Keras Callback for CANDLE-compliant Benchmarks to use for checkpointing
    Creates a JSON file alongside the weights and optimizer checkpoints that
    includes important metadata, particularly for restarting and tracking
    complex workflows.
    """

    def __init__(self, gParameters: Dict, logger="DEFAULT", verbose=True):
        super().__init__()
        if logger != "DEFAULT":
            self.logger = logger
        self.scan_params(gParameters)

    # Keras Callback API:
    def on_epoch_end(self, epoch, logs=None):
        ckpt_metric = logs[self.save_best_metric]
        self.ckpt_epoch(self.model, epoch, ckpt_metric)

    def write_model_backend(self, model):
        # Keras-specific method
        model.save(self.model_file)  # save_format="h5"

    def build_model(self, model, model_file):
        model.load_weights(model_file)
