
"""
CKPT PYTORCH UTILS.

CANDLE checkpoint/restart utilities for PyTorch

"""

# Python imports
from typing import Dict

# PyTorch imports
import torch

# CANDLE imports
import candle
from ckpt_utils import CandleCkpt


class CandleCkptPyTorch(CandleCkpt):
    """
    PyTorch Callback for CANDLE-compliant Benchmarks to use for checkpointing
    Creates a JSON file alongside the weights and optimizer checkpoints that
    includes important metadata, particularly for restarting and tracking
    complex workflows.
    """

    def __init__(self, gParams: Dict, logger="DEFAULT", verbose=True):
        super().__init__(gParams, logger, verbose)
        self.ckpt_metric_fn = \
            candle.get_pytorch_function(gParams["save_best_metric"])

    def ckpt_epoch(self, epoch, model):
        ckpt_metric = self.ckpt_metric_fn.item()
        super().ckpt_epoch(self.model, epoch, ckpt_metric)

    def write_model_backend(self, epoch, model):
        # PyTorch-specific method
        torch.save({
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": 0
        }, self.model_file)

    def build_model(self, model, model_file):
        model.load_weights(model_file)
