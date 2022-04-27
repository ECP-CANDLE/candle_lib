"""Unit test for the clr_keras_utils module."""

import pytest

import candle


def test_clr_check_args():
    """Checks if the arguments for cyclical learning rate are valid."""
    valid_args = {
        "clr_mode": "triangular",
        "clr_base_lr": 1e-4,
        "clr_max_lr": 1e-3,
        "clr_gamma": 0.999994,
    }
    invalid_args = {
        "clr_mode": "triangulr",
        "clr_base_lr": 1e-4,
        "clr_max_lr": 1e-3,
    }  # Missing gamma

    assert candle.clr_check_args(valid_args)
    assert not candle.clr_check_args(invalid_args)


def test_clr_set_args():
    args = {
        "clr_mode": "trng1",
        "clr_base_lr": 1e-4,
        "clr_max_lr": 1e-3,
        "clr_gamma": 0.999994,
    }

    clr_keras_kwargs = candle.clr_set_args(args)

    assert clr_keras_kwargs["mode"] == "trng1"
    assert clr_keras_kwargs["base_lr"] == 1e-4
    assert clr_keras_kwargs["max_lr"] == 1e-3
    assert clr_keras_kwargs["gamma"] == 0.999994


def test_clr_callback():

    clr = candle.clr_callback(mode="trng1", base_lr=1e-4, max_lr=1e-3)

    assert clr.mode == "triangular"
    assert clr.base_lr == 1e-4
    assert clr.max_lr == 1e-3
    assert clr.gamma == 1.0

    clr = candle.clr_callback(mode="trng2", base_lr=1e-4, max_lr=1e-3)
    assert clr.mode == "triangular2"

    clr = candle.clr_callback(mode="exp", base_lr=1e-4, max_lr=1e-3, gamma=0.999994)
    assert clr.mode == "exp_range"
    assert clr.gamma == 0.999994


@pytest.mark.xfail(raises=KeyError)
def test_clr_callback_invalid():
    candle.clr_callback(mode="invalid", base_lr=1e-4, max_lr=1e-3)
