"""Unit and regression test for the helper_utils module."""

import pytest

import argparse
import logging
import os
import tempfile

import candle


def test_verify_path():
    temp_dir = tempfile.TemporaryDirectory()
    candle.verify_path(temp_dir.name)
    assert os.path.exists(temp_dir.name)


def test_set_up_logger():
    temp_dir = tempfile.TemporaryDirectory()
    logfile = os.path.join(temp_dir.name, "log")
    logger = logging.getLogger(__name__)
    candle.set_up_logger(logfile, logger)

    assert logger.getEffectiveLevel() == logging.DEBUG


@pytest.mark.parametrize("varstr", [
    'YES', 'TRUE', 'T', 'Y', '1', 'Yes', 'yes', 'yeS', 'True', 'true', 'tRuE',
    'TruE', 't', 'y'
])
def test_str2bool_true(varstr):
    assert candle.str2bool(varstr)


@pytest.mark.parametrize("varstr", [
    'NO', 'FALSE', 'F', 'N', '0', 'No', 'no', 'nO', 'False', 'false', 'fAlSe',
    'f', 'n'
])
def test_str2bool_false(varstr):
    assert not candle.str2bool(varstr)


def test_str2bool_exception():
    with pytest.raises(argparse.ArgumentTypeError):
        candle.str2bool('candle')
