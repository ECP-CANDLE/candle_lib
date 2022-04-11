"""
Unit and regression test for the helper_utils module.
"""

import pytest
import argparse
import candle


@pytest.mark.parametrize("varstr", ['YES', 'TRUE', 'T', 'Y', '1', 'Yes', 'yes', 'yeS', 'True', 'true', 'tRuE', 'TruE', 't', 'y'])
def test_str2bool_true(varstr):
    assert candle.str2bool(varstr)


@pytest.mark.parametrize("varstr", ['NO', 'FALSE', 'F', 'N', '0', 'No', 'no', 'nO', 'False', 'false', 'fAlSe', 'f', 'n'])
def test_str2bool_false(varstr):
    assert not candle.str2bool(varstr)


def test_str2bool_exception():
    with pytest.raises(argparse.ArgumentTypeError):
        candle.str2bool('candle')
