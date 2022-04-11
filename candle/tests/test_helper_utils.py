"""
Unit and regression test for the helper_utils module.
"""

import pytest
import argparse
import candle


@pytest.mark.parametrize("varstr", ['YES', 'TRUE', 'T', 'Y', '1'])
def test_str2bool_true(varstr):
    assert candle.str2bool(varstr) == True

@pytest.mark.parametrize("varstr", ['NO', 'FALSE', 'F', 'N', '0'])
def test_str2bool_false(varstr):
    assert candle.str2bool(varstr) == False

def test_str2bool_exception():
    with pytest.raises(argparse.ArgumentTypeError):
        candle.str2bool('candle')
