"""
Unit and regression test for the candle package.
"""

import candle
import sys


def test_candle_imported():
    assert "candle" in sys.modules
