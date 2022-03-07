"""
Unit and regression test for the file_utils module.
"""

import candle


def test_get_file():
    """
    Test the get_file function.
    """
    MD5_HASH = 'bd638baadc190176439f388d28d8a091'
    candle.get_file(fname='LICENSE.txt',
                    origin='https://raw.githubusercontent.com/hyoo/candle_lib/master/LICENSE.txt',
                    md5_hash=MD5_HASH,
                    datadir='./')

    candle.get_file(fname='README.md',
                    origin='https://raw.githubusercontent.com/hyoo/candle_lib/master/README.md',
                    md5_hash=MD5_HASH,
                    datadir='./')

    # should match MD5
    assert candle.validate_file(fpath='LICENSE.txt', md5_hash=MD5_HASH)

    # should not match MD5
    assert not candle.validate_file(fpath='README.md', md5_hash=MD5_HASH)
