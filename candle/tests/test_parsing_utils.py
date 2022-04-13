"""
Unit test for the parsing utils module.
"""

import pytest

import argparse

import candle


def test_argument_struct():
    epochs = 10
    learning_rate = 1e-3
    argdict = {'epochs': epochs, 'learning_rate': learning_rate}
    argstruct = candle.ArgumentStruct(**argdict)
    assert argstruct.epochs == epochs
    assert argstruct.learning_rate == learning_rate


class SetupTest:
    def __init__(self):
        self.config = [{'name': 'rng_seed',
                        'abv': 'r',
                        'type': float,
                        'default': 123,
                        'help': 'random number generator seed.'},
                       {'name': 'train_bool',
                        'type': candle.str2bool,
                        'default': True,
                        'help': 'train model.'},
                       {'name': 'clr_flag',
                        'type': candle.str2bool,
                        'default': True,
                        'help': 'CLR flag (boolean).'},
                       ]
        self.parser = argparse.ArgumentParser(conflict_handler='resolve')


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


def test_parse_from_dictlist(testobj):
    parser_with_conf = candle.parse_from_dictlist(testobj.config, testobj.parser)
    argswc = parser_with_conf.parse_args(['--rng_seed', '456'])
    assert argswc.rng_seed == 456
    assert argswc.train_bool


def test_flag_conflicts(testobj):
    cnf = list(testobj.config)
    parser = candle.parse_from_dictlist(cnf, testobj.parser)
    argswc = parser.parse_args(['--rng_seed', '456'])
    argswc_dict = vars(argswc)
    candle.check_flag_conflicts(argswc_dict)


def test_flag_conflicts_exception(testobj):
    conf2 = [{'name': 'warmup_lr',
              'type': candle.str2bool,
              'default': True,
              'help': 'gradually increase learning rate on start'},
             ]
    cnf = list(testobj.config) + conf2
    parser = candle.parse_from_dictlist(cnf, testobj.parser)
    argswc = parser.parse_args(['--rng_seed', '456'])
    argswc_dict = vars(argswc)
    with pytest.raises(Exception):
        candle.check_flag_conflicts(argswc_dict)


def test_indirect_finalize_parameters(testobj):
    """Cannot check directly the function since it
    requires proper command line options. Checking
    some functionality inside of it.
    """
    filepath = './'
    defmodel = ''
    framework = 'keras'
    prog = 'fake_bmk'
    desc = 'Fake benchmark'
    bmk = candle.Benchmark(filepath,
                           defmodel,
                           framework,
                           prog,
                           desc,
                           )
    bmk.parser = candle.parse_from_dictlist(testobj.config, bmk.parser)
    argswc = bmk.parser.parse_args(['--rng_seed', '456'])
    gpar = vars(argswc)
    try:
        bmk.check_required_exists(gpar)
        candle.check_flag_conflicts(gpar)
    except Exception as e:
        print(e)
        assert 0
