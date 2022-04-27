"""Unit test for the benchmark module."""

import pytest

import candle

additional_definitions1 = [
    {"name": "base_lr", "type": float, "help": "base learning rate"},
]
additional_definitions2 = [
    {"name": "latent_dim", "type": int, "help": "latent dimensions"},
    {
        "name": "model",
        "default": "ae",
        "choices": ["ae", "vae", "cvae"],
        "help": "model to use: ae,vae,cvae",
    },
]
required1 = ["loss", "latent_dim", "model"]
required2 = ["dropout"]


class SetupTest:
    def __init__(self):
        self.additional_definitions = additional_definitions1
        self.required = required1
        self.filepath = "./"
        self.defmodel = ""
        self.framework = "keras"
        self.prog = "fake_bmk"
        self.desc = "Fake benchmark"


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


def test_benchmark_build(testobj):
    try:
        candle.Benchmark(
            testobj.filepath,
            testobj.defmodel,
            testobj.framework,
            testobj.prog,
            testobj.desc,
        )
    except Exception as e:
        print(e)
        assert 0


@pytest.mark.parametrize("required", [None, required1, required2])
def test_benchmark_required(testobj, required):
    class Benchmark2(candle.Benchmark):
        def set_locals(self):
            if required is not None:
                self.required = set(required)
            self.additional_definitions = testobj.additional_definitions

    try:
        Benchmark2(
            testobj.filepath,
            testobj.defmodel,
            testobj.framework,
            testobj.prog,
            testobj.desc,
        )
    except Exception as e:
        print(e)
        assert 0


@pytest.mark.parametrize(
    "additional_definitions", [None, additional_definitions1, additional_definitions2]
)
def test_benchmark_additional(testobj, additional_definitions):
    class Benchmark2(candle.Benchmark):
        def set_locals(self):
            self.required = set(testobj.required)
            if additional_definitions is not None:
                self.additional_definitions = additional_definitions

    try:
        Benchmark2(
            testobj.filepath,
            testobj.defmodel,
            testobj.framework,
            testobj.prog,
            testobj.desc,
        )
    except Exception as e:
        print(e)
        assert 0
