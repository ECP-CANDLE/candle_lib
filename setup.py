#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="candle",
    version="0.0.1",
    description="CANDLE lib pip installable package",
    author="Harry Yoo",
    author_email="hsyoo@anl.gov",
    packages=find_packages(
        include=["candle", "candle.*"],
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"],
    ),
    license="LICENSE.txt",
    install_requires=[
        "numpy",
        "tqdm",
        "requests",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "astropy",
        "scipy",
        "patsy",
        "statsmodels",
        "protobuf",
    ],
    setup_requires=["pytest-runner", "flake8"],
    tests_requires=["pytest"],
)
