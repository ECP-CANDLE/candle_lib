#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='shared',
      version='0.0.1',
      description='CANDLE lib pip installable package',
      author='Harry Yoo',
      author_email='hsyoo@anl.gov',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      license='LICENSE.txt',
    )
