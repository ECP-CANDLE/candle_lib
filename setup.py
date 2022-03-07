#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='candle',
      version='0.0.1',
      description='CANDLE lib pip installable package',
      author='Harry Yoo',
      author_email='hsyoo@anl.gov',
      packages=find_packages(include=['candle', 'candle.*'], exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
      license='LICENSE.txt',
      install_requires=[
          'numpy',
          'six',
          'tqdm',
          'requests',
      ],
      setup_requires=['flake8'],
      )
