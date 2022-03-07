# CANDLE Library

## Install
```
pip install git+https://github.com/hyoo/candle_lib.git
```

## File Util Example
```
import candle
candle.get_file('lincs1000.tsv', 'https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B1/lincs1000.tsv', datadir='./data')
candle.get_file(fname='P3B1_data.tar.gz', origin='https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P3B1/P3B1_data.tar.gz', unpack=True, datadir='./data')
```
See [here](https://ecp-candle.github.io/Candle/candle_lib/file_utils.html) for more detail about file utils.

