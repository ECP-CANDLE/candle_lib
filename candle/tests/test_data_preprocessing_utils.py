"""Unit test for data_preprocessing_utils."""

import candle
import pytest
import pandas as pd
import numpy as np


@pytest.mark.skip(reason='This test take a while to run.')
def test_quantile_normalization():
    data_url = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Data_For_Testing/'
    file_name = 'Gene_Expression_Full_Data_Unique_Samples.txt'
    gene_expression = candle.get_file(file_name,
                                      data_url + file_name,
                                      cache_subdir='examples')
    data = pd.read_csv(gene_expression,
                       sep='\t',
                       engine='c',
                       na_values=['na', '-', ''],
                       header=0,
                       index_col=[0, 1],
                       low_memory=False)

    norm_data = candle.quantile_normalization(np.transpose(data))
    norm_data = np.transpose(norm_data)
    third_quartile = norm_data.quantile(0.75, axis=0)

    assert np.round(a=np.max(third_quartile) - np.min(third_quartile),
                    decimals=2) == 0.01


def isArrayEqual(list_a, list_b):
    if len(list_a) != len(list_b):
        return False
    return all([a == b for a, b in zip(list_a, list_b)])


def test_generate_cross_validation_partition():
    generated = candle.generate_cross_validation_partition(range(5),
                                                           n_folds=5,
                                                           n_repeats=2,
                                                           portions=None,
                                                           random_seed=0)
    expected = [[[2], [0, 1, 3, 4]], [[0], [1, 2, 3, 4]], [[1], [0, 2, 3, 4]],
                [[3], [0, 1, 2, 4]], [[4], [0, 1, 2, 3]], [[0], [1, 2, 3, 4]],
                [[2], [0, 1, 3, 4]], [[1], [0, 2, 3, 4]], [[4], [0, 1, 2, 3]],
                [[3], [0, 1, 2, 4]]]

    eval_1 = [
        isArrayEqual(generated[i][0], expected[i][0])
        for i in range(len(generated))
    ]
    eval_2 = [
        isArrayEqual(generated[i][1], expected[i][1])
        for i in range(len(generated))
    ]

    assert all(eval_1)
    assert all(eval_2)
