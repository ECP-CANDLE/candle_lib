"""Unit test for feature selection utils."""

import candle
import pandas as pd


def load_data():
    data_url = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Data_For_Testing/'
    file_name = 'small_drug_descriptor_data_unique_samples.txt'
    drug_descriptor = candle.get_file(file_name,
                                      data_url + file_name,
                                      cache_subdir='examples')
    data = pd.read_csv(drug_descriptor,
                       sep='\t',
                       engine='c',
                       na_values=['na', '-', ''],
                       header=0,
                       index_col=0,
                       low_memory=False)

    return data


def test_select_features_by_missing_values():
    data = load_data()
    selected = candle.select_features_by_missing_values(data, threshold=0.1)
    assert all([a == b for a, b in zip(selected, [0, 1, 2, 3, 4, 5, 6])])

    selected = candle.select_features_by_missing_values(data.values,
                                                        threshold=0.3)
    assert all([a == b for a, b in zip(selected, [0, 1, 2, 3, 4, 5, 6, 9])])


def test_select_features_by_variation():
    data = load_data()
    selected = candle.select_features_by_variation(data,
                                                   variation_measure='var',
                                                   threshold=100,
                                                   portion=None,
                                                   draw_histogram=False)
    assert all([a == b for a, b in zip(selected, [0, 3, 5])])

    selected = candle.select_features_by_variation(data,
                                                   variation_measure='std',
                                                   portion=0.2)
    assert all([a == b for a, b in zip(selected, [0, 5])])


def test_decorrelated_features():
    data = load_data()
    selected = candle.select_decorrelated_features(data,
                                                   threshold=None,
                                                   random_seed=None)
    assert all([a == b for a, b in zip(selected, [0, 1, 2, 3, 4, 5, 6, 9])])

    selected = candle.select_decorrelated_features(data,
                                                   method='spearman',
                                                   threshold=0.8,
                                                   random_seed=10)
    assert all([a == b for a, b in zip(selected, [0, 2, 6, 9])])
