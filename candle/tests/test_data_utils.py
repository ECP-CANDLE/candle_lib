"""
Unit test for the data_tuils module.
"""

import candle
import pytest


@pytest.mark.skip(reason="used by load_Xy_data_noheader")
def test_to_categorical():
    pass


@pytest.mark.skip(reason="used by load_Xy_data2")
def test_convert_to_class():
    pass


@pytest.mark.skip(reason="used by impute_and_scale_array")
def test_scale_array():
    pass


# should we keep this?
@pytest.mark.skip(reason="impute_and_scale_array is not used")
def test_impute_and_scale_array():
    pass


# should we keep this?
@pytest.mark.skip(reason="this function is not used")
def test_drop_impute_and_scale_dataframe():
    pass


# should we keep this?
@pytest.mark.skip(reason="this function is not used")
def test_discretize_dataframe():
    pass


# should we keep this?
@pytest.mark.skip(reason="this function is not used")
def test_discretize_array():
    pass


# should we keep this?
@pytest.mark.skip(reason="this function is not used")
def test_lookup():
    pass


# should we keep this?
@pytest.mark.skip(reason="referenced in p1b1 but succeeded by load_csv_data. no-longer")
def test_load_X_data():
    pass


# should we keep this?
@pytest.mark.skip(reason="this function is not used")
def test_load_X_data2():
    pass


# should we keep this?
@pytest.mark.skip(reason="this function is not used")
def test_load_Xy_one_hot_data():
    pass


# should we keep this?
@pytest.mark.skip(reason="used by p1b2 only")
def test_load_Xy_one_hot_data2():
    pass


# should we keep this?
@pytest.mark.skip(reason="used by p1b2 only")
def test_load_Xy_data2():
    pass


# should we keep this?
@pytest.mark.skip(reason="used by tc1 only")
def test_load_Xy_data_noheader():
    pass


def test_load_csv_data():
    import numpy as np
    DEFAULT_DATATYPE = np.float32  # will be replaced by default_utils.DEFAULT_DATATYPE once available
    params = {
        'data_url': 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B1/',
        'train_data': 'P1B1.dummy.train.csv',
        'test_data': 'P1B1.dummy.test.csv',
        'feature_subsample': 0,
        'shuffle': False,
        'scaling': 'minmax',
        'data_type': DEFAULT_DATATYPE,
        'val_split': 0.1,
    }
    train_path = candle.fetch_file(params['data_url'] + params['train_data'], 'Pilot1')
    test_path = candle.fetch_file(params['data_url'] + params['test_data'], 'Pilot1')
    x_cols = None
    drop_cols = ['case_id']
    onehot_cols = ['cancer_type']
    y_cols = ['cancer_type']
    seed = 2017

    x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels = candle.load_csv_data(
        train_path,
        test_path,
        x_cols=x_cols,
        y_cols=y_cols,
        drop_cols=drop_cols,
        onehot_cols=onehot_cols,
        n_cols=params['feature_subsample'],
        shuffle=params['shuffle'],
        scaling=params['scaling'],
        dtype=params['data_type'],
        validation_split=params['val_split'],
        return_dataframe=False,
        return_header=True,
        nrows=params['train_samples'] if 'train_samples' in params and params['train_samples'] > 0 else None,
        seed=seed)

    assert len(x_train) == 9
    assert len(x_train[0]) == 60483
    assert len(y_train) == 9
    assert len(x_val) == 1
    assert len(y_val) == 1
    assert len(x_test) == 1
    assert len(y_test) == 1
    assert len(x_labels) == 60483
    assert len(y_labels) == 1
