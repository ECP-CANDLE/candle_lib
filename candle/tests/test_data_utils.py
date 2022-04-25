"""Unit test for the data_tuils module."""

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
@pytest.mark.skip(
    reason="referenced in p1b1 but succeeded by load_csv_data. no longer used")
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


# used by p1b2
def test_load_Xy_one_hot_data2():
    import numpy as np
    DEFAULT_DATATYPE = np.float32  # will be replaced by default_utils.DEFAULT_DATATYPE once available

    params = {
        'data_url': 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B2/',
        'train_data': 'P1B2.dummy.train.csv',
        'test_data': 'P1B2.dummy.test.csv',
        'feature_subsample': 0,
        'shuffle': True,
        'scaling': 'minmax',
        'val_split': 0.1,
        'data_type': DEFAULT_DATATYPE
    }
    file_train = candle.fetch_file(params['data_url'] + params['train_data'],
                                   subdir='Pilot1')
    file_test = candle.fetch_file(params['data_url'] + params['test_data'],
                                  subdir='Pilot1')
    seed = 2017
    (x_train,
     y_train), (x_val, y_val), (x_test, y_test) = candle.load_Xy_one_hot_data2(
         file_train,
         file_test,
         class_col=['cancer_type'],
         drop_cols=['case_id', 'cancer_type'],
         n_cols=params['feature_subsample'],
         shuffle=params['shuffle'],
         scaling=params['scaling'],
         validation_split=params['val_split'],
         dtype=params['data_type'],
         seed=seed)

    assert x_train.shape == (9, 28204)
    assert len(y_train) == 9
    assert len(x_val) == 0
    assert len(y_val) == 0
    assert len(x_test) == 1
    assert len(y_test) == 1


# should we keep this?
@pytest.mark.skip(reason="referenced in p1b2 but not used")
def test_load_Xy_data2():
    pass


# used by tc1
def test_load_Xy_data_noheader():
    import numpy as np
    DEFAULT_DATATYPE = np.float32  # will be replaced by default_utils.DEFAULT_DATATYPE once available
    params = {
        'data_url':
            'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/type-class/',
        'train_data':
            'type_18_300_train.dummy.csv',
        'test_data':
            'type_18_300_test.dummy.csv',
        'data_type':
            DEFAULT_DATATYPE,
        'classes':
            36,
    }
    train_path = candle.fetch_file(params['data_url'] + params['train_data'],
                                   'Pilot1')
    test_path = candle.fetch_file(params['data_url'] + params['test_data'],
                                  'Pilot1')
    usecols = None

    x_train, y_train, x_test, y_test = candle.load_Xy_data_noheader(
        train_path,
        test_path,
        params['classes'],
        usecols,
        scaling='maxabs',
        dtype=params['data_type'])

    assert x_train.shape == (10, 60483)
    assert len(y_train) == 10
    assert x_test.shape == (2, 60483)
    assert len(y_test) == 2


# used by p1b1
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
    train_path = candle.fetch_file(params['data_url'] + params['train_data'],
                                   'Pilot1')
    test_path = candle.fetch_file(params['data_url'] + params['test_data'],
                                  'Pilot1')
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
        nrows=params['train_samples']
        if 'train_samples' in params and params['train_samples'] > 0 else None,
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
