"""
Unit and regression test for the uq_utils module.
"""

import pytest
import numpy as np
import pandas as pd

import candle


@pytest.mark.parametrize("N", [100, 1000])
def test1_frac_distribution(N):
    trf = np.random.rand(1)
    vf = 0.
    ttf = 1. - trf
    config = {'uq_train_fr': trf, 'uq_valid_fr': vf, 'uq_test_fr': ttf}

    numtrain = N
    numtest = N // 2
    numval = 0
    total = numtrain + numtest + numval
    indTr, indVl, indTt = candle.generate_index_distribution(numtrain, numtest, numval, config)
    numtr = int(np.round(total * trf))
    numtt = total - numtr
    assert len(indTr) == numtr
    assert len(indTt) == numtt
    assert indVl is None


@pytest.mark.parametrize("N", [100, 1000])
def test2_frac_distribution(N):
    trf = np.random.rand(1)
    vf = (1. - trf) / 2.
    ttf = 1. - trf - vf

    config = {'uq_train_fr': trf, 'uq_valid_fr': vf, 'uq_test_fr': ttf}

    numtrain = N
    numtest = N // 2
    numval = numtest
    total = numtrain + numtest + numval
    indTr, indVl, indTt = candle.generate_index_distribution(numtrain, numtest, numval, config)
    numtr = int(np.round(total * trf))
    numv = int(np.round(total * vf))
    numtt = total - numtr - numv
    assert len(indTr) == numtr
    assert len(indVl) == numv
    assert len(indTt) == numtt


@pytest.mark.parametrize("N", [100, 1000])
def test3_frac_distribution(N):
    trf = np.random.rand(1)
    ttf = 0.
    vf = 1. - trf
    config = {'uq_train_fr': trf, 'uq_valid_fr': vf, 'uq_test_fr': ttf}

    numtrain = N
    numtest = 0
    numval = N // 2
    total = numtrain + numtest + numval
    indTr, indVl, indTt = candle.generate_index_distribution(numtrain, numtest, numval, config)
    numtr = int(np.round(total * trf))
    numv = total - numtr
    assert len(indTr) == numtr
    assert len(indVl) == numv
    assert indTt is None


@pytest.mark.parametrize("N", [100, 1000])
def test1_blk_distribution(N):
    trb = 2
    ttb = 1
    vb = 0
    config = {'uq_train_bks': trb, 'uq_valid_bks': vb, 'uq_test_bks': ttb}

    numtrain = N
    numtest = N // 2
    numval = 0
    total = numtrain + numtest + numval
    numblk = trb + ttb + vb
    indTr, indVl, indTt = candle.generate_index_distribution(numtrain, numtest, numval, config)
    numtr = int(np.round(total * trb / numblk))
    numtt = total - numtr
    assert len(indTr) == numtr
    assert len(indTt) == numtt
    assert indVl is None


@pytest.mark.parametrize("N", [100, 1000])
def test2_blk_distribution(N):
    trb = 4
    ttb = 1
    vb = 1
    config = {'uq_train_bks': trb, 'uq_valid_bks': vb, 'uq_test_bks': ttb}

    numtrain = N
    numtest = N // 2
    numval = 0
    total = numtrain + numtest + numval
    numblk = trb + ttb + vb
    indTr, indVl, indTt = candle.generate_index_distribution(numtrain, numtest, numval, config)
    numtr = int(np.round(total * trb / numblk))
    numv = int(np.round(total * vb / numblk))
    numtt = total - numtr - numv
    assert len(indTr) == numtr
    assert len(indVl) == numv
    assert len(indTt) == numtt


@pytest.mark.parametrize("N", [100, 1000])
def test2_blklst_distribution(N):
    numtrain = N
    numtest = N // 2
    numval = numtest
    total = numtrain + numtest + numval
    blcksz = int(np.round(total / 6))

    all_lst = range(total)
    ttv = all_lst[:blcksz]
    offset = 2 * blcksz
    vv = all_lst[blcksz:offset]
    trv = all_lst[offset:]
    config = {'uq_train_vec': trv, 'uq_valid_vec': vv, 'uq_test_vec': ttv}

    indTr, indVl, indTt = candle.generate_index_distribution(numtrain, numtest, numval, config)
    assert len(indTr) == len(trv)
    assert len(indVl) == len(vv)
    assert len(indTt) == len(ttv)


def test_stats_hom_summary():
    numsamples = 100
    ncols = 8
    data = np.random.randn(numsamples, ncols)
    df = pd.DataFrame(data)
    try:
        candle.compute_statistics_homoscedastic_summary(df)
    except Exception as e:
        print(e)
        assert 0


def test_stats_hom():
    numsamples = 100
    numpred = 5
    ncols = 6 + numpred
    data = np.random.randn(numsamples, ncols)
    df = pd.DataFrame(data)
    try:
        candle.compute_statistics_homoscedastic(df)
    except Exception as e:
        print(e)
        assert 0


def test_stats_hom_exception():
    numsamples = 100
    ncols = 3
    data = np.random.randn(numsamples, ncols)
    df = pd.DataFrame(data)
    with pytest.raises(Exception):
        candle.compute_statistics_homoscedastic(df)


def test_stats_het():
    numsamples = 100
    numpred = 5
    ncols = 6 + 2 * numpred
    data = np.random.randn(numsamples, ncols)
    df = pd.DataFrame(data)
    try:
        candle.compute_statistics_heteroscedastic(df)
    except Exception as e:
        print(e)
        assert 0


def test_stats_het_exception():
    numsamples = 100
    ncols = 3
    data = np.random.randn(numsamples, ncols)
    df = pd.DataFrame(data)
    with pytest.raises(Exception):
        candle.compute_statistics_heteroscedastic(df)


def test_stats_qtl():
    numsamples = 100
    numpred = 5
    ncols = 6 + 3 * numpred
    data = np.random.randn(numsamples, ncols)
    df = pd.DataFrame(data)
    try:
        candle.compute_statistics_quantile(df)
    except Exception as e:
        print(e)
        assert 0


def test_stats_qtl_exception():
    numsamples = 100
    ncols = 3
    data = np.random.randn(numsamples, ncols)
    df = pd.DataFrame(data)
    with pytest.raises(Exception):
        candle.compute_statistics_quantile(df)


def test_split_for_calibration():
    numsamples = 100
    ytrue = np.random.randn(numsamples)
    ypred = np.random.randn(numsamples)
    sigma = np.random.randn(numsamples) + 1.
    splitf = np.random.rand(1)
    try:
        candle.split_data_for_empirical_calibration(ytrue, ypred, sigma, splitf)
    except Exception as e:
        print(e)
        assert 0


def test_calibration_interpolation():
    numsamples = 100
    psigma = np.random.randn(numsamples) + 1.
    ytrue = np.random.randn(numsamples)
    ypred = np.random.randn(numsamples)

    try:
        candle.compute_empirical_calibration_interpolation(psigma, ypred, ytrue, cv=3)
    except Exception as e:
        print(e)
        assert 0
