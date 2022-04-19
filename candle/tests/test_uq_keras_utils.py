"""
Unit and regression test for the uq_keras_utils module.
"""

import pytest

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

import candle


class SetupTest:
    def __init__(self):
        # Size of data
        self.ndata = 100
        # Random ground truth
        ytrue_ = np.random.randint(0, high=50, size=self.ndata)
        ytrue0 = ytrue_ % 2
        ytrue1 = 1 - ytrue0
        ytrue = np.zeros((self.ndata, 3))
        ytrue[:, 0] = ytrue0
        ytrue[:, 1] = ytrue1
        # Random predictions including abstention
        ypred0 = np.random.rand(self.ndata)
        ypred1 = 1. - ypred0
        ypred2 = np.random.rand(self.ndata)
        ypred = np.zeros((self.ndata, 3))
        ypred[:, 0] = ypred0
        ypred[:, 1] = ypred1
        ypred[:, 2] = ypred2
        # Mask for abstention
        mask0 = np.zeros(self.ndata)
        mask1 = np.ones(self.ndata)
        mask = np.vstack([mask0, mask0, mask1]).T
        self.mask = mask

        # Convert to keras variables
        self.ytrue_class = K.variable(value=ytrue)
        self.ypred_class = K.variable(value=ypred)

        # Regression
        self.nout_reg = 2
        ytrue = np.random.randn(self.ndata, self.nout_reg)
        self.ytrue = K.variable(value=ytrue)
        ypred = np.random.randn(self.ndata, self.nout_reg)
        self.ypred = K.variable(value=ypred)
        ypred = np.random.randn(self.ndata, self.nout_reg * 2)
        self.ypred_het = K.variable(value=ypred)
        ypred = np.random.randn(self.ndata, self.nout_reg * 3)
        self.ypred_qtl = K.variable(value=ypred)

        # Regression Contamination
        self.nout_reg_cont = 1
        ytrue = np.random.randn(self.ndata, self.nout_reg_cont)
        self.ytrue_cont = K.variable(value=ytrue)
        ytrue_aug_ = candle.add_index_to_output(ytrue)
        self.ytrue_aug = K.variable(value=ytrue_aug_)
        ypred = np.random.randn(self.ndata, self.nout_reg_cont)
        self.ypred_cont = K.variable(value=ypred)

        # Define model
        self.num_ouputs = 1
        inputs = Input(shape=(20,))
        x = Dense(10, activation='relu')(inputs)
        outputs = Dense(self.num_ouputs, activation='elu')(x)
        self.model = Model(inputs=inputs, outputs=outputs)


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


def test_abstention_loss(testobj):
    alpha0 = 0.1
    alpha = K.variable(value=alpha0)
    absloss = candle.abstention_loss(alpha, testobj.mask)
    try:
        absloss(testobj.ytrue_class, testobj.ypred_class)
    except Exception as e:
        print(e)
        assert 0


def test_sparse_abstention_loss():
    alpha0 = 0.1
    alpha = K.variable(value=alpha0)
    ndata = 100
    # Random ground truth
    # Direct classes (not one-hot encoded)
    ytrue_ = np.random.randint(0, high=50, size=ndata)
    ytrue = ytrue_ % 2  # 2 classes: 0, 1 labels
    # Random predictions including abstention
    # Scores distribution (2 classes) + abstention
    ypred0 = np.random.rand(ndata)
    ypred1 = 1. - ypred0
    ypred2 = np.random.rand(ndata)
    ypred = np.zeros((ndata, 3))
    ypred[:, 0] = ypred0
    ypred[:, 1] = ypred1
    ypred[:, 2] = ypred2
    # Mask for abstention
    mask0 = np.zeros(ndata)
    mask1 = np.ones(ndata)
    mask = np.vstack([mask0, mask0, mask1]).T
    mask = mask

    # Convert to keras variables
    ytrue = K.variable(value=ytrue)
    ypred = K.variable(value=ypred)

    absloss = candle.sparse_abstention_loss(alpha, mask)
    try:
        absloss(ytrue, ypred)
    except Exception as e:
        print(e)
        assert 0


def test_abstention_acc_metric(testobj):
    absmtc = candle.abstention_acc_metric(2)
    try:
        absmtc(testobj.ytrue_class, testobj.ypred_class)
    except Exception as e:
        print(e)
        assert 0


def test_sparse_abstention_acc_metric(testobj):
    absmtc = candle.sparse_abstention_acc_metric(2)
    try:
        absmtc(testobj.ytrue_class, testobj.ypred_class)
    except Exception as e:
        print(e)
        assert 0


def test_abstention_metric(testobj):
    absmtc = candle.abstention_metric(2)
    try:
        absmtc(testobj.ytrue_class, testobj.ypred_class)
    except Exception as e:
        print(e)
        assert 0


def test_acc_class_i_metric(testobj):
    absmtc = candle.acc_class_i_metric(0)
    try:
        absmtc(testobj.ytrue_class, testobj.ypred_class)
    except Exception as e:
        print(e)
        assert 0


def test_abstention_acc_class_i_metric(testobj):
    absmtc = candle.abstention_acc_class_i_metric(2, 0)
    try:
        absmtc(testobj.ytrue_class, testobj.ypred_class)
    except Exception as e:
        print(e)
        assert 0


def test_abstention_class_i_metric(testobj):
    absmtc = candle.abstention_class_i_metric(2, 1)
    try:
        absmtc(testobj.ytrue_class, testobj.ypred_class)
    except Exception as e:
        print(e)
        assert 0


def test_absadapt_callback():
    alpha0 = 0.1

    try:
        candle.AbstentionAdapt_Callback(acc_monitor='val_abstention_acc', abs_monitor='val_abstention', alpha0=alpha0)
    except Exception as e:
        print(e)
        assert 0


def test2_absadapt_callback():
    # Random data
    ndata = 150
    nin = 5
    nb_classes = 3
    mask = np.zeros(nb_classes + 1)
    mask[-1] = 1
    xtrain = np.random.randn(ndata, nin)
    ytrain_ = np.random.randint(0, high=50, size=ndata) % nb_classes
    yyt = np.random.randint(0, high=50, size=50) % nb_classes
    yyv = np.random.randint(0, high=50, size=50) % nb_classes
    ytrain, _, _ = candle.modify_labels(nb_classes + 1, ytrain_, yyt, yyv)

    inputs = Input(shape=(nin,))
    x = Dense(10, activation='relu')(inputs)
    outputs = Dense(nb_classes, activation='elu')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model = candle.add_model_output(model, mode='abstain', num_add=1, activation='sigmoid')

    alpha0 = 0.1
    cbk = candle.AbstentionAdapt_Callback(acc_monitor='val_abstention_acc', abs_monitor='val_abstention', alpha0=alpha0)
    callbacks = [cbk]
    sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    try:
        model.compile(loss=candle.abstention_loss(cbk.alpha, mask), optimizer=sgd, metrics=['acc', candle.abstention_acc_metric(nb_classes), candle.abstention_metric(nb_classes)])

        model.fit(xtrain, ytrain, batch_size=10, epochs=1, callbacks=callbacks)

    except Exception as e:
        print(e)
        assert 0


def test_modify_labels():

    nb_classes = 4
    ytrain_ = np.random.randint(0, high=50, size=200)
    ytrain = ytrain_ % nb_classes
    yval_ = np.random.randint(0, high=50, size=20)
    yval = yval_ % nb_classes
    ytest_ = np.random.randint(0, high=50, size=30)
    ytest = ytest_ % nb_classes

    nb_cl_abs = nb_classes + 1
    ytrain, ytest, yval = candle.modify_labels(nb_cl_abs, ytrain, ytest, yval)

    assert ytrain.shape[1] == nb_cl_abs
    assert ytest.shape[1] == nb_cl_abs
    assert yval.shape[1] == nb_cl_abs


def test1_add_model_output(testobj):

    num_add = 1
    model = candle.add_model_output(testobj.model, mode='abstain', num_add=num_add, activation='sigmoid')

    assert model.layers[-1].output_shape[-1] == (testobj.num_ouputs + num_add)


@pytest.mark.parametrize("mode", ['het', 'qtl'])
def test2_add_model_output(testobj, mode):

    model = candle.add_model_output(testobj.model, mode=mode)

    factor = 2
    if mode == 'qtl':
        factor = 3
    assert model.layers[-1].output_shape[-1] == (testobj.num_ouputs * factor)


def test3_add_model_output_exception(testobj):

    with pytest.raises(Exception):
        candle.add_model_output(testobj.model, mode="no")


@pytest.mark.parametrize("metric", [candle.r2_heteroscedastic_metric, candle.mae_heteroscedastic_metric, candle.mse_heteroscedastic_metric, candle.meanS_heteroscedastic_metric, ])
def test_heteroscedastic_metrics(testobj, metric):
    mtcfun = metric(testobj.nout_reg)
    try:
        mtcfun(testobj.ytrue, testobj.ypred_het)
    except Exception as e:
        print(e)
        assert 0


def test_het_loss(testobj):
    hetloss = candle.heteroscedastic_loss(testobj.nout_reg)
    try:
        hetloss(testobj.ytrue, testobj.ypred_het)
    except Exception as e:
        print(e)
        assert 0


def test_triple_qtl_loss(testobj):
    low_quantile = 0.1
    high_quantile = 0.9

    qtlloss = candle.triple_quantile_loss(testobj.nout_reg, low_quantile, high_quantile)
    try:
        qtlloss(testobj.ytrue, testobj.ypred_qtl)
    except Exception as e:
        print(e)
        assert 0


def test_qtl_loss(testobj):
    yshape = K.shape(testobj.ytrue)
    yqtl0 = K.reshape(testobj.ypred_qtl[:, 0::3], yshape)
    try:
        candle.quantile_loss(0.5, testobj.ytrue, yqtl0)
    except Exception as e:
        print(e)
        assert 0


def test_quantile_metric(testobj):
    mtcfun = candle.quantile_metric(testobj.nout_reg, 2, 0.9)
    try:
        mtcfun(testobj.ytrue, testobj.ypred_qtl)
    except Exception as e:
        print(e)
        assert 0


@pytest.mark.parametrize("nout", [1, 4])
def test_add_index_to_output(nout):
    ndata = 200
    if nout > 1:
        ytrain = np.random.randn(ndata, nout)
    else:
        ytrain = np.random.randn(ndata)
    ytrain_aug = candle.add_index_to_output(ytrain)

    assert ytrain_aug.shape[1] == (nout + 1)
    assert ytrain_aug[0, -1] == 0
    assert ytrain_aug[ndata - 1, -1] == (ndata - 1)


def test_contamination_loss(testobj):
    tk0 = np.random.rand(testobj.ndata)
    tk1 = 1. - tk0
    tk = np.vstack([tk0, tk1]).T

    Tk = K.variable(tk)
    a = K.variable(value=0.95)
    sigmaSQ = K.variable(value=0.01)
    gammaSQ = K.variable(value=0.09)

    contmloss = candle.contamination_loss(testobj.nout_reg_cont, Tk, a, sigmaSQ, gammaSQ)
    try:
        contmloss(testobj.ytrue_aug, testobj.ypred_cont)
    except Exception as e:
        print(e)
        assert 0


def test_contamination_callback():
    ndata = 100
    nin = 20
    nout = 1
    x = np.random.randn(ndata, nin)
    y = np.random.randn(ndata, nout)

    try:
        candle.Contamination_Callback(x, y)
    except Exception as e:
        print(e)
        assert 0


def test_contamination_callback_exception():
    ndata = 100
    nin = 20
    nout = 3
    x = np.random.randn(ndata, nin)
    y = np.random.randn(ndata, nout)

    with pytest.raises(Exception):
        candle.Contamination_Callback(x, y)


def test2_contamination_callback():
    # Random data
    ndata = 100
    nin = 20
    nout = 1
    xtrain = np.random.randn(ndata, nin)
    ytrain = np.random.randn(ndata, nout)
    ytrain_augmented = candle.add_index_to_output(ytrain)

    inputs = Input(shape=(nin,))
    x = Dense(10, activation='relu')(inputs)
    outputs = Dense(nout, activation='elu')(x)
    model = Model(inputs=inputs, outputs=outputs)

    cbk = candle.Contamination_Callback(xtrain, ytrain)
    callbacks = [cbk]
    sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    try:
        model.compile(loss=candle.contamination_loss(nout, cbk.T_k, cbk.a, cbk.sigmaSQ, cbk.gammaSQ), optimizer=sgd, metrics=[candle.mae_contamination_metric(nout),
                      candle.r2_contamination_metric(nout)])
        model.fit(xtrain, ytrain_augmented, batch_size=10, epochs=1, callbacks=callbacks)

    except Exception as e:
        print(e)
        assert 0


@pytest.mark.parametrize("metric", [candle.mse_contamination_metric, candle.mae_contamination_metric, candle.r2_contamination_metric, ])
def test_contamination_metrics(testobj, metric):
    mtcfun = metric(testobj.nout_reg_cont)
    try:
        mtcfun(testobj.ytrue_aug, testobj.ypred_cont)
    except Exception as e:
        print(e)
        assert 0
