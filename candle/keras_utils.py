import json
import warnings
from datetime import datetime
from typing import Dict

import numpy as np
from scipy.stats.stats import pearsonr
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dropout
from tensorflow.keras.metrics import (
    binary_crossentropy,
    mean_absolute_error,
    mean_squared_error,
)
from tensorflow.keras.utils import get_custom_objects

from .helper_utils import set_seed as set_seed_defaultUtils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.metrics import r2_score

import os


def set_parallelism_threads():
    """
    Set the number of parallel threads according to the number available on
    the hardware.
    """

    if (
        K.backend() == "tensorflow"
        and "NUM_INTRA_THREADS" in os.environ
        and "NUM_INTER_THREADS" in os.environ
    ):
        import tensorflow as tf

        # print('Using Thread Parallelism: {} NUM_INTRA_THREADS, {} NUM_INTER_THREADS'.format(os.environ['NUM_INTRA_THREADS'], os.environ['NUM_INTER_THREADS']))
        session_conf = tf.ConfigProto(
            inter_op_parallelism_threads=int(os.environ["NUM_INTER_THREADS"]),
            intra_op_parallelism_threads=int(os.environ["NUM_INTRA_THREADS"]),
        )
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)


def set_seed(seed: int):
    """
    Set the random number seed to the desired value.

    :param int seed: Random number seed.
    """

    set_seed_defaultUtils(seed)

    if K.backend() == "tensorflow":
        import tensorflow as tf

        if tf.__version__ < "2.0.0":
            tf.compat.v1.set_random_seed(seed)
        else:
            tf.random.set_seed(seed)


def get_function(name: str):
    mapping = {}

    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No Keras function found for "{}"'.format(name))

    return mapped


def build_optimizer(optimizer, lr, kerasDefaults):
    """
    Set the optimizer to the appropriate Keras optimizer function based on
    the input string and learning rate. Other required values are set to the
    Keras default values.

    :param string optimizer: String to choose the optimizer \
        Options recognized: 'sgd', 'rmsprop', 'adagrad', adadelta', 'adam' \
        See the Keras documentation for a full description of the options
    :param float lr: Learning rate
    :param Dict kerasDefaults: Dictionary of default parameter values to ensure consistency between frameworks

    :return: The appropriate Keras optimizer function
    """

    import tensorflow as tf

    # Some optimizer argument keywords change after TF 2.11:
    if tf.__version__ < "2.11":
        if optimizer == "sgd":
            return optimizers.SGD(
                lr=lr,
                decay=kerasDefaults["decay_lr"],
                momentum=kerasDefaults["momentum_sgd"],
                nesterov=kerasDefaults["nesterov_sgd"],
            )

        elif optimizer == "rmsprop":
            return optimizers.RMSprop(
                lr=lr,
                rho=kerasDefaults["rho"],
                epsilon=kerasDefaults["epsilon"],
                decay=kerasDefaults["decay_lr"],
            )

        elif optimizer == "adagrad":
            return optimizers.Adagrad(
                lr=lr,
                epsilon=kerasDefaults["epsilon"],
                decay=kerasDefaults["decay_lr"],
            )

        elif optimizer == "adadelta":
            return optimizers.Adadelta(
                lr=lr,
                rho=kerasDefaults["rho"],
                epsilon=kerasDefaults["epsilon"],
                decay=kerasDefaults["decay_lr"],
            )

        elif optimizer == "adam":
            return optimizers.Adam(
                lr=lr,
                beta_1=kerasDefaults["beta_1"],
                beta_2=kerasDefaults["beta_2"],
                epsilon=kerasDefaults["epsilon"],
                decay=kerasDefaults["decay_lr"],
            )
    else:  # TF >= 2.12
        # Define a decay schedule that mimics the prior Keras behavior:
        # Note that kerasDefaults["decay_lr"] is 0
        decay_function = tf.keras.optimizers.schedules.ExponentialDecay(
            lr, 100000, kerasDefaults["decay_lr"], staircase=True
        )
        if optimizer == "sgd":
            return optimizers.SGD(
                learning_rate=decay_function,
                momentum=kerasDefaults["momentum_sgd"],
                nesterov=kerasDefaults["nesterov_sgd"],
            )

        elif optimizer == "rmsprop":
            return optimizers.RMSprop(
                learning_rate=decay_function,
                rho=kerasDefaults["rho"],
                epsilon=kerasDefaults["epsilon"],
            )

        elif optimizer == "adagrad":
            return optimizers.Adagrad(
                learning_rate=decay_function,
                epsilon=kerasDefaults["epsilon"],
            )

        elif optimizer == "adadelta":
            return optimizers.Adadelta(
                learning_rate=decay_function,
                rho=kerasDefaults["rho"],
                epsilon=kerasDefaults["epsilon"],
            )

        elif optimizer == "adam":
            return optimizers.Adam(
                learning_rate=decay_function,
                beta_1=kerasDefaults["beta_1"],
                beta_2=kerasDefaults["beta_2"],
                epsilon=kerasDefaults["epsilon"],
            )


def build_initializer(
    initializer: str, kerasDefaults: Dict, seed: int = None, constant: float = 0.0
):
    """
    Set the initializer to the appropriate Keras initializer function based
    on the input string and learning rate. Other required values are set to the
    Keras default values.

    :param string initializer: String to choose the initializer \
        Options recognized: 'constant', 'uniform', 'normal', \
        'glorot_uniform', 'lecun_uniform', 'he_normal' \
        See the Keras documentation for a full description of the options
    :param Dict kerasDefaults: Dictionary of default parameter values to ensure consistency between frameworks
    :param int seed: Random number seed
    :param float constant: Constant value (for the constant initializer only)

    :return: The appropriate Keras initializer function
    """

    if initializer == "constant":
        return initializers.Constant(value=constant)

    elif initializer == "uniform":
        return initializers.RandomUniform(
            minval=kerasDefaults["minval_uniform"],
            maxval=kerasDefaults["maxval_uniform"],
            seed=seed,
        )

    elif initializer == "normal":
        return initializers.RandomNormal(
            mean=kerasDefaults["mean_normal"],
            stddev=kerasDefaults["stddev_normal"],
            seed=seed,
        )

    elif initializer == "glorot_normal":
        # aka Xavier normal initializer. keras default
        return initializers.glorot_normal(seed=seed)

    elif initializer == "glorot_uniform":
        return initializers.glorot_uniform(seed=seed)

    elif initializer == "lecun_uniform":
        return initializers.lecun_uniform(seed=seed)

    elif initializer == "he_normal":
        return initializers.he_normal(seed=seed)


def xent(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def covariance(x, y):
    return K.mean(x * y) - K.mean(x) * K.mean(y)


def corr(y_true, y_pred):
    cov = covariance(y_true, y_pred)
    var1 = covariance(y_true, y_true)
    var2 = covariance(y_pred, y_pred)
    return cov / (K.sqrt(var1 * var2) + K.epsilon())


def evaluate_autoencoder(y_pred, y_test):
    mse = mean_squared_error(y_pred, y_test)
    r2 = r2_score(y_test, y_pred)
    corr, _ = pearsonr(y_pred.flatten(), y_test.flatten())
    # print('Mean squared error: {}%'.format(mse))
    return {"mse": mse, "r2_score": r2, "correlation": corr}


class PermanentDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(PermanentDropout, self).__init__(rate, **kwargs)
        self.uses_learning_phase = False

    def call(self, x, mask=None):
        if 0.0 < self.rate < 1.0:
            noise_shape = self._get_noise_shape(x)
            x = K.dropout(x, self.rate, noise_shape)
        return x


def register_permanent_dropout():
    get_custom_objects()["PermanentDropout"] = PermanentDropout


class LoggingCallback(Callback):
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (
            epoch,
            ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())),
        )
        self.print_fcn(msg)


def compute_trainable_params(model):
    """
    Extract number of parameters from the given Keras model

    :param model: Keras model

    :return: python dictionary that contains trainable_params, non_trainable_params and total_params
    """
    if str(type(model)).startswith("<class 'keras."):
        from keras import backend as K
    else:
        import tensorflow.keras.backend as K

    trainable_count = int(np.sum([K.count_params(w) for w in model.trainable_weights]))
    non_trainable_count = int(
        np.sum([K.count_params(w) for w in model.non_trainable_weights])
    )

    return {
        "trainable_params": trainable_count,
        "non_trainable_params": non_trainable_count,
        "total_params": (trainable_count + non_trainable_count),
    }


class TerminateOnTimeOut(Callback):
    """
    This class implements timeout on model training.

    When the script reaches timeout,
    this class sets model.stop_training = True
    """

    def __init__(self, timeout_in_sec=10):
        """
        Initialize TerminateOnTimeOut class.

        :param int timeout_in_sec: seconds to timeout
        """

        super(TerminateOnTimeOut, self).__init__()
        self.run_timestamp = None
        self.timeout_in_sec = timeout_in_sec

    def on_train_begin(self, logs={}):
        """Start clock to calculate timeout."""
        self.run_timestamp = datetime.now()

    def on_epoch_end(self, epoch, logs={}):
        """On every epoch end, check whether it exceeded timeout and terminate
        training if necessary."""
        run_end = datetime.now()
        run_duration = run_end - self.run_timestamp
        run_in_sec = run_duration.total_seconds()
        print("Current time ....%2.3f" % run_in_sec)
        if self.timeout_in_sec != -1:
            if run_in_sec >= self.timeout_in_sec:
                print(
                    "Timeout==>Runtime: %2.3fs, Maxtime: %2.3fs"
                    % (run_in_sec, self.timeout_in_sec)
                )
                self.model.stop_training = True


class CandleRemoteMonitor(Callback):
    """
    Capture Run level output and store/send for monitoring.
    """

    def __init__(self, params=None):
        super(CandleRemoteMonitor, self).__init__()

        self.global_params = params

        # init
        self.experiment_id = None
        self.run_id = None
        self.run_timestamp = None
        self.epoch_timestamp = None
        self.log_messages = []

    def on_train_begin(self, logs=None):
        logs = logs or {}
        self.run_timestamp = datetime.now()
        self.experiment_id = (
            self.global_params["experiment_id"]
            if "experiment_id" in self.global_params
            else "EXP_default"
        )
        self.run_id = (
            self.global_params["run_id"]
            if "run_id" in self.global_params
            else "RUN_default"
        )

        run_params = []
        for key, val in self.global_params.items():
            run_params.append("{}: {}".format(key, val))

        send = {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "parameters": run_params,
            "start_time": str(self.run_timestamp),
            "status": "Started",
        }
        # print("on_train_begin", send)
        self.log_messages.append(send)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_timestamp = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        epoch_total = self.global_params["epochs"]
        epoch_duration = datetime.now() - self.epoch_timestamp
        epoch_in_sec = epoch_duration.total_seconds()
        epoch_line = "epoch: {}/{}, duration: {}s, loss: {}, val_loss: {}".format(
            (epoch + 1), epoch_total, epoch_in_sec, loss, val_loss
        )

        send = {
            "run_id": self.run_id,
            "status": {"set": "Running"},
            "training_loss": {"set": loss},
            "validation_loss": {"set": val_loss},
            "run_progress": {"add": [epoch_line]},
        }
        # print("on_epoch_end", send)
        self.log_messages.append(send)

    def on_train_end(self, logs=None):
        logs = logs or {}
        run_end = datetime.now()
        run_duration = run_end - self.run_timestamp
        run_in_hour = run_duration.total_seconds() / (60 * 60)

        send = {
            "run_id": self.run_id,
            "runtime_hours": {"set": run_in_hour},
            "end_time": {"set": str(run_end)},
            "status": {"set": "Finished"},
            "date_modified": {"set": "NOW"},
        }
        # print("on_train_end", send)
        self.log_messages.append(send)

        # save to file when finished
        self.save()

    def save(self):
        """Save log_messages to file."""
        # path = os.getenv('TURBINE_OUTPUT') if 'TURBINE_OUTPUT' in os.environ else '.'
        path = (
            self.global_params["output_dir"]
            if "output_dir" in self.global_params
            else "."
        )
        if not os.path.exists(path):
            os.makedirs(path)

        filename = "/run.{}.json".format(self.run_id)
        with open(path + filename, "a") as file_run_json:
            file_run_json.write(
                json.dumps(self.log_messages, indent=4, separators=(",", ": "))
            )
