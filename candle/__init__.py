name = "candle"
__version__ = '0.0.1'

from .benchmark_def import Benchmark

# import from data_utils
from .data_utils import (
    load_csv_data,
    load_Xy_one_hot_data2,
    load_Xy_data_noheader,
)

# import from file_utils
from .file_utils import get_file, validate_file

# import from generic_utils
from .generic_utils import Progbar

# import from parsing_utils
from .parsing_utils import ( ArgumentStruct,
    finalize_parameters,
    check_flag_conflicts,
    parse_from_dictlist,
)

# import from helper_utils
from .helper_utils import ( fetch_file,
    set_up_logger,
    verify_path,
    str2bool,
    keras_default_config,
)

# import from data_preprocessing_utils
from .data_preprocessing_utils import (
    quantile_normalization,
    generate_cross_validation_partition,
)

# feature selection
from .feature_selection_utils import (
    select_features_by_missing_values,
    select_features_by_variation,
    select_decorrelated_features,
)

# import from viz_utils
from .viz_utils import plot_history
from .viz_utils import plot_scatter
from .viz_utils import plot_array
from .viz_utils import plot_density_observed_vs_predicted
from .viz_utils import plot_2d_density_sigma_vs_error
from .viz_utils import plot_histogram_error_per_sigma
from .viz_utils import plot_decile_predictions
from .viz_utils import plot_calibration_interpolation
from .viz_utils import plot_calibrated_std
from .viz_utils import plot_contamination

# Milestone 16 specific
from .P1_utils import (
    coxen_single_drug_gene_selection,
    coxen_multi_drug_gene_selection,
    generate_gene_set_data,
    combat_batch_effect_removal,
)

# import framework dependent utils
import sys
try:
    import tensorflow
except ImportError:
    pass

if 'tensorflow' in sys.modules:
    print('Importing candle utils for keras')

    from .keras_utils import (
        build_initializer,
        build_optimizer,
        get_function,
        set_seed,
        set_parallelism_threads,
        PermanentDropout,
        register_permanent_dropout,
        LoggingCallback,
        r2,
        mae,
        mse,
        compute_trainable_params,
        TerminateOnTimeOut,
    )

    from .viz_utils import plot_metrics

    from .clr_keras_utils import (
        CyclicLR,
        clr_check_args,
        clr_set_args,
        clr_callback,
    )

    from ckpt_keras_utils import MultiGPUCheckpoint
    from ckpt_keras_utils import CandleCheckpointCallback
    from ckpt_keras_utils import restart

elif 'torch' in sys.modules:
    print('Importing candle utils for pytorch')
    from .pytorch_utils import (
        set_seed,
        build_optimizer,
        build_activation,
        get_function,
        initialize,
        xent,
        mse,
        set_parallelism_threads,
    )

else:
    raise Exception('No backend has been specified.')
