name = "candle"
__version__ = "0.0.1"

# import framework dependent utils
import sys

from .benchmark_def import (
    Benchmark,
    create_params,
)

# import from data_preprocessing_utils
from .data_preprocessing_utils import (
    generate_cross_validation_partition,
    quantile_normalization,
)

# import from data_utils
from .data_utils import load_csv_data, load_Xy_data_noheader, load_Xy_one_hot_data2

# feature selection
from .feature_selection_utils import (
    select_decorrelated_features,
    select_features_by_missing_values,
    select_features_by_variation,
)

# import from file_utils
from .file_utils import get_file, validate_file

# import from generic_utils
from .generic_utils import Progbar

# import from helper_utils
from .helper_utils import (
    fetch_file,
    keras_default_config,
    set_up_logger,
    str2bool,
    verify_path,
)

# noise injection
from .noise_utils import (
    add_cluster_noise,
    add_column_noise,
    add_gaussian_noise,
    add_noise,
    label_flip,
    label_flip_correlated,
)

# Milestone 16 specific
from .P1_utils import (
    combat_batch_effect_removal,
    coxen_multi_drug_gene_selection,
    coxen_single_drug_gene_selection,
    generate_gene_set_data,
)

# import from parsing_utils
from .parsing_utils import (
    ArgumentStruct,
    check_flag_conflicts,
    finalize_parameters,
    parse_from_dictlist,
)

# import from uq_utils
from .uq_utils import (
    compute_empirical_calibration_interpolation,
    compute_statistics_heteroscedastic,
    compute_statistics_homoscedastic,
    compute_statistics_homoscedastic_summary,
    compute_statistics_quantile,
    generate_index_distribution,
    split_data_for_empirical_calibration,
)

# import from viz_utils
from .viz_utils import (
    plot_2d_density_sigma_vs_error,
    plot_array,
    plot_calibrated_std,
    plot_calibration_interpolation,
    plot_contamination,
    plot_decile_predictions,
    plot_density_observed_vs_predicted,
    plot_histogram_error_per_sigma,
    plot_history,
    plot_scatter,
)

try:
    import tensorflow
except ImportError:
    pass

try:
    import torch
except ImportError:
    pass

if "tensorflow" in sys.modules:
    print("Importing candle utils for keras")

    from .ckpt_keras_utils import CandleCkptKeras, MultiGPUCheckpoint
    from .ckpt_utils import CandleCkpt, ModelType
    from .clr_keras_utils import CyclicLR, clr_callback, clr_check_args, clr_set_args
    from .keras_utils import (
        CandleRemoteMonitor,
        LoggingCallback,
        PermanentDropout,
        TerminateOnTimeOut,
        build_initializer,
        build_optimizer,
        compute_trainable_params,
        get_function,
        mae,
        mse,
        r2,
        register_permanent_dropout,
        set_parallelism_threads,
        set_seed,
    )
    from .uq_keras_utils import (
        AbstentionAdapt_Callback,
        Contamination_Callback,
        abstention_acc_class_i_metric,
        abstention_acc_metric,
        abstention_class_i_metric,
        abstention_loss,
        abstention_metric,
        acc_class_i_metric,
        add_index_to_output,
        add_model_output,
        contamination_loss,
        heteroscedastic_loss,
        mae_contamination_metric,
        mae_heteroscedastic_metric,
        meanS_heteroscedastic_metric,
        modify_labels,
        mse_contamination_metric,
        mse_heteroscedastic_metric,
        quantile_loss,
        quantile_metric,
        r2_contamination_metric,
        r2_heteroscedastic_metric,
        sparse_abstention_acc_metric,
        sparse_abstention_loss,
        triple_quantile_loss,
    )
    from .viz_utils import plot_metrics

if "torch" in sys.modules:
    print("Importing candle utils for pytorch")
    from .ckpt_pytorch_utils import CandleCkptPyTorch
    from .pytorch_utils import (
        build_pytorch_activation,
        build_pytorch_optimizer,
        get_pytorch_function,
        pytorch_initialize,
        pytorch_mse,
        pytorch_xent,
        set_pytorch_seed,
        set_pytorch_threads,
    )

# else:
#     raise Exception("No backend has been specified.")


__all__ = [
    "Benchmark",
    "generate_cross_validation_partition",
    "quantile_normalization",
    "load_csv_data",
    "load_Xy_data_noheader",
    "load_Xy_one_hot_data2",
    "select_decorrelated_features",
    "select_features_by_missing_values",
    "select_features_by_variation",
    "get_file",
    "validate_file",
    "Progbar",
    "fetch_file",
    "keras_default_config",
    "set_up_logger",
    "str2bool",
    "verify_path",
    "add_cluster_noise",
    "add_column_noise",
    "add_gaussian_noise",
    "add_noise",
    "label_flip",
    "label_flip_correlated",
    "combat_batch_effect_removal",
    "coxen_multi_drug_gene_selection",
    "coxen_single_drug_gene_selection",
    "generate_gene_set_data",
    "ArgumentStruct",
    "check_flag_conflicts",
    "finalize_parameters",
    "parse_from_dictlist",
    "compute_empirical_calibration_interpolation",
    "compute_statistics_heteroscedastic",
    "compute_statistics_homoscedastic",
    "compute_statistics_homoscedastic_summary",
    "compute_statistics_quantile",
    "generate_index_distribution",
    "split_data_for_empirical_calibration",
    "plot_2d_density_sigma_vs_error",
    "plot_array",
    "plot_calibrated_std",
    "plot_calibration_interpolation",
    "plot_contamination",
    "plot_decile_predictions",
    "plot_density_observed_vs_predicted",
    "plot_histogram_error_per_sigma",
    "plot_history",
    "plot_scatter",
    # Keras/tensorflow
    "CandleCheckpointCallback",
    "MultiGPUCheckpoint",
    "restart",
    "CyclicLR",
    "clr_callback",
    "clr_check_args",
    "clr_set_args",
    "CandleRemoteMonitor",
    "LoggingCallback",
    "PermanentDropout",
    "TerminateOnTimeOut",
    "build_initializer",
    "build_optimizer",
    "compute_trainable_params",
    "get_function",
    "mae",
    "mse",
    "r2",
    "register_permanent_dropout",
    "set_parallelism_threads",
    "set_seed",
    "AbstentionAdapt_Callback",
    "Contamination_Callback",
    "abstention_acc_class_i_metric",
    "abstention_acc_metric",
    "abstention_class_i_metric",
    "abstention_loss",
    "abstention_metric",
    "acc_class_i_metric",
    "add_index_to_output",
    "add_model_output",
    "contamination_loss",
    "heteroscedastic_loss",
    "mae_contamination_metric",
    "mae_heteroscedastic_metric",
    "meanS_heteroscedastic_metric",
    "modify_labels",
    "mse_contamination_metric",
    "mse_heteroscedastic_metric",
    "quantile_loss",
    "quantile_metric",
    "r2_contamination_metric",
    "r2_heteroscedastic_metric",
    "sparse_abstention_acc_metric",
    "sparse_abstention_loss",
    "triple_quantile_loss",
    "plot_metrics",
    # pytorch
    "build_pytorch_activation",
    "build_pytorch_optimizer",
    "get_pytorch_function",
    "pytorch_initialize",
    "pytorch_mse",
    "pytorch_xent",
    "set_pytorch_seed",
    "set_pytorch_threads",
]
