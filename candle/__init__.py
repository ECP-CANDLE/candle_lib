name = "candle"
__version__ = '0.0.1'

# import from data_utils
from .data_utils import load_csv_data
from .data_utils import load_Xy_one_hot_data2
from .data_utils import load_Xy_data_noheader

# import from file_utils
from .file_utils import get_file
from .file_utils import validate_file

# import from generic_utils
from .generic_utils import Progbar

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


# import framework dependent utils
import sys
try:
    import tensorflow
except ImportError:
    pass

if 'tensorflow' in sys.modules:
    print('Importing candle utils for keras')

    from .viz_utils import plot_metrics

    from .clr_keras_utils import CyclicLR
    from .clr_keras_utils import clr_check_args
    from .clr_keras_utils import clr_set_args
    from .clr_keras_utils import clr_callback