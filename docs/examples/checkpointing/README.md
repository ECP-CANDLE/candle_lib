 **CANDLE Checkpoint/Restart Hyperparameters**

**Hyperparameters**

The following hyperparameters control the behavior of CANDLE's checkpoint/restart functionality:

* **ckpt_restart_mode** (string): Controls whether and how to restart from a previous checkpoint. Valid values are:
    * "off": Disables restart functionality.
    * "auto": Automatically attempts to restart from the most recent checkpoint if one is found.
    * "required": Fails if a checkpoint is not found to restart from.
    * Default: "auto"

* **ckpt_save_best** (boolean): If True, saves a checkpoint whenever the specified metric improves.
    * Default: True

* **ckpt_save_best_metric** (string): Specifies the metric to track for improvement when ckpt_save_best is True.
    * Default: "val_loss"

* **ckpt_skip_epochs** (integer): Specifies the number of initial epochs to skip before writing checkpoints.
    * Default: 0

* **ckpt_save_interval** (integer): Saves a checkpoint whenever epoch % save_interval == 0. Set to 0 to disable periodic checkpoints.
    * Default: 1 (save everything)

* **ckpt_checksum** (boolean): If True, computes a checksum for the model and stores it in the JSON file for integrity verification.
    * Default: False

* **ckpt_keep_mode** (string): Specifies the policy for keeping checkpoints. Valid values are:
    * "linear": Keeps a linear number of checkpoints, up to the limit.
    * "exponential": Keeps an exponential number of checkpoints (not yet implemented).
    * Default: "linear"

* **ckpt_keep_limit** (integer): Specifies the maximum number of checkpoints to keep, excluding the best checkpoint.
    * Default: 5

* **ckpt_directory** (string): Specifies the top-level directory for storing checkpoints.
    * Default: CANDLE parameter output_dir

* **ckpt_metadata** (string): Arbitrary string to include in the JSON file for additional information (e.g., job ID, hardware location).
    * Default: None

* **ckpt_best_metric_last** (float or None): Specifies the last best metric value to use after a restart.
    * Default: None

**Usage**

**Before training:**

1. Set the initial epoch to 0.
2. Attempt to restart from a checkpoint using `candle.restart(gParameters, model)`.
3. If a checkpoint is found, update the initial epoch with the epoch from the checkpoint.

**Set up a callback:**

1. Create a `CandleCkpt` callback using the specified hyperparameters.
2. Include the callback in the `model.fit` call.

**Optionally, log a final report:**

1. Use `ckpt.report_final()` to log a final checkpoint report.

**Controlling restart:**

* Most restart options are controlled through the `gParameters` dictionary.
* The `restart()` function typically uses a soft link named `ckpts/last` to determine the most recent checkpoint directory.
* To roll back to a previous checkpoint, simply re-link `ckpts/last` to the desired directory.

**Keep policy:**

* The `ckpt_keep` settings only apply to the current run and do not affect checkpoints from previous runs.
* Checkpoints from prior runs can be manually removed if desired.

**Logging:**

* A log of checkpoint operations is written to `ckpt_directory/ckpt.log`.
