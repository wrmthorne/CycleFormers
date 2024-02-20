from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Iterable, Optional, Union

from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.plugins import _PLUGIN_INPUT
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
    _PRECISION_INPUT,
)

@dataclass
class TrainerConfig:
    accelerator: Union[str, Accelerator] = field(
        default='auto',
        metadata={'help': 'Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto") as well as custom accelerator instances.'}
    )
    strategy: Union[str, Strategy] = field(
        default='auto',
        metadata={'help': 'Supports different training strategies with aliases as well custom strategies. Default: "auto".'}
    )
    devices: Union[List[int], str, int] = field(
        default='auto',
        metadata={'help': 'The devices to use. Can be set to a positive number (int or str), a sequence of device indices (list or str), the value -1 to indicate all available devices should be used, or "auto" for automatic selection based on the chosen accelerator. Default: "auto".'}
    )
    num_nodes: int = field(
        default=1,
        metadata={'help': 'Number of GPU nodes for distributed training. Default: 1.'}
    )
    precision: Optional[_PRECISION_INPUT] = field(
        default=None,
        metadata={'help': 'Double precision (64, "64" or "64-true"), full precision (32, "32" or "32-true"), 16bit mixed precision (16, "16", "16-mixed") or bfloat16 mixed precision ("bf16", "bf16-mixed"). Can be used on CPU, GPU, TPUs, HPUs or IPUs. Default: "32-true".'}
    )
    logger: Optional[Union[Logger, Iterable[Logger], bool]] = field(
        default=None,
        metadata={'help': 'Logger (or iterable collection of loggers) for experiment tracking. A True value uses the default TensorBoardLogger if it is installed, otherwise CSVLogger. False will disable logging. If multiple loggers are provided, local files (checkpoints, profiler traces, etc.) are saved in the log_dir of the first logger. Default: True.'}
    )
    callbacks: Optional[Union[List[Callback], Callback]] = field(
        default=None,
        metadata={'help': 'Add a callback or list of callbacks. Default: None.'}
    )
    fast_dev_run: Union[int, bool] = field(
        default=False,
        metadata={'help': 'Runs n if set to n (int) else 1 if set to True batch(es) of train, val and test to find any bugs (ie: a sort of unit test). Default: False.'}
    )
    max_epochs: Optional[int] = field(
        default=None,
        metadata={'help': 'Stop training once this number of epochs is reached. Disabled by default (None). If both max_epochs and max_steps are not specified, defaults to max_epochs = 1000. To enable infinite training, set max_epochs = -1.'}
    )
    min_epochs: Optional[int] = field(
        default=None,
        metadata={'help': 'Force training for at least these many epochs. Disabled by default (None).'}
    )
    max_steps: int = field(
        default=-1,
        metadata={'help': 'Stop training after this number of steps. Disabled by default (-1). If max_steps = -1 and max_epochs = None, will default to max_epochs = 1000. To enable infinite training, set max_epochs to -1.'}
    )
    min_steps: Optional[int] = field(
        default=None,
        metadata={'help': 'Force training for at least these number of steps. Disabled by default (None).'}
    )
    max_time: Optional[Union[str, timedelta, Dict[str, int]]] = field(
        default=None,
        metadata={'help': 'Stop training after this amount of time has passed. Disabled by default (None). The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a datetime.timedelta, or a dictionary with keys that will be passed to datetime.timedelta.'}
    )
    limit_train_batches: Optional[Union[int, float]] = field(
        default=None,
        metadata={'help': 'How much of training dataset to check (float = fraction, int = num_batches). Default: 1.0.'}
    )
    limit_val_batches: Optional[Union[int, float]] = field(
        default=None,
        metadata={'help': 'How much of validation dataset to check (float = fraction, int = num_batches). Default: 1.0.'}
    )
    limit_test_batches: Optional[Union[int, float]] = field(
        default=None,
        metadata={'help': 'How much of test dataset to check (float = fraction, int = num_batches). Default: 1.0.'}
    )
    limit_predict_batches: Optional[Union[int, float]] = field(
        default=None,
        metadata={'help': 'How much of prediction dataset to check (float = fraction, int = num_batches). Default: 1.0.'}
    )
    overfit_batches: Union[int, float] = field(
        default=0.0,
        metadata={'help': 'Overfit a fraction of training/validation data (float) or a set number of batches (int). Default: 0.0.'}
    )
    val_check_interval: Optional[Union[int, float]] = field(
        default=None,
        metadata={'help': 'How often to check the validation set. Pass a float in the range [0.0, 1.0] to check after a fraction of the training epoch. Pass an int to check after a fixed number of training batches. An int value can only be higher than the number of training batches when check_val_every_n_epoch=None, which validates after every N training batches across epochs or during iteration-based training. Default: 1.0.'}
    )
    check_val_every_n_epoch: Optional[int] = field(
        default=1,
        metadata={'help': 'Perform a validation loop every after every `N` training epochs. If None, validation will be done solely based on the number of training batches, requiring val_check_interval to be an integer value. Default: 1.'}
    )
    num_sanity_val_steps: Optional[int] = field(
        default=None,
        metadata={'help': 'Sanity check runs n validation batches before starting the training routine. Set it to -1 to run all batches in all validation dataloaders. Default: 2.'}
    )
    log_every_n_steps: Optional[int] = field(
        default=None,
        metadata={'help': 'How often to log within steps. Default: 50.'}
    )
    enable_checkpointing: Optional[bool] = field(
        default=None,
        metadata={'help': 'If True, enable checkpointing. It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in Trainer.callbacks. Default: True.'}
    )
    enable_progress_bar: Optional[bool] = field(
        default=None,
        metadata={'help': 'Whether to enable to progress bar by default. Default: True.'}
    )
    enable_model_summary: Optional[bool] = field(
        default=None,
        metadata={'help': 'Whether to enable model summarization by default. Default: True.'}
    )
    # ====================
    accumulate_grad_batches: int = field(
        default=1,
        metadata={'help': 'Accumulates gradients over k batches before stepping the optimizer. Default: 1.'}
    )
    gradient_clip_val: Optional[Union[int, float]] = field(
        default=None,
        metadata={'help': 'The value at which to clip gradients. Passing gradient_clip_val=None disables gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before. Default: None.'}
    )
    gradient_clip_algorithm: Optional[str] = field(
        default=None,
        metadata={'help': 'The gradient clipping algorithm to use. Pass gradient_clip_algorithm="value" to clip by value, and gradient_clip_algorithm="norm" to clip by norm. By default it will be set to "norm".'}
    )
    # ===================
    deterministic: Optional[Union[bool, _LITERAL_WARN]] = field(
        default=None,
        metadata={'help': 'If True, sets whether PyTorch operations must use deterministic algorithms. Set to "warn" to use deterministic algorithms whenever possible, throwing warnings on operations that don\'t support deterministic mode. If not set, defaults to False. Default: None.'}
    )
    benchmark: Optional[bool] = field(
        default=None,
        metadata={'help': 'The value (True or False) to set torch.backends.cudnn.benchmark to. The value for torch.backends.cudnn.benchmark set in the current session will be used (False if not manually set). If Trainer.deterministic is set to True, this will default to False. Override to manually set a different value. Default: None.'}
    )
    inference_mode: bool = field(
        default=True,
        metadata={'help': 'Whether to use torch.inference_mode or torch.no_grad during evaluation (validate/test/predict).'}
    )
    use_distributed_sampler: bool = field(
        default=True,
        metadata={'help': 'Whether to wrap the DataLoader"s sampler with torch.utils.data.DistributedSampler. If not specified this is toggled automatically for strategies that require it. By default, it will add shuffle=True for the train sampler and shuffle=False for validation/test/predict samplers. If you want to disable this logic, you can pass False and add your own distributed sampler in the dataloader hooks. If True and a distributed sampler was already added, Lightning will not replace the existing one. For iterable-style datasets, we don"t do this automatically.'}
    )
    profiler: Optional[Union[Profiler, str]] = field(
        default=None,
        metadata={'help': 'To profile individual steps during training and assist in identifying bottlenecks. Default: None.'}
    )
    detect_anomaly: bool = field(
        default=False,
        metadata={'help': 'Enable anomaly detection for the autograd engine. Default: False.'}
    )
    barebones: bool = field(
        default=False,
        metadata={'help': 'Whether to run in "barebones mode", where all features that may impact raw speed are disabled. This is meant for analyzing the Trainer overhead and is discouraged during regular training runs. The following features are deactivated: enable_checkpointing, logger, enable_progress_bar, log_every_n_steps, enable_model_summary, num_sanity_val_steps, fast_dev_run, detect_anomaly, profiler, LightningModule.log, LightningModule.log_dict.'}
    )
    plugins: Optional[Union[_PLUGIN_INPUT, List[_PLUGIN_INPUT]]] = field(
        default=None,
        metadata={'help': 'Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins. Default: None.'}
    )
    sync_batchnorm: bool = field(
        default=False,
        metadata={'help': 'Synchronize batch norm layers between process groups/whole world. Default: False.'}
    )
    reload_dataloaders_every_n_epochs: int = field(
        default=0,
        metadata={'help': 'Set to a positive integer to reload dataloaders every n epochs. Default: 0.'}
    )
    default_root_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Default path for logs and weights when no logger/ckpt_callback passed. Default: os.getcwd(). Can be remote file paths such as s3://mybucket/path or hdfs://path/'}
    )

    def to_dict(self):
        return self.__dict__
            