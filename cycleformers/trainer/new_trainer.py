from collections import OrderedDict
import logging
from typing import Dict, List, Optional, Tuple, Union

from datasets import Dataset
from lightning.pytorch.utilities import CombinedLoader
from pytorch_lightning import LightningModule, Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    DataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    Trainer as HFTrainer,
)
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from cycleformers.cycles import CausalCycle, Seq2SeqCycle
from cycleformers.cycles.cycle_utils import CycleSequence
from ..import_utils import is_peft_available
from .model_config import ModelConfig
from .trainer_config import TrainerConfig
from .trainer_utils import (
    get_parameter_names,
    validate_collator_new,
    validate_tokenizer,
)

if is_peft_available():
    from peft import PeftModel


logger = logging.get_logger(__name__)


# TODO: Add support for multi-adapter
def init_cycle(self, gen_model, gen_tokenizer, gen_config, train_model, train_tokenizer, train_config):
    gen_cycle = Seq2SeqCycle if getattr(gen_config, 'is_encoder_decoder', False) else CausalCycle
    train_cycle = Seq2SeqCycle if getattr(train_config, 'is_encoder_decoder', False) else CausalCycle

    cycle_stages = CycleSequence(OrderedDict({
        'Generate Synthetic IDs': gen_cycle.generate(gen_model, gen_tokenizer, gen_config.generation_config),
    }))

    if not self.skip_reencode:
        cycle_stages.extend(OrderedDict({
            'Decode Synthetic IDs to Text': gen_cycle.decode(gen_tokenizer),
            'Encode Synthetic Text to Train IDs': train_cycle.encode(train_tokenizer)
        }))

    cycle_stages.extend(OrderedDict({
        'Format Synthetic Train IDs': train_cycle.format(train_model, train_tokenizer),
        'Calculate Train Model Reconstruction Loss': train_cycle.train(train_model)
    }))

    return cycle_stages


class CycleTrainer(Trainer):
    '''
    CycleTrainer is a wrapper class for the PyTorch Lightning Trainer class, specifically designed to
    train models using cycle consistency training.

    Args:
        models (`dict` or `PeftModel`, *required*):
            The models to be trained. If a dictionary is supplied, the keys should be the names of the
            models and the values should be the models themselves. If a PeftModel is supplied, there must
            be at least two attached adapters and the model will be trained using those adapters.
        tokenizers (`dict` or `PreTrainedTokenizerBase`, *required*):
            The tokenizers to be used for each model. If a dictionary is supplied, the keys should be the
            names of the models and the values should be the tokenizers themselves. If a single tokenizer
            is supplied, it will be used for all models.
        args (`TrainerConfig`, *optional*):
            General, non model-specific training arguments. If not supplied, default arguments will be used.
        model_args (`dict`, *optional*):
            The model arguments to be used for training each model. The keys should be the names of the models
            and the values should be the model arguments themselves. If a single set of model arguments is
            supplied, it will be used for all models. If no model arguments are supplied for a specific model,
            the default model arguments will be used.
        data_collators (`dict` or `DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of the dataset. If a dictionary is
            supplied, the keys should be the names of the models and the values should be the data collators
            themselves. If a single data collator is supplied, it will be used for all models. If no collator
            is supplied for a specific model, a collator will be chosen based on the model type.
        train_datasets (`dict`, *optional*):
            The training datasets to be used for each model. The keys should be the names of the models and
            the values should be the datasets themselves.
        eval_datasets (`dict`, *optional*):
            The evaluation datasets to be used for each model. The keys should be the names of the models and
            the values should be the datasets themselves. A subset of models can be evaluated by supplying
            only a subset of model names.
        optimizers (`dict`, *optional*):
            The optimizers to be used for each model. The keys should be the names of the models and the
            values should be a tuple of the optimizer and the learning rate scheduler. Will default to
            AdamW and a linear learning rate scheduler if nothing is supplied for any specific model.
    '''
    def __init__(
        self,
        models: Union[Dict[str, Union[PreTrainedModel, nn.Module]], 'PeftModel'],
        tokenizers: Union[Dict[str, PreTrainedTokenizerBase], PreTrainedTokenizerBase],
        args: Optional[TrainerConfig] = None,
        model_args: Optional[Union[Dict[str, ModelConfig], ModelConfig]] = dict(),
        data_collators: Optional[Union[Dict[str, DataCollator], DataCollator]] = dict(),
        train_datasets: Optional[Dict[str, Dataset]] = dict(),
        eval_datasets: Optional[Dict[str, Dataset]] = None,
        # model_init                    TODO: implement later
        # compute_metrics               TODO: implement later
        # callbacks                     TODO: implement later
        optimizers: Dict[Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]] = dict(),
        # preprocess_logits_for_metrics TODO: implement later
    ) -> None:
        # TODO: How to handle training args for each model separately?
        if args is None:
            args = TrainerConfig()
        self.args = args

        # Validate models
        if isinstance(models, dict):
            if len(models) < 2:
                raise ValueError(f'Expected at least 2 models, got {len(models)}')
            
            self.model_names = list(models.keys())
        else:
            if not is_peft_available():
                raise ValueError(
                    'You have supplied only one model but PEFT is not installed. Please install '
                    'PEFT to use a single model with multiple adapters.'
                )

            if not isinstance(models, PeftModel):
                raise ValueError(
                    'Single models must be of type PeftModel with 2 or more attached adapters. '
                    f'Instead got {type(models)}. To use multiple models, please supply a dictionary '
                    'of models.'
                )
            
            if (num_adapters := len(models.peft_config.keys())) < 2:
                raise ValueError(f'Expected at least 2 adapters, got {num_adapters}')
            
            self.model_names = list(models.peft_config.keys())
            models = {name: models for name in self.model_names}

        self.models = models

        # Validate model configs
        if not isinstance(model_args, (dict, ModelConfig)):
            raise ValueError(
                f'Expected model_args to be a dictionary with keys matching model names '
                f'{self.model_names}. Instead got {type(model_args)}')
        
        if isinstance(model_args, dict):
            if any(name not in self.model_names for name in model_args.keys()):
                raise ValueError(
                    f'Model args contains a key not found in models. Expected keys to be a subset of '
                    f'{self.model_names}, got {list(model_args.keys())}.'
                )
            
            for name in self.model_names:
                if name not in model_args:
                    model_args[name] = ModelConfig()
                elif not isinstance(model_args[name], ModelConfig):
                    raise ValueError(
                        f'Expected model_args to be a ModelConfig, got {type(model_args[name])}.'
                    )
                
            else:
                model_args = {name: model_args for name in self.model_names}
            
        self.model_args = model_args

        # Tokenizer validation
        self.tokenizers = tokenizers
        self.create_tokenizers()
        
        # Validate collators
        self.data_collators = data_collators
        self.create_data_collators()
                
        # Validate datasets
        if not isinstance(train_datasets, dict):
            raise ValueError(f'Expected train_datasets to be a dictionary, got {type(train_datasets)}')
        
        if self.model_names != list(train_datasets.keys()):
            raise ValueError(
                f'Expected train_datasets to have keys {self.model_names}, got '
                f'{list(train_datasets.keys())}. Please ensure that the keys in train_datasets '
                'match the keys in models.'
            )
        
        for name, dataset in train_datasets.items():
            if dataset is None:
                raise ValueError(f'Training dataset for model {name} is None. Please supply a dataset.')
            
        self.train_datasets = train_datasets

        if not isinstance(eval_datasets, dict):
            raise ValueError(f'Expected eval_datasets to be a dictionary, got {type(eval_datasets)}')
        
        # Can have eval datasets for subset of models
        if any(name not in self.model_names for name in eval_datasets.keys()):
            raise ValueError(
                f'Eval datasets contains a key not found in models. Expected keys to be a subset of '
                f'{self.model_names}, got {list(eval_datasets.keys())}.'
            )

        # Validate optimizers
        self.optimizers = optimizers
        self.create_optimizers_and_schedulers()

    def __post_init__(self):
        '''
        Currently adding cycle consistency training specific attributes here. May be moved in the future
        to enable arbitrary multi-model training.
        '''
        if len(self.model_names) != 2:
            raise NotImplementedError('Training with more than 2 models is not yet supported.')
        
        # TODO: Add support for multi-adapter
        # TODO: Add support for 3+ models
        name_A, name_B = self.model_names
        self.models = {
            name_A: init_cycle(
                self.models[name_A], self.tokenizers[name_A], self.model_args[name_A],
                self.models[name_B], self.tokenizers[name_B], self.model_args[name_B]
            ),
            name_B: init_cycle(
                self.models[name_B], self.tokenizers[name_B], self.model_args[name_B],
                self.models[name_A], self.tokenizers[name_A], self.model_args[name_A]
            )
        }

    def create_tokenizers(self) -> None:
        '''
        Method to create and validate the tokenizers for the models.
        '''
        if not isinstance(self.tokenizers, (dict, PreTrainedTokenizerBase)):
            raise ValueError(
                'Expected tokenizers to be a dictionary of tokenizers or a PreTrainedTokenizerBase, got '
                f'{type(self.tokenizers)}.')
        
        if isinstance(self.tokenizers, dict):
            if self.model_names != list(self.tokenizers.keys()):
                raise ValueError(
                    f'Expected tokenizers to have keys {self.model_names}, got '
                    f'{list(self.tokenizers.keys())}. Please ensure that the keys in tokenizers '
                    'match the keys in models.'
                )  
        else:
            self.tokenizers = {name: self.tokenizers for name in self.model_names}

        # Check to see if tokenizers are correct for model type
        for name, tokenizer in self.tokenizers.items():
            validate_tokenizer(tokenizer, self.models[name])

    def create_data_collators(self) -> None:
        '''
        Method to create and validate the data collators for the models.
        '''
        if isinstance(self.data_collators, dict):
            if any(name not in self.model_names for name in self.data_collators.keys()):
                raise ValueError(
                    f'Data collators contains a key not found in models. Expected keys to be a subset of '
                    f'{self.model_names}, got {list(self.data_collators.keys())}.'
                )
            
            for name in self.model_names:
                if name not in self.data_collators:
                    if self.models[name].config.is_encoder_decoder:
                        self.data_collators[name] = DataCollatorForSeq2Seq(self.tokenizers[name])
                    else:
                        self.data_collators[name] = DataCollatorForLanguageModeling(self.tokenizers[name], mlm=False)

        else:
            self.data_collators = {name: self.data_collators for name in self.model_names}

        # Check to see if collators are correct for model type
        for name, collator in self.data_collators.items():
            validate_collator_new(collator, self.models[name])
            

    def get_train_dataloader(self) -> DataLoader:
        '''
        Method to create the training dataloader for the models.
        
        Returns:
            CombinedLoader: The combined dataloader for the all models.
        '''
        if self.train_datasets is None:
            raise ValueError('No training datasets found. Please supply training datasets.')
        
        data_loaders = {}
        for name, dataset in self.train_datasets.items():
            data_loaders[name] = DataLoader(
                dataset,
                batch_size=self.model_args[name].per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.data_collators[name]
            )

        return CombinedLoader(data_loaders, 'max_size')
    

    def get_eval_dataloader(self, eval_datasets: Optional[Dict[str, Dataset]] = None) -> DataLoader:
        '''
        Method to create the evaluation dataloader for the models.
        
        Args:
            eval_datasets (`dict`, *optional*):
                The evaluation datasets to be used for each model. The keys should be the names of the models and
                the values should be the datasets themselves. A subset of models can be evaluated by supplying
                only a subset of model names.
                
        Returns:
            CombinedLoader: The combined dataloader for the all models.
        '''
        if (len(eval_datasets) == 0 or eval_datasets is None) and \
           (len(self.eval_datasets) == 0 or self.eval_datasets is None):
            raise ValueError('No evaluation datasets found. Please supply evaluation datasets.')
        
        eval_datasets = eval_datasets if eval_datasets is not None else self.eval_datasets

        data_loaders = {}
        for name, dataset in eval_datasets.items():
            data_loaders[name] = DataLoader(
                dataset,
                batch_size=self.model_args[name].per_device_eval_batch_size,
                shuffle=True,
                collate_fn=self.data_collators[name]
            )

        return CombinedLoader(data_loaders, 'max_size')
    
    # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
    def get_decay_parameter_names(self, model) -> List[str]:
        '''
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        '''
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if 'bias' not in name]
        return decay_parameters
    

    def create_optimizers_and_schedulers(self):
        '''
        Method to create the optimizers and learning rate schedulers for the models.
        '''
        if any(name not in self.model_names for name in self.optimizers.keys()):
            raise ValueError(
                f'Optimizers contains a key not found in models. Expected keys to be a subset of '
                f'{self.model_names}, got {list(self.optimizers.keys())}.'
            )

        for name, model in self.models.items():
            if self.optimizers[name] == (None, None):
                decay_parameters = self.get_decay_parameter_names(model)
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.model_args[name].weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                optimizer_cls, optimizer_kwargs = HFTrainer.get_optimizer_cls_and_kwargs(self.self.model_args[name])
                optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                scheduler = get_scheduler(
                    self.self.model_args[name].lr_scheduler_type,
                    optimizer=optimizer, 
                    num_warmup_steps=self.self.model_args[name].get_warmup_steps(-1),
                    num_training_steps=self.self.model_args[name].max_steps,
                    scheduler_specific_kwargs=self.self.model_args[name].lr_scheduler_kwargs,
                )

                self.optimizers[name] = (optimizer, scheduler)

    def train(self):
        '''
        Main training entry point.

        MORE COMPATABILITY STUFF WILL BE ADDED HERE EVENTUALLY
        '''

        return self._inner_training_loop()
    
    def _inner_training_loop(self):
        '''
        Inner training loop for the CycleTrainer.

        MORE COMPATABILITY STUFF WILL BE ADDED HERE EVENTUALLY
        '''
        
        train_dataloader = self.get_train_dataloader()

        # Train!
        logger.info("***** Running training *****")

        steps_trained_in_current_epoch = 0

        total_batched_samples = 0
        for epoch in range(self.args.num_train_epochs):
            epoch_iterator = train_dataloader

            steps_in_epoch = len(epoch_iterator)

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                for name in self.model_names:
                    model = self.models[name]
                    model_inputs = inputs[name]

                    if model_inputs is None:
                        continue

                    tr_loss_step = self.training_step(model, model_inputs) / self.model_args[name].gradient_accumulation_steps

                    if total_batched_samples % self.model_args[name].gradient_accumulation_steps == 0:
                        self.optimizers[name].step()
                        self.lr_scheduler[name].step()

                    model.zero_grad()


    def training_step(self, model, inputs):
        model = model

        loss = self.compute_loss(model, inputs)
        loss.backward()

        return loss.detach()



    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
        

        