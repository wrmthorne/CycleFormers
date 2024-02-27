from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

from datasets import Dataset
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import DataCollator, EvalPrediction, PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from transformers.utils.import_utils import is_peft_available

from cycleformers.cycles import CausalCycle, Seq2SeqCycle, CycleSequence
from .multi_model_trainer import MultiModelTrainer
from .training_args import TrainingArguments, ModelTrainingArguments

if is_peft_available():
    from peft import PeftModel


class CycleTrainer(MultiModelTrainer):
    def __init__(
        self,
        models: Union[Dict[str, Union[PreTrainedModel, nn.Module]], 'PeftModel'],
        tokenizers: Union[Dict[str, PreTrainedTokenizerBase], PreTrainedTokenizerBase],
        args: Optional[TrainingArguments] = None,
        model_args: Optional[Union[Dict[str, ModelTrainingArguments], ModelTrainingArguments]] = dict(),
        data_collators: Optional[Union[Dict[str, DataCollator], DataCollator]] = dict(),
        train_datasets: Optional[Dict[str, Dataset]] = dict(),
        eval_datasets: Optional[Dict[str, Dataset]] = None,
        model_init: Optional[Dict[str, Callable[[], PreTrainedModel]]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[Union[Dict[str, List[TrainerCallback]], List[TrainerCallback]]] = None,
        optimizers: Optional[Dict[str, Tuple[Optimizer, LambdaLR]]] = dict(),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            models=models,
            tokenizers=tokenizers,
            args=args,
            model_args=model_args,
            data_collators=data_collators,
            train_datasets=train_datasets,
            eval_datasets=eval_datasets,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if len(self.handlers) > 2:
            raise ValueError("CycleTrainer only supports two models")
        
        for i, name in enumerate(self._model_names):
            other_name = self._model_names[~i]

            # Other model becomes generator and this current model is trained
            self.handlers[name].cycle = self.init_cycle(
                self.handlers[other_name].model,
                self.handlers[other_name].tokenizer,
                {},
                self.handlers[name].model,
                self.handlers[name].tokenizer,
            )
        
    # TODO: Add support for multi-adapter
    def init_cycle(self, gen_model, gen_tokenizer, gen_model_generation_config, train_model, train_tokenizer):
        gen_cycle = Seq2SeqCycle if gen_model.config.is_encoder_decoder else CausalCycle
        train_cycle = Seq2SeqCycle if train_model.config.is_encoder_decoder else CausalCycle

        skip_reencode = gen_tokenizer.get_vocab() == train_tokenizer.get_vocab() and type(gen_tokenizer) == type(train_tokenizer)

        cycle_stages = CycleSequence(OrderedDict({
            'Generate Synthetic IDs': gen_cycle.generate(gen_model, gen_tokenizer, gen_model_generation_config),
        }))

        if not skip_reencode:
            cycle_stages.extend(OrderedDict({
                'Decode Synthetic IDs to Text': gen_cycle.decode(gen_tokenizer),
                'Encode Synthetic Text to Train IDs': train_cycle.encode(train_tokenizer)
            }))
        else:
            print('Skipping re-encoding')

        cycle_stages.extend(OrderedDict({
            'Format Synthetic Train IDs': train_cycle.format(train_model, train_tokenizer),
            'Calculate Train Model Reconstruction Loss': train_cycle.train(train_model)
        }))

        return cycle_stages
    
    def training_step(self, curr_handler, inputs, all_handlers):
        model = curr_handler.model

        model.train()
        inputs = curr_handler._prepare_inputs(inputs)

        with curr_handler.compute_loss_context_manager():
            loss = self.compute_loss(curr_handler.cycle, inputs)
            loss.backward()

        if curr_handler.args.n_gpu > 1:
            loss = loss.mean()

        return loss.detach() / curr_handler.args.gradient_accumulation_steps