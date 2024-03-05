from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

from datasets import Dataset
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import DataCollator, EvalPrediction, PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from transformers.utils.import_utils import is_apex_available, is_peft_available, is_sagemaker_mp_enabled

from cycleformers.cycles import CausalCycle, Seq2SeqCycle, CycleSequence
from .multi_model_trainer import MultiModelTrainer
from .training_args import TrainingArguments, ModelTrainingArguments

if is_apex_available():
    from apex import amp

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
        
    # TODO: Add support for multi-adapter
    def init_cycle(self, gen_model, gen_tokenizer, gen_model_generation_config, train_model, train_tokenizer, train_collator):
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

        cycle_stages.extend(OrderedDict({
            'Format Synthetic Train IDs': train_cycle.format(train_tokenizer, train_collator),
            'Calculate Train Model Reconstruction Loss': train_cycle.train(train_model)
        }))

        return cycle_stages
    
    def training_step(self, curr_handler, inputs, all_handlers):
        curr_handler._model.train()

        inputs = curr_handler._prepare_inputs(inputs)

        if not getattr(curr_handler, 'cycle', False):
            other_model = self._model_names[~self._model_names.index(curr_handler._name)]
            curr_handler.cycle = self.init_cycle(
                gen_model=all_handlers[other_model]._model,
                gen_tokenizer=all_handlers[other_model].tokenizer,
                gen_model_generation_config={},
                train_model=curr_handler._model,
                train_tokenizer=curr_handler.tokenizer,
                train_collator=self.data_collators[curr_handler._name]
            )

            print(curr_handler.cycle)

        with curr_handler.compute_loss_context_manager():
            loss = self.compute_loss(curr_handler.cycle, inputs)

        if curr_handler.args.n_gpu > 1:
            loss = loss.mean()

        if curr_handler.use_apex:
            with amp.scale_loss(loss, curr_handler.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            curr_handler.accelerator.backward(loss)

        return loss.detach() / curr_handler.args.gradient_accumulation_steps