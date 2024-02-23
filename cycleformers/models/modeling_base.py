from collections import OrderedDict

from pytorch_lightning import LightningModule
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, Trainer
from transformers.optimization import get_scheduler

from cycleformers.cycles import CausalCycle, Seq2SeqCycle
from cycleformers.cycles.cycle_utils import CycleSequence
from cycleformers.core import TrainCycle
from cycleformers.trainer import ModelConfig
from cycleformers.import_utils import is_peft_available

if is_peft_available():
    from peft import (
        PeftConfig,
        PeftModel,
    )
    

class CycleModel(LightningModule):
    def __init__(
        self,
        model_A_config: ModelConfig,
        model_B_config: ModelConfig = None,
    ):
        super().__init__()
        self.automatic_optimization = False

        if not isinstance(model_A_config, ModelConfig):
            raise ValueError(f"model_config_A must be a ModelConfig, got {type(model_A_config)}")
        if model_B_config is not None and not isinstance(model_B_config, ModelConfig):
            raise ValueError(f"model_config_B must be a ModelConfig, got {type(model_B_config)}")
        
        self.model_A_config = model_A_config
        self.model_B_config = model_B_config if model_B_config is not None else model_A_config

        self.model_A, self.tokenizer_A = self._load_model(model_A_config)
        self.model_B, self.tokenizer_B = self._load_model(model_B_config)

        self.skip_reencode = False
        if self.tokenizer_A.get_vocab() == self.tokenizer_B.get_vocab() and type(self.tokenizer_A) == type(self.tokenizer_A):
            self.skip_reencode = True
            print('Same model and tokenizer detected. Using fast cycle. This can be manually disabled in the lightning config.')

        self.cycle_A = self._init_cycle(self.model_A, self.tokenizer_A, model_A_config, self.model_B, self.tokenizer_B, model_B_config)
        self.cycle_B = self._init_cycle(self.model_B, self.tokenizer_B, model_B_config, self.model_A, self.tokenizer_A, model_A_config)

        print(f'Cycle A: {self.cycle_A}')
        print(f'Cycle B: {self.cycle_B}')
        
    def _load_model(self, config):
        model_config = AutoConfig.from_pretrained(config.pretrained_model_name_or_path)
        if getattr(model_config, 'is_encoder_decoder', False):
            model = AutoModelForSeq2SeqLM.from_pretrained(config=model_config, **config.pretrained_model_kwargs())
        else:
            model = AutoModelForCausalLM.from_pretrained(config=model_config, **config.pretrained_model_kwargs())

        tokenizer = AutoTokenizer.from_pretrained(**config.pretrained_tokenizer_kwargs())
        return model, tokenizer
    
    def _init_cycle(self, gen_model, gen_tokenizer, gen_config, train_model, train_tokenizer, train_config):
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

    def configure_optimizers(self):
        optimisers = []
        schedulers = []

        for config, model in zip([self.model_A_config, self.model_B_config], [self.model_A, self.model_B]):
            optim, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(config)
            optimiser = optim(model.parameters(), **optim_kwargs)
            optimisers.append(optimiser)

            schedulers.append(get_scheduler(
                config.lr_scheduler_type,
                optimizer=optimiser,
                num_warmup_steps=config.get_warmup_steps(-1),
                num_training_steps=-1,
                scheduler_specific_kwargs=config.lr_scheduler_kwargs,
            ))
        
        # TODO updated schedulers later in case 
        return optimisers, schedulers
        
    def training_step(self, batch, batch_idx):
        opt_A, opt_B = self.optimizers()
        sch_A, sch_B = self.lr_schedulers()
        
        batch, _, _ = batch        

        if batch[TrainCycle.A]:
            loss_A = self.cycle_A(batch[TrainCycle.A]).loss
            self.manual_backward(loss_A)

            opt_A.step()
            sch_A.step()
            opt_A.zero_grad()

        if batch[TrainCycle.B]:
            loss_B = self.cycle_B(batch[TrainCycle.B]).loss
            self.manual_backward(loss_B)

            opt_B.step()
            sch_B.step()
            opt_B.zero_grad()

    def validation_step(self, batch, batch_idx):
        batch, _, _ = batch

        if batch[TrainCycle.A]:
            loss_A = self.model_A(
                input_ids=batch[TrainCycle.A]['input_ids'].to(self.model_A.device),
                attention_mask=batch[TrainCycle.A]['attention_mask'].to(self.model_A.device),
                labels=batch[TrainCycle.A]['labels'].to(self.model_A.device),
            ).loss
        if batch[TrainCycle.B]:
            loss_B = self.model_B(
                input_ids=batch[TrainCycle.B]['input_ids'].to(self.model_B.device),
                attention_mask=batch[TrainCycle.B]['attention_mask'].to(self.model_B.device),
                labels=batch[TrainCycle.B]['labels'].to(self.model_B.device),
            ).loss