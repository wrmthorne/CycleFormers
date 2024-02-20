from pytorch_lightning import LightningModule
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, Trainer
from transformers.optimization import get_scheduler

from ..cycles import CausalCycle, Seq2SeqCycle
from ..core import TrainCycle
from ..trainer import ModelConfig

class CycleSequence:
    def __init__(self, *args):
        self.modules = args

    def __call__(self, input):
        for module in self.modules:
            input = module(input)
        return input
    
    def __iter__(self):
        return iter(self.modules)
    
    def __add__(self, other):
        if isinstance(other, CycleSequence) or isinstance(other, list):
            for module in other:
                self.modules.append(module)

        elif callable(other):
            self.modules.append(other)
    

class CycleModel(LightningModule):
    def __init__(
        self,
        model_A_config: ModelConfig,
        model_B_config: ModelConfig,
    ):
        super().__init__()
        self.automatic_optimization = False

        if not isinstance(model_A_config, ModelConfig):
            raise ValueError(f"model_config_A must be a ModelConfig, got {type(model_A_config)}")
        if not isinstance(model_B_config, ModelConfig):
            raise ValueError(f"model_config_B must be a ModelConfig, got {type(model_B_config)}")
        
        self.model_A_config = model_A_config
        self.model_B_config = model_B_config

        self.model_A, self.tokenizer_A = self.load_model(model_A_config)
        self.model_B, self.tokenizer_B = self.load_model(model_B_config)

        self.skip_reencode = False
        if self.tokenizer_A.get_vocab() == self.tokenizer_B.get_vocab() and type(self.tokenizer_A) == type(self.tokenizer_A):
            self.skip_reencode = True
            print('Same model and tokenizer detected. Using fast cycle. This can be manually disabled in the lightning config.')

        self.cycle_A = self.init_cycle(self.model_A, self.tokenizer_A, model_A_config, self.model_B, self.tokenizer_B)
        self.cycle_B = self.init_cycle(self.model_B, self.tokenizer_B, model_B_config, self.model_A, self.tokenizer_A)
        
    def load_model(self, config):
        model_config = AutoConfig.from_pretrained(config.pretrained_model_name_or_path)
        if getattr(model_config, 'is_encoder_decoder', False):
            model = AutoModelForSeq2SeqLM.from_pretrained(config.pretrained_model_name_or_path, config=model_config, **config)
        else:
            model = AutoModelForCausalLM.from_pretrained(config.pretrained_model_name_or_path, config=model_config, **config)

        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path, **config)
        return model, tokenizer
    
    def init_cycle(self, gen_model, gen_tokenizer, gen_config, train_model, train_tokenizer):
        gen_cycle = Seq2SeqCycle if train_model.config.is_encoder_decoder else CausalCycle
        train_cycle = Seq2SeqCycle if train_model.config.is_encoder_decoder else CausalCycle

        cycle_stages = CycleSequence(gen_cycle.generate(gen_model, gen_tokenizer, gen_config.generation_config))

        if not self.skip_reencode:
            cycle_stages.append([gen_cycle.decode(gen_tokenizer), train_cycle.encode(train_tokenizer)])

        cycle_stages.append([train_cycle.format(train_model, train_tokenizer), train_cycle.train(train_model)])

        return cycle_stages

    def configure_optimizers(self):
        optimisers = []
        schedulers = []

        for config, model in zip([self.model_A_config, self.model_B_config], [self.model_A, self.model_B]):
            optim, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(config)
            optimisers = optim(model.parameters(), **optim_kwargs)

            schedulers.append(get_scheduler(
                config.lr_scheduler_type,
                optimizer=optimisers,
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

    

class CycleModelOld:
    def __init__(
        self,
        pretrained_model_a_name_or_path: str,
        pretrained_model_b_name_or_path: str,
        model_a_args = {},
        model_b_args = {},
    ):
        if isinstance(model_a_args, ModelConfig):
            model_a_args = model_a_args.to_dict()
        elif not isinstance(model_a_args, dict):
            raise ValueError(f"model_a_args must be a dict or ModelConfig, got {type(model_a_args)}")
        if isinstance(model_b_args, ModelConfig):
            model_b_args = model_b_args.to_dict()
        elif not isinstance(model_b_args, dict):
            raise ValueError(f"model_b_args must be a dict or ModelConfig, got {type(model_b_args)}")
        
        self.model_a, self.tokenizer_a = self._load_model(pretrained_model_a_name_or_path, model_a_args)
        self.model_b, self.tokenizer_b = self._load_model(pretrained_model_b_name_or_path, model_b_args)

        self.cycle_a = self.init_cycle(self.model_a, self.tokenizer_a, self.model_b, self.tokenizer_b)
        self.cycle_b = self.init_cycle(self.model_b, self.tokenizer_b, self.model_a, self.tokenizer_a)

        

    def _load_model(self, pretrained_model_name_or_path, model_args):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        if config.is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path, config=config, **model_args)
        else:
            model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, config=config, **model_args)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **model_args)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer
    
    def init_cycle(self, train_model, train_tokenizer, gen_model, gen_tokenizer):
        gen_cycle = Seq2SeqCycle if train_model.config.is_encoder_decoder else CausalCycle
        train_cycle = Seq2SeqCycle if train_model.config.is_encoder_decoder else CausalCycle

        # TODO Add in sequences for shortcuts e.g. when model is the same for both cycles or tokenizer matches to save decode encode overhead
        return CycleSequence(
            gen_cycle.generate(train_model, train_tokenizer, {}),
            gen_cycle.decode(gen_tokenizer),
            train_cycle.encode(train_tokenizer),
            train_cycle.format(train_model, train_tokenizer),
            train_cycle.train(train_model),
        )