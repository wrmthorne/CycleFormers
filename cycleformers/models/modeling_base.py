from huggingface_hub import hf_hub_download
from peft import (
    PeftConfig,
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSeq2SeqLM,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

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


class CycleModel:
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