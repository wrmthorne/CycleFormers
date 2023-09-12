from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelWithLMHead,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from peft.tuners.lora import LoraLayer
import torch
import bitsandbytes as bnb
from utils.data_collator import DataCollatorForCausalLM

def print_trainable_parameters(model_hparams, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    if model_hparams['bits'] == 4: trainable_params /= 2
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param}")

def find_all_linear_names(model_hparams, model):
    cls = bnb.nn.Linear4bit if model_hparams['bits'] == 4 else (bnb.nn.Linear8bitLt if model_hparams['bits'] == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_model(model_hparams):
    compute_dtype = (torch.float16 if model_hparams['fp16'] else (torch.bfloat16 if model_hparams['bf16'] else torch.float32))

    if model_hparams['full_finetune']: assert model_hparams['bits'] in [16, 32]

    model = AutoModelWithLMHead.from_pretrained(
        model_hparams['model_name_or_path'],
        load_in_4bit=model_hparams['bits'] == 4,
        load_in_8bit=model_hparams['bits'] == 8,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=model_hparams['bits'] == 4,
            load_in_8bit=model_hparams['bits'] == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_hparams['double_quant'],
            bnb_4bit_quant_type=model_hparams['quant_type'] # {'fp4', 'nf4'}
        ),
        torch_dtype=(torch.float32 if model_hparams['fp16'] else (torch.bfloat16 if model_hparams['bf16'] else torch.float32)))
    
    tokenizer = AutoTokenizer.from_pretrained(model_hparams['model_name_or_path'])
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    if model.__class__.__name__ in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES.values() or model_hparams['task'] == 'SEQ2SEQ_LM':
        model_hparams['task'] = 'SEQ2SEQ_LM'
        collator = DataCollatorForSeq2Seq(tokenizer, return_tensors='pt', padding=True)
    elif model.__class__.__name__ in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values() or model_hparams['task'] == 'CAUSAL_LM':
        model_hparams['task'] = 'CAUSAL_LM'
        collator = DataCollatorForCausalLM(tokenizer, return_tensors='pt', padding=True)
        tokenizer.padding_side = 'left'
    else:
        raise ValueError('Model is not a seq2seq or causal LM model')
    
    if model_hparams['modules'] is None:
        model_hparams['modules'] = find_all_linear_names(model_hparams, model)

    if not model_hparams['full_finetune']:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=model_hparams['gradient_checkpointing'])
    if model_hparams['gradient_checkpointing']:
        model.gradient_checkpointing_enable()

    config = LoraConfig(
        r=model_hparams['lora_r'],
        lora_alpha=model_hparams['lora_alpha'],
        target_modules=model_hparams['modules'],
        lora_dropout=model_hparams['lora_dropout'],
        bias="none" if not model_hparams['lora_bias'] else model_hparams['lora_bias'],
        task_type=model_hparams['task'],
    )
    
    if not model_hparams['full_finetune']:
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if model_hparams['bf16']:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if model_hparams['bf16'] and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    print_trainable_parameters(model_hparams, model)
    return model, tokenizer, collator
