from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    LoraLayer,
)
import torch
import BitsAndBytes as bnb


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_model(hparams):
    if hparams.task.upper() == 'CAUSAL_LM':
        AutoModel = AutoModelForCausalLM
    elif hparams.task.upper() == 'SEQ2SEQ_LM':
        AutoModel = AutoModelForSeq2SeqLM

    compute_dtype = (torch.float16 if hparams.fp16 else (torch.bfloat16 if hparams.bf16 else torch.float32))

    model = AutoModel.from_pretrained(
        hparams.model_name_or_path,
        load_in_4bit=hparams.bits == 4,
        load_in_8bit=hparams.bits == 8,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=hparams.bits == 4,
            load_in_8bit=hparams.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=hparams.double_quant,
            bnb_4bit_quant_type=hparams.quant_type # {'fp4', 'nf4'}
        ),
        torch_dtype=(torch.float32 if hparams.fp16 else (torch.bfloat16 if hparams.bf16 else torch.float32)))
    
    if hparams.modules is None:
        hparams.modules = find_all_linear_names(hparams, model)

    if not hparams.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=hparams.gradient_checkpointing)

    config = LoraConfig(
        r=hparams.lora_r,
        lora_alpha=hparams.lora_alpha,
        target_modules=hparams.modules,
        lora_dropout=hparams.lora_dropout,
        bias="none" if not hparams.lora_bias else hparams.lora_bias,
        task_type="CAUSAL_LM" if hparams.task == 'causal_lm' else "SEQ2SEQ_LM",
    )
    
    if not hparams.full_finetune:
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if hparams.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if hparams.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    return model