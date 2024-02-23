from warnings import warn

from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, PreTrainedTokenizerBase


def has_length(dataset):
    try:
        return len(dataset) is not None
    except TypeError:
        return False

def validate_train_dataset(train_dataset, dataset_name, max_steps):
    if train_dataset is not None and not has_length(train_dataset) and max_steps <= 0:
        raise ValueError(
            f"{dataset_name} does not implement __len__, max_steps has to be specified. "
            "The number of steps needs to be known in advance for the learning rate scheduler."
        )

def prepare_data_collator(model, tokenizer, data_collator):
    if getattr(model, 'is_encoder_decoder', False):
        if isinstance(data_collator, DataCollatorForLanguageModeling):
            warn('Using DataCollatorForCausalLM for a Seq2Seq model. This might lead to unexpected behavior.')

        return DataCollatorForSeq2Seq(tokenizer)
    
    elif not getattr(model, 'is_encoder_decoder', False):
        if isinstance(data_collator, DataCollatorForSeq2Seq):
            warn('Using DataCollatorForSeq2Seq for a CausalLM model. This might lead to unexpected behavior.')

        return DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    return data_collator               

def validate_data_collator(data_collator, data_collator_name):
    if not callable(data_collator) and callable(getattr(data_collator, "collate_batch", None)):
        raise ValueError(f"{data_collator_name} should be a simple callable (function, class with `__call__`).")
    

def validate_collator_new(data_collator, model):
    # TODO: Add check for if data_collator is a valid collator 
    if model.config.is_encoder_decoder:
        if not isinstance(data_collator, DataCollatorForSeq2Seq):
            warn(
                f'Model {model} is an encoder-decoder model but the provided data_collator is not '
                f'of type DataCollatorForSeq2Seq. This may cause unintended behaviour '
                'during training.'
            )
    else:
        if not isinstance(data_collator, DataCollatorForLanguageModeling):
            warn(
                f'Model {model} is a causal model but the data_collator is not of type '
                    'DataCollatorForLanguageModeling. This may cause unintended behaviour '
                    'during training.'
            )

def validate_tokenizer(tokenizer, model):
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError(
            f'Expected tokenizer to be of type PreTrainedTokenizer, got {type(tokenizer)}.'
        )
    
    if tokenizer.name_or_path != model.config._name_or_path:
        warn(
            f'Tokenizer {tokenizer} does not match model {model}. This may cause unintended behaviour '
            'during training. You can ignore this warning if you have loaded a pretrained model and '
            'the base tokenizer for the model.'
        )


# https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py
def get_parameter_names(model, forbidden_layer_types):
    '''
    Returns the names of the model parameters that are not inside a forbidden layer.
    '''
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result