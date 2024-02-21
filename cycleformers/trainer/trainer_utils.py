import warnings

from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq


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
            warnings.warn('Using DataCollatorForCausalLM for a Seq2Seq model. This might lead to unexpected behavior.')

        return DataCollatorForSeq2Seq(tokenizer)
    
    elif not getattr(model, 'is_encoder_decoder', False):
        if isinstance(data_collator, DataCollatorForSeq2Seq):
            warnings.warn('Using DataCollatorForSeq2Seq for a CausalLM model. This might lead to unexpected behavior.')

        return DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    return data_collator               

def validate_data_collator(data_collator, data_collator_name):
    if not callable(data_collator) and callable(getattr(data_collator, "collate_batch", None)):
        raise ValueError(f"{data_collator_name} should be a simple callable (function, class with `__call__`).")