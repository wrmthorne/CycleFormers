from cycleformers import CycleTrainer, CycleModel, ModelConfig, TrainerConfig
from datasets import load_dataset, DatasetDict
from pytorch_lightning.loggers import WandbLogger
from transformers import GenerationConfig
from nltk.tokenize.treebank import TreebankWordDetokenizer



# Dictionary mapping BIO tags to compound tags and then to word based tags
tag_to_idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
tag_to_string = {'PER': 'person', 'ORG': 'organisation', 'LOC': 'location', 'MISC': 'miscellaneous'}
string_to_tag = {string: tag for tag, string in tag_to_string.items()}

def build_ent_sequences(tokens, tags, sep_token):
    '''Combines BIO tags used in the CoNLL2003 dataset into combined PER, ORG, LOC, MISC labels.
    Also splits sequence into expected entity sequence format.

    Args:
        tokens : Array of tokens
        tags : Array of tags in BIO format
        sep_token : Token used by tokeniser to separate entities and tags
    Returns:
        Array of entity sequences
    '''
    compound_tokens = []

    for token, tag in zip(tokens, tags):
        if tag in [1, 3, 5, 7]:
            compound_tokens.append([token, sep_token, f"{tag_to_string[idx_to_tag[tag].split('-')[-1]]}"])
        elif tag in [2, 4, 6, 8]:
            compound_tokens[-1][-2:-2] = [token]
    
    return [' '.join(token_tags) for token_tags in compound_tokens]


def prepare_dataset():
    dataset = load_dataset("conll2003", revision='master')
    
    original_columns = dataset['train'].column_names
    dataset = dataset.map(lambda batch: {
            'sents': TreebankWordDetokenizer().detokenize(batch['tokens']),
            'ent_seqs': f' | '.join(build_ent_sequences(batch['tokens'], batch['ner_tags'], '|')),
        }).remove_columns(original_columns)
    
    # Handles cases where text contains no entities
    dataset = dataset.map(lambda example: {'ent_seqs': example['ent_seqs'] if example['ent_seqs'] else ' '})

    a_train_dataset = dataset['train'].remove_columns(['ent_seqs']).rename_column('sents', 'text')
    b_train_dataset = dataset['train'].remove_columns(['sents']).rename_column('ent_seqs', 'text')

    a_val_dataset = dataset['validation'].rename_column('sents', 'text').rename_column('ent_seqs', 'label')
    b_val_dataset = dataset['validation'].rename_column('ent_seqs', 'text').rename_column('sents', 'label')

    a_dataset = DatasetDict({
        'train': a_train_dataset,
        'validation': a_val_dataset,
    })

    b_dataset = DatasetDict({
        'train': b_train_dataset,
        'validation': b_val_dataset,
    })
    
    return a_dataset, b_dataset


def main():
    model_a_config = ModelConfig(
        pretrained_model_name_or_path='google/flan-t5-small',
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        generation_config=GenerationConfig(
            max_new_tokens=50
        )
    )

    model_b_config = ModelConfig(
        pretrained_model_name_or_path='gpt2',
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        padding_side='left',
        generation_config=GenerationConfig(
            max_new_tokens=50
        )
    )

    model = CycleModel(model_a_config, model_b_config)

    model.tokenizer_B.pad_token_id = model.tokenizer_B.eos_token_id
    
    a_dataset, b_dataset = prepare_dataset()
    
    a_dataset = a_dataset.map(lambda example: {
        **model.tokenizer_A(example['text'], padding=True),
    })
    a_dataset['validation'] = a_dataset['validation'].map(lambda example: {
        'labels': model.tokenizer_A(example['label'], padding=True)['input_ids'],
    }, remove_columns=['label'])
    a_dataset = a_dataset.remove_columns('text')


    b_dataset = b_dataset.map(lambda example: {
        **model.tokenizer_B(example['text'] + '\n' + example['label']),
    })
    b_dataset['validation']['labels'] = b_dataset['validation']['input_ids']
    b_dataset = b_dataset.remove_columns('text')

    trainer_config = TrainerConfig(
        logger=WandbLogger(project='CycleNER', name='CycleNER Example'),
        log_every_n_steps=10,
    )

    trainer = CycleTrainer(
        model,
        args=trainer_config,
        train_dataset_A=a_dataset['train'],
        train_dataset_B=b_dataset['train'],
        eval_dataset_A=a_dataset['validation'],
        eval_dataset_B=b_dataset['validation'],
    )

    trainer.train()


if __name__ == "__main__":
    main()