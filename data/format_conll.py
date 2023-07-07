from datasets import load_dataset, DatasetDict
import argparse
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os


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

def main(args):
    dataset = load_dataset("conll2003")
    
    original_columns = dataset['train'].column_names
    dataset = dataset.map(lambda batch: {
            'sents': TreebankWordDetokenizer().detokenize(batch['tokens']),
            'ent_seqs': f' {args.sep_token} '.join(build_ent_sequences(batch['tokens'], batch['ner_tags'], args.sep_token)),
        }).remove_columns(original_columns)
    
    a_train_dataset = dataset['train'].remove_columns(['ent_seqs']).rename_column('sents', 'text')
    b_train_dataset = dataset['train'].remove_columns(['sents']).rename_column('ent_seqs', 'text')

    a_val_dataset = dataset['validation'].rename_column('sents', 'text').rename_column('ent_seqs', 'label')
    b_val_dataset = dataset['validation'].rename_column('ent_seqs', 'text').rename_column('sents', 'label')

    a_test_dataset = dataset['test'].rename_column('sents', 'text').rename_column('ent_seqs', 'label')
    b_test_dataset = dataset['test'].rename_column('ent_seqs', 'text').rename_column('sents', 'label')

    if args.s_examples:
        a_train_dataset = a_train_dataset.shuffle(seed=args.seed).select(range(args.s_examples))
        
    if args.e_examples:
        b_train_dataset = b_train_dataset.shuffle(seed=args.seed).select(range(args.e_examples))

    a_dataset = DatasetDict({
        'train': a_train_dataset,
        'validation': a_val_dataset,
        'test': a_test_dataset
    })

    b_dataset = DatasetDict({
        'train': b_train_dataset,
        'validation': b_val_dataset,
        'test': b_test_dataset
    })

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    a_dataset.save_to_disk(os.path.join(args.data_dir, 'A'))
    b_dataset.save_to_disk(os.path.join(args.data_dir, 'B'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CycleNER training script.')
    parser.add_argument('--s_examples', default=None, help='Number of training sentences to use. None is all sentences (default: None).', type=int)
    parser.add_argument('--e_examples', default=None, help='Number of training entity sequences to use. None is all entity sequences (default: None).', type=int)
    parser.add_argument('--sep_token', default='|', help='Separator token used in model (default: |).')
    parser.add_argument('--data_dir', default='conll2003', help='Directory to save data (default: conll2003).')
    parser.add_argument('--seed', default=42, help='Random seed (default: 42).', type=int)
    args = parser.parse_args()
    main(args)