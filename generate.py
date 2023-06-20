import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from data_loader import TestDataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.task.upper() == 'CAUSAL_LM':
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    elif args.task.upper() == 'SEQ2SEQ_LM':
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f'Unknown task {args.task}')
    
    test_data = TestDataset(tokenizer).load_data(args.data_path)
    collator = DataCollatorForSeq2Seq(tokenizer, return_tensors='pt', padding=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collator)

    for batch in test_loader:
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']

        # TODO: Add generate params
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)

        with open(args.output_dir + 'output.txt', 'a+') as f:
            for o in decoded_output:
                f.write(o + '\n')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", help="Model name or path to model for model A (default: gpt2)")
    parser.add_argument("--data_path", type=str, default="data/example/samples.jsonl", help="Path to data (default: data/example/samples.json)")
    parser.add_argument("--output_dir", type=str, default="models/example/", help="Path to save model checkpoints (default: models/example)")
    parser.add_argument("--task", type=str, default="causal_lm", help="Task for the model (default: causal_lm))")

    args = parser.parse_args()
    main(args)