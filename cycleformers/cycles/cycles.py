import torch
from transformers import GenerationConfig

class Cycle:
    @staticmethod
    def generate(batch):
        raise NotImplementedError
    
    @staticmethod
    def decode(tokenizer):
        def cycle_decode(**inputs):
            return {'text': tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True),
                    'labels': tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True)}
        
        return cycle_decode
    
    @staticmethod
    def encode(tokenizer):
        def cycle_encode(**inputs):
            new_inputs = tokenizer(inputs['text'], return_tensors='pt', padding=True)
            labels = tokenizer(inputs['labels'], return_tensors='pt', padding=True)['input_ids']

            return new_inputs | {'labels': labels}
        
        return cycle_encode
    
    @staticmethod
    def format(batch):
        raise NotImplementedError
    
    @staticmethod
    def train(batch, labels):
        raise NotImplementedError
    

class CausalCycle(Cycle):
    @staticmethod
    def generate(model, tokenizer, generation_config):
        if isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config, max_new_tokens=100)
        elif not isinstance(generation_config, GenerationConfig):
            raise ValueError(f"generation_config must be a dict or GenerationConfig, got {type(generation_config)}")
        
        def causal_generate(**inputs):
            model.eval()

            # Force left padding while generating
            default_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    pad_token_id=tokenizer.pad_token_id,
                    generation_config=generation_config,
                )

            tokenizer.padding_side = default_padding_side
            response_ids = output_ids[:, inputs['input_ids'].shape[-1]:]

            print(tokenizer.batch_decode(response_ids, skip_special_tokens=True))
            
            # Original inputs become new labels
            return {'input_ids': response_ids, 'labels': inputs['input_ids']}
        
        return causal_generate
    
    @staticmethod
    def format(model, tokenizer, data_collator):
        def causal_format(**inputs):
            inputs = {k: v.cpu() for k, v in inputs.items()}

            # Convert from dict of lists to list of dicts to use collator
            inputs = [dict(zip(inputs.keys(), [v[v != tokenizer.pad_token_id] for v in values])) for values in zip(*inputs.values())]
            inputs = data_collator(inputs)

            return inputs
        
        return causal_format
    
    @staticmethod
    def train(model):
        def causal_train(**inputs):
            model.train()

            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            return outputs
        
        return causal_train
    


class Seq2SeqCycle(Cycle):
    @staticmethod
    def generate(model, tokenizer, generation_config):
        if isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config, max_new_tokens=100)
        elif not isinstance(generation_config, GenerationConfig):
            raise ValueError(f"generation_config must be a dict or GenerationConfig, got {type(generation_config)}")
        
        def seq2seq_generate(**inputs):
            model.eval()

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    pad_token_id=tokenizer.pad_token_id,
                    generation_config=generation_config,
                )

            # Strip leading pad token if generated
            if output_ids[0, 0] == tokenizer.pad_token_id:
                output_ids = output_ids[:, 1:]
            
            print(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

            # Original inputs become new labels
            return {'input_ids': output_ids, 'labels': inputs['input_ids']}
        
        return seq2seq_generate
    
    @staticmethod
    def format(model, tokenizer, data_collator):
        def seq2seq_format(**inputs):
            # Set attention mask if skipping re-encoding because of same tokeniser for both models
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                inputs['attention_mask'][inputs['input_ids']==tokenizer.pad_token_id] = 0

            inputs = {k: v.to('cpu') for k, v in inputs.items()}

            # Convert from dict of lists to list of dicts to use collator
            inputs = [dict(zip(inputs.keys(), [v[v != tokenizer.pad_token_id] for v in values])) for values in zip(*inputs.values())]
            inputs = data_collator(inputs)

            return inputs
        
        return seq2seq_format
    
    @staticmethod
    def train(model):
        def seq2seq_train(**inputs):
            model.train()
            
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            return outputs
        
        return seq2seq_train