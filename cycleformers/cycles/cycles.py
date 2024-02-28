import torch
from transformers import GenerationConfig

class Cycle:
    @staticmethod
    def generate(batch):
        raise NotImplementedError
    
    @staticmethod
    def decode(batch):
        raise NotImplementedError
    
    @staticmethod
    def encode(batch):
        raise NotImplementedError
    
    @staticmethod
    def format(batch):
        raise NotImplementedError
    
    @staticmethod
    def train(batch, labels):
        raise NotImplementedError
    

class CausalCycle:
    @staticmethod
    def generate(model, tokenizer, generation_config):
        if isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)
        elif not isinstance(generation_config, GenerationConfig):
            raise ValueError(f"generation_config must be a dict or GenerationConfig, got {type(generation_config)}")
        
        def causal_generate(**inputs):
            model.eval()

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
            
            return inputs | {'labels': response_ids}
        
        return causal_generate
    
    @staticmethod
    def decode(tokenizer):
        def causal_decode(**inputs):
            return {'text': tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True),
                    'labels': tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True)}
        
        return causal_decode
    
    @staticmethod
    def encode(tokenizer):
        def causal_encode(**inputs):
            new_inputs = tokenizer(inputs['labels'], return_tensors='pt', padding=True)
            labels = tokenizer(inputs['text'], return_tensors='pt', padding=True)['input_ids']

            return new_inputs | {'labels': labels}
        
        return causal_encode
    
    @staticmethod
    def format(model, tokenizer):
        def causal_format(**inputs):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_ids, attention_mask, labels = inputs['input_ids'], inputs['attention_mask'], inputs['labels']

            labels = torch.cat((input_ids, labels), dim=-1)
            labels[labels==tokenizer.pad_token_id] = -100

            pad_size = labels.shape[-1] - input_ids.shape[-1]
            pad_tensor = torch.full((labels.shape[0], pad_size), tokenizer.pad_token_id).to(model.device)
            attention_mask = torch.cat((torch.zeros_like(pad_tensor), torch.ones_like(input_ids)), dim=-1)
            input_ids = torch.cat((pad_tensor, input_ids), dim=-1)

            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        
        return causal_format
    
    @staticmethod
    def train(model):
        def causal_train(**inputs):
            model.train()

            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            return outputs
        
        return causal_train
    


class Seq2SeqCycle:
    @staticmethod
    def generate(model, tokenizer, generation_config):
        if isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)
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

                print(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

            return inputs | {'labels': output_ids}
        
        return seq2seq_generate
    
    @staticmethod
    def decode(tokenizer):
        def seq2seq_decode(**inputs):
            return {'text': tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True),
                    'labels': tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True)}
        
        return seq2seq_decode
    
    @staticmethod
    def encode(tokenizer):
        def seq2seq_encode(**inputs):
            input_ids, attention_mask = tokenizer(inputs['labels'], return_tensors='pt', padding=True).values()
            labels = tokenizer(inputs['text'], return_tensors='pt', padding=True)['input_ids']

            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        
        return seq2seq_encode
    
    @staticmethod
    def format(model, tokenizer):
        def seq2seq_format(**inputs):
            inputs['labels'][inputs['labels']==tokenizer.pad_token_id] = -100
            return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': inputs['labels']}
        
        return seq2seq_format
    
    @staticmethod
    def train(model):
        def seq2seq_train(**inputs):
            model.train()
            
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            return outputs
        
        return seq2seq_train