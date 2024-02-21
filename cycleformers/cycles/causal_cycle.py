import torch
from transformers import GenerationConfig


class CausalCycle:
    @staticmethod
    def generate(model, tokenizer, generation_config):
        if isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)
        elif not isinstance(generation_config, GenerationConfig):
            raise ValueError(f"generation_config must be a dict or GenerationConfig, got {type(generation_config)}")
        
        def causal_generate(batch):
            default_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'

            input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
            output_ids = model.generate(
                input_ids         = input_ids.to(model.device),
                attention_mask    = attention_mask.to(model.device),
                pad_token_id      = tokenizer.pad_token_id,
                generation_config = generation_config,
            )

            tokenizer.padding_side = default_padding_side
            response_ids = output_ids[:, input_ids.shape[-1]:]
            
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': response_ids}
        
        return causal_generate
    
    @staticmethod
    def decode(tokenizer):
        def causal_decode(batch):
            return {'input': tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True),
                    'labels': tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)}
        
        return causal_decode
    
    @staticmethod
    def encode( tokenizer):
        def causal_encode(batch):
            input_ids, attention_mask = tokenizer(batch['labels'], return_tensors='pt', padding=True).values()
            labels = tokenizer(batch['input'], return_tensors='pt', padding=True)['input_ids']

            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        
        return causal_encode
    
    @staticmethod
    def format(model, tokenizer):
        def causal_format(batch):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

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
        def causal_train(batch):
            outputs = model(
                input_ids      = batch['input_ids'].to(model.device),
                attention_mask = batch['attention_mask'].to(model.device),
                labels         = batch['labels'].to(model.device),
            )
            
            return outputs
        
        return causal_train