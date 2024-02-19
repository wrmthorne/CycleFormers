from transformers import GenerationConfig


class Seq2SeqCycle:
    @staticmethod
    def generate(model, tokenizer, generation_config):
        if isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)
        elif not isinstance(generation_config, GenerationConfig):
            raise ValueError(f"generation_config must be a dict or GenerationConfig, got {type(generation_config)}")
        
        def seq2seq_generate(batch):
            input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
            output_ids = model.generate(
                input_ids         = input_ids.to(model.device),
                attention_mask    = attention_mask.to(model.device),
                pad_token_id      = tokenizer.pad_token_id,
                generation_config = generation_config,
            )

            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': output_ids}
        
        return seq2seq_generate
    
    @staticmethod
    def decode(tokenizer):
        def seq2seq_decode(batch):
            return {'input': tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True),
                    'labels': tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)}
        
        return seq2seq_decode
    
    @staticmethod
    def encode(tokenizer):
        def seq2seq_encode(batch):
            input_ids, attention_mask = tokenizer(batch['labels'], return_tensors='pt', padding=True).values()
            labels = tokenizer(batch['input'], return_tensors='pt', padding=True)['input_ids']

            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        
        return seq2seq_encode
    
    @staticmethod
    def format(model, tokenizer):
        def seq2seq_format(batch):
            batch['labels'][batch['labels']==tokenizer.pad_token_id] = -100
            return {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['labels']}
        
        return seq2seq_format
    
    @staticmethod
    def train(model):
        def seq2seq_train(batch):
            outputs = model(
                input_ids      = batch['input_ids'].to(model.device),
                attention_mask = batch['attention_mask'].to(model.device),
                labels         = batch['labels'].to(model.device),
            )
            
            return outputs
        
        return seq2seq_train