import torch


class Cycle:
    def __init__(self, model, tokenizer, generation_config):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    def generate(self, batch):
        raise NotImplementedError
    
    def decode(self, batch):
        raise NotImplementedError
    
    def encode(self, batch):
        raise NotImplementedError
    
    def format(self, batch):
        raise NotImplementedError
    
    def train(self, batch, labels):
        raise NotImplementedError
    
    def encode_and_format(self, batch):
        return self.format(self.encode(batch))
    

class Seq2SeqCycle(Cycle):
    def __init__(self, model, tokenizer, generation_config):
        super().__init__(model, tokenizer, generation_config)

    def generate(self, batch):
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        output_ids = self.model.generate(
            input_ids         = input_ids.to(self.model.device),
            attention_mask    = attention_mask.to(self.model.device),
            pad_token_id      = self.tokenizer.pad_token_id,
            generation_config = self.generation_config,
            )

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': output_ids}
    
    def decode(self, batch):
        return {'input': self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True),
                'labels': self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)}
    
    def encode(self, batch):
        input_ids, attention_mask = self.tokenizer.batch_encode_plus(batch['labels'], return_tensors='pt', padding=True).values()
        labels = self.tokenizer.batch_encode_plus(batch['input'], return_tensors='pt', padding=True)['input_ids']

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    
    def format(self, batch):
        labels = batch['labels'][batch['labels']==self.tokenizer.pad_token_id] = -100
        return {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': labels}
    
    def train(self, batch):
        outputs = self.model(
            input_ids         = batch['input_ids'].to(self.model.device),
            attention_mask    = batch['attention_mask'].to(self.model.device),
            labels            = batch['labels'].to(self.model.device),
            )
        
        return outputs


class CausalCycle(Cycle):
    def __init__(self, model, tokenizer, generation_config):
        super().__init__(model, tokenizer, generation_config)

    def generate(self, batch):
        ''' Generate a response to the input_ids using the model

        real_input --model.generate->> real_input + synthetic_response
        '''
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        output_ids = self.model.generate(
            input_ids         = input_ids.to(self.model.device),
            attention_mask    = attention_mask.to(self.model.device),
            pad_token_id      = self.tokenizer.pad_token_id,
            generation_config = self.generation_config,
            )
        
        response_ids = output_ids[:, input_ids.shape[-1]:]

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': response_ids}

    def decode(self, batch):
        return {'input': self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True),
                'labels': self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)}
    
    def encode(self, batch):
        input_ids, attention_mask = self.tokenizer.batch_encode_plus(batch['labels'], return_tensors='pt', padding=True)
        labels = self.tokenizer.batch_encode_plus(batch['input'], return_tensors='pt', padding=True)['input_ids']

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'response': labels}
    
    def format(self, batch):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        labels = torch.cat((input_ids, labels), dim=-1)
        labels[labels==self.tokenizer.pad_token_id] = -100

        pad_size = labels.shape[-1] - input_ids.shape[-1]
        pad_tensor = torch.full((labels.shape[0], pad_size), self.tokenizer.pad_token_id).to(self.model.device)
        attention_mask = torch.cat((torch.zeros_like(pad_tensor), torch.ones_like(input_ids)), dim=-1)
        input_ids = torch.cat((pad_tensor, input_ids), dim=-1)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def train(self, batch):
        ''' Train the model on the generated response and the original input

        synthetic_response --model.__call__->> Reconstruction loss(synthetic_response + real_input)
        '''
        outputs = self.model(
            input_ids         = batch['input_ids'].to(self.model.device),
            attention_mask    = batch['attention_mask'].to(self.model.device),
            labels            = batch['labels'].to(self.model.device),
            )
        
        return outputs
        
def initialise_cycle(model, tokenizer, generation_config, task):
    if task == 'SEQ2SEQ_LM':
        return Seq2SeqCycle(model, tokenizer, generation_config)
    elif task == 'CAUSAL_LM':
        return CausalCycle(model, tokenizer, generation_config)
    else:
        raise NotImplementedError('There is no cycle class implemented for this task.')