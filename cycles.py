

class Seq2SeqCycle:
    def __init__(self, model, tokenizer, generation_config):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    def generate(self, batch):
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        output_ids = self.model.generate(
            input_ids         = input_ids.to(self.device),
            attention_mask    = attention_mask.to(self.device),
            pad_token_id      = self.tokenizer.pad_token_id,
            generation_config = self.generation_config,
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def train(self, batch, labels):
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']