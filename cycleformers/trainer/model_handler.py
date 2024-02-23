


class ModelHandler:
    def __init__(self, model, tokenizer, args, data_collator, optimizer, lr_scheduler):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.data_collator = data_collator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler