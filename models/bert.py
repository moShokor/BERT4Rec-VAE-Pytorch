from .base import BaseModel
from .bert_modules.bert import BERT

import torch.nn as nn


class BERTModel(BaseModel):
    def __init__(self, args, extractors):
        super().__init__(args, extractors)
        self.bert = BERT(args, extractors)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 1)

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x, additional):
        x = self.bert(x, additional)
        return self.out(x)
