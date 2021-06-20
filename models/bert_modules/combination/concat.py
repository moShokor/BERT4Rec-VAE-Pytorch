import torch
import torch.nn as nn
from .base import CombinationLayer


class ConcatCombination(CombinationLayer):
    def __init__(self, extractors, embedding_size):
        super().__init__(extractors, embedding_size)
        input_size = embedding_size + sum([extractor.embedding_size()
                                           for extractor in extractors.values()])
        self.reduction = nn.Linear(input_size, embedding_size)

    def forward(self, x, additional):
        # simply stack the elements first
        print(f'x: {x.shape}')
        print(f'a: {additional[0].shape}')
        res = torch.cat([x] + additional, dim=-1)
        return self.reduction(res)

    @classmethod
    def code(cls):
        return 'concat'
