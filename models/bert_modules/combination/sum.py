import torch

from .base import CombinationLayer


class SumCombination(CombinationLayer):
    def __init__(self, extractors, embedding_size):
        super().__init__(extractors, embedding_size)

    def forward(self, x, additional):
        # simply sums all the elements
        # TODO here we must make sure that all the inputs have the same dimension for the
        # sum to work
        return torch.stack([x] + additional, dim=0).sum(dim=0)

    @classmethod
    def code(cls):
        return 'sum'
