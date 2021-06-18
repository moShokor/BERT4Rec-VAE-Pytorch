import torch
import torch.nn as nn
from .base import CombinationLayer


class FactoredCombination(CombinationLayer):
    def __init__(self, extractors, embedding_size):
        super().__init__(extractors, embedding_size)

    def forward(self, x, additional):
        raise NotImplemented

    @classmethod
    def code(cls):
        return 'factored'
