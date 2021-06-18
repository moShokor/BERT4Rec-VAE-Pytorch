from abc import abstractmethod

import torch.nn as nn


class CombinationLayer(nn.Module):
    def __init__(self, extractors, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.extractors = extractors

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def forward(self, x, additional):
        pass
