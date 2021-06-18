import torch.nn as nn

from abc import *


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, args, extractors):
        super().__init__()
        self.extractors = extractors
        self.args = args

    @classmethod
    @abstractmethod
    def code(cls):
        pass

