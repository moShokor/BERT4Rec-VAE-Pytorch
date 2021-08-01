import pickle as pk

import numpy as np
import torch

from features_extractors.base import AbstractExtractor


class FastTextExtractor(AbstractExtractor):

    def load_model(self, args):
        file_path = self.get_raw_model_path()
        with open(file_path, 'rb') as f:
            self.s, self.sid2model_id = pk.load(f)
        self.s = np.concatenate([self.s, np.zeros((1, self.s.shape[-1]))])
        self.s = torch.from_numpy(self.s).float()
        self.zeroth_index = len(self.s) - 1

    def embedding_size(self):
        return self.s.shape[1]

    @classmethod
    def code(cls):
        return 'fasttext'

    @classmethod
    def extension(cls):
        raise NotImplemented

    def url(self):
        raise NotImplemented

    def build_correspondence(self, sid2imdbId=None):
        return self.sid2model_id

    def embed(self, idx_tensor):
        return torch.index_select(self.s, 0, idx_tensor)

    def asset_name(self):
        return 'Fasttext_extractor.pk'

    def get_raw_model_path(self):
        return self.get_raw_asset_folder_path().joinpath(self.asset_name())

    @classmethod
    def compressed_file_content_is_folder(cls):
        return False
