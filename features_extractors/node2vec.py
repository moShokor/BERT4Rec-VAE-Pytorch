import pickle as pk

import numpy as np
import torch
from tqdm import tqdm

from features_extractors.base import AbstractExtractor


class Node2VecExtractor(AbstractExtractor):

    def load_model(self, args):
        file_path = self.get_raw_model_path()
        with open(file_path, 'rb') as f:
            self.model = pk.load(f)
        try:
            self.s = self.model.syn1
        except:
            self.s = self.model.syn1neg
        self.s = np.concatenate([self.s, np.zeros((1, self.s.shape[-1]))])
        self.s = torch.from_numpy(self.s).float()
        self.zeroth_index = len(self.s) - 1

    def embedding_size(self):
        return self.s.shape[1]

    @classmethod
    def code(cls):
        return 'node2vec'

    @classmethod
    def extension(cls):
        raise NotImplemented

    def url(self):
        raise NotImplemented

    def build_correspondence(self, sid2imdbId):
        sid2model_id = {}
        print(f'parsing {self.code()} features')
        for sid, imdbId in tqdm(sid2imdbId.items()):
            el_id = self.model.wv.key_to_index.get(str(imdbId), None)
            el_id = el_id if el_id else self.zeroth_index
            sid2model_id[sid] = el_id
        # adding the zeroth index for the mask token
        sid2model_id[len(sid2imdbId)] = self.zeroth_index
        return sid2model_id

    def embed(self, idx_tensor):
        return torch.index_select(self.s, 0, idx_tensor)

    def asset_name(self):
        return 'graph.pk'

    def get_raw_model_path(self):
        return self.get_raw_asset_folder_path().joinpath(self.asset_name())

    @classmethod
    def compressed_file_content_is_folder(cls):
        return False
