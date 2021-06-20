import torch
import numpy as np
from wikipedia2vec import Wikipedia2Vec

from datasets.downloadable import Extension
from features_extractors.base import AbstractExtractor
from tqdm import tqdm


class Wiki2VecExtractor(AbstractExtractor):

    def __init__(self, args):
        self.dimension = args.wiki2vec_dimension
        self.type = args.wiki2vec_model_type
        self.type = '' if self.type == 'NA' else self.type
        super().__init__(args)

    def embedding_size(self):
        return int(self.dimension)

    def load_model(self, args):
        file_path = self.get_raw_model_path()
        self.model = Wikipedia2Vec.load(file_path)
        self.s = self.model.syn0
        self.s = np.concatenate([self.s, np.zeros((1, self.s.shape[-1]))])
        self.s = torch.from_numpy(self.s).float()
        self.zeroth_index = len(self.s) - 1

    @classmethod
    def code(cls):
        return 'wiki2vec'

    @classmethod
    def extension(cls):
        return Extension.BZ

    def url(self):
        return f'http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/' \
               f'{self.asset_name()}.bz2'

    # TODO maybe merge the methods build_correspondence and embed
    def build_correspondence(self, sid2name):
        sid2wiki_id = {}
        lost = 0
        print(f'parsing {self.code()} features')
        for sid, name in tqdm(sid2name.items()):
            for new_name in self.pre_process_name(name):
                el_id = self.model.dictionary.get_entity(new_name)
                el_id = el_id if el_id else self.model.dictionary.get_word(new_name)
                if el_id:
                    sid2wiki_id[sid] = el_id.index
                    break
            if not sid2wiki_id.get(sid, None):
                sid2wiki_id[sid] = self.zeroth_index
                lost += 1
        # adding the zeroth index for the mask token
        sid2wiki_id[len(sid2name)]= self.zeroth_index
        total = len(sid2name)
        print(f'when using the {self.code()} feature we could not match {lost} out of {total}'
              f' items ratio: {(lost / total):2f}%')
        return sid2wiki_id

    def embed(self, idx_tensor):
        return torch.index_select(self.s, 0, idx_tensor)

    def asset_name(self):
        return f'enwiki_20180420_{self.type}{self.dimension}d.pkl'

    def get_raw_model_path(self):
        return self.get_raw_asset_folder_path().joinpath(self.asset_name())

    @classmethod
    def compressed_file_content_is_folder(cls):
        return False
