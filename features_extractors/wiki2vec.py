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
        model = Wikipedia2Vec.load(file_path)
        zeroth_index = 0
        return model, zeroth_index

    @classmethod
    def code(cls):
        return 'wiki2vec'

    @classmethod
    def extension(cls):
        return Extension.BZ

    def url(self):
        return f'http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/' \
               f'{self.model_name()}.bz2'

    # TODO maybe merge the methods build_correspondence and embed
    def build_correspondence(self, sid2name):
        sid2wiki_id = {}
        lost = 0
        print(f'parsing {self.code()} features')
        for sid, name in tqdm(sid2name.items()):
            for new_name in self.pre_process_name(name):
                el_id = self.model.dictionary.get_entity(new_name)
                el_id = el_id if el_id else self.model.dictionary.get_word(new_name)
                el_id = el_id.index if el_id else self.zeroth_index
                assert el_id < len(self.model.syn0)
                sid2wiki_id[sid] = el_id
                if not el_id:
                    lost += 1

        total = len(sid2name)
        print(f'when using the {self.code()} feature we could not match {lost} out of {total}'
              f' items ratio: {(lost / total):2f}%')
        return sid2wiki_id

    def embed(self, idx_tensor):
        return torch.tensor(np.array([list(self.model.syn0[el])
                                      if el else list(torch.zeros(self.model.syn0.shape[1]))
                                      for el in idx_tensor]))

    def model_name(self):
        return f'enwiki_20180420_{self.type}{self.dimension}d.pkl'

    def get_raw_model_path(self):
        return self.get_raw_asset_folder_path().joinpath(self.model_name())
