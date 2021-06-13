from wikipedia2vec import Wikipedia2Vec

from datasets.downloadable import Extension
from features_extractors.base import AbstractExtractor
from tqdm import tqdm


class Wiki2VecExtractor(AbstractExtractor):

    def __init__(self, args):
        self.dimension = args.wiki2vec_dimension
        self.type = args.wiki2vec_model_type
        super().__init__(args)

    def load_model(self, args):
        file_path = self.get_raw_asset_root_path()
        model = Wikipedia2Vec.load(file_path)
        zeroth_index = model.syn0.shape[0]
        return model, zeroth_index

    @classmethod
    def code(cls):
        return 'wiki2vec'

    @classmethod
    def extension(cls):
        return Extension.BZ

    def url(self):
        return f'http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/' \
               f'enwiki_20180420_{self.type}{self.dimension}d.pkl.bz2'

    def build_correspondence(self, sid2name):
        sid2wiki_id = {}
        print(f'parsing {self.code()} features')
        for sid, name in tqdm(sid2name.items()):
            for new_name in self.pre_process_name(name):
                el_id = self.model.dictionary.get_entity(new_name)
                el_id = el_id if el_id else self.model.dictionary.get_word(new_name)
                el_id = el_id if el_id else self.zeroth_index
                if el_id != self.zeroth_index:
                    sid2wiki_id[sid] = el_id

        total = len(sid2name)
        lost = total - len(sid2wiki_id)
        print(f'when using the {self.code()} feature we could not match {lost} out of {total}'
              f' items ratio: {(lost / total):2f}%')
        return sid2wiki_id
