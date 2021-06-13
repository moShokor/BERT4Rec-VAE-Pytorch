import ast

import pandas as pd

from .base import AbstractDataset
from .steam_meta import SteamMetaDataset
from .utils import date2timestamp
from .downloadable import Extension


class SteamV2Dataset(AbstractDataset):

    def __init__(self, args):
        super().__init__(args)
        self.meta_data = SteamMetaDataset(args)

    @classmethod
    def code(cls):
        return 'steamV2'

    def url(self):
        return 'http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz'

    @classmethod
    def compressed_file_content_is_folder(cls):
        return False

    @classmethod
    def extension(cls):
        return Extension.GZIP

    @classmethod
    def all_raw_file_names(cls):
        return ['steam_new.json']

    def load_ratings_df(self):
        file_path = self._get_rawdata_folder_path()
        with open(file_path) as f:
            lines = f.readlines()
        results = []
        skipped_reviews = 0
        for line in lines:
            if not line:
                continue
            user = ast.literal_eval(line)
            try:
                # TODO maybe add the reviews later if wanted
                results.append({'uid': user['username'], 'sid': user['product_id'], 'rating': 1,
                                'timestamp': date2timestamp(user['date'], '%Y-%m-%d')})
            except:
                skipped_reviews += 1
                continue

        print(
            f'{skipped_reviews} reviews skipped out of {skipped_reviews + len(results)}'
            f' ({skipped_reviews / (skipped_reviews + len(results))})')
        df = pd.DataFrame(results)
        return df

    def get_sid2name(self):
        self.meta_data.get_sid2name()
