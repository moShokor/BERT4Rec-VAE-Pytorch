import ast

import pandas as pd
from tqdm import tqdm

from .base import AbstractDataset
from .utils import date2timestamp


class SteamV2Dataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'steamV2'

    @classmethod
    def url(cls):
        return 'http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz'

    @classmethod
    def zip_file_content_is_folder(cls):
        return False

    @classmethod
    def is_gzipfile(cls):
        return True

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
