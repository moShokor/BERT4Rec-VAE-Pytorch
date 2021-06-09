import ast

import pandas as pd
from tqdm import tqdm

from .base import AbstractDataset
from .utils import date2timestamp


class SteamV1Dataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'steamV1'

    @classmethod
    def url(cls):
        # return 'http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz'
        return 'http://deepx.ucsd.edu/public/jmcauley/steam/australian_user_reviews.json.gz'

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

        for line in tqdm(lines):
            try:
                # parses the line into a user dict if parsable
                user = ast.literal_eval(line)
            except:
                continue
            uid = user['user_id']
            user_ratings = []
            for el in user['reviews']:
                try:
                    # TODO maybe add the reviews text later if wanted
                    user_ratings.append({'uid': uid, 'sid': el['item_id'], 'rating': 1,
                                         'timestamp': date2timestamp(
                                             ' '.join(el['posted'].split()[1:]))})
                except:
                    skipped_reviews += 1
                    continue
            results += user_ratings
        print(
            f'{skipped_reviews} reviews skipped out of {skipped_reviews + len(results)}'
            f' ({skipped_reviews / (skipped_reviews + len(results))})')
        df = pd.DataFrame(results)
        return df
