import pandas as pd
from .base import AbstractDataset
import pickle5 as pk

class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def compressed_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'movies.dat',
                'ratings.dat',
                'users.dat']

    def load_ratings_df(self):
        folder_path = self.get_raw_asset_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df

    def get_sid2name(self):
        folder_path = self.get_raw_asset_folder_path()
        file_path = folder_path.joinpath('movies.dat')
        df = pd.read_csv(file_path, sep='::', header=None)
        df.columns = ['sid', 'name', 'tags']
        return dict(zip(df.sid, df.name))

    def get_sid2id(self):
        with open('./ml-1m-Links.pk', 'rb') as f:
            id2imdbId = pk.load(f)
        return id2imdbId
