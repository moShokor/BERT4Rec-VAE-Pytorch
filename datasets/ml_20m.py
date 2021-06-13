import pandas as pd

from .base import AbstractDataset


class ML20MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-20m'

    def url(self):
        return 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'

    @classmethod
    def compressed_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['genome-scores.csv',
                'genome-tags.csv',
                'links.csv',
                'movies.csv',
                'ratings.csv',
                'README.txt',
                'tags.csv']

    def load_ratings_df(self):
        folder_path = self.get_raw_asset_folder_path()
        file_path = folder_path.joinpath('ratings.csv')
        df = pd.read_csv(file_path)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df

    def get_sid2name(self):
        folder_path = self.get_raw_asset_folder_path()
        file_path = folder_path.joinpath('movies.csv')
        df = pd.read_csv(file_path)
        df.columns = ['sid', 'name', 'tags']
        return dict(zip(df.sid, df.name))
