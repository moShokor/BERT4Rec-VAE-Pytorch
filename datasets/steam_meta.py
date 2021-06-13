import ast

from tqdm import tqdm

from .base import AbstractDataset
from .downloadable import Extension


class SteamMetaDataset(AbstractDataset):

    @classmethod
    def code(cls):
        return 'steamMeta'

    def url(self):
        return 'http://cseweb.ucsd.edu/~wckang/steam_games.json.gz'

    @classmethod
    def compressed_file_content_is_folder(cls):
        return False

    @classmethod
    def extension(cls):
        return Extension.GZIP

    @classmethod
    def all_raw_file_names(cls):
        return ['steam_games.json']

    def load_ratings_df(self):
        pass

    def get_sid2name(self):
        # TODO we might also want to include further information about the meta data like
        # the puplisher or the tags
        file_path = self._get_rawdata_folder_path()
        with open(file_path) as f:
            lines = f.readlines()
        sid2name = {}
        for line in tqdm(lines):
            try:
                # parses the line into a user dict if parsable
                item = ast.literal_eval(line)
                sid = item['id']
                name = item['title']
                sid2name[sid] = name
            except:
                continue
        return sid2name
