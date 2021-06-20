from abc import ABCMeta, abstractmethod
from pathlib import Path

from config import RAW_EXTRACTOR_ROOT_FOLDER
from datasets.downloadable import Downloadable


class AbstractExtractor(Downloadable, metaclass=ABCMeta):
    def __init__(self, args):
        self.maybe_download_raw_asset()
        self.load_model(args)

    @abstractmethod
    def embedding_size(self):
        pass

    @abstractmethod
    def load_model(self, args):
        pass

    @classmethod
    def pre_process_name(cls, name):
        n1 = name.split(' (')[0]
        n2 = ' '.join(reversed(n1.split(','))).strip()
        return n1, n2

    @abstractmethod
    def build_correspondence(self, sid2name):
        """
        builds a correspondence between the item name and it's ids in the dataset
        :param sid2name: a dictionary whose keys are the sids and values are the unprocessed names
        :return: an sid2id dictionary which keys are the sid and values are the ids in the
        model matrix
        """
        pass

    def get_raw_asset_root_path(self):
        return Path(RAW_EXTRACTOR_ROOT_FOLDER)

    def maybe_download_raw_asset(self):
        print(f"fetching the {self.code()} extractor")
        super(AbstractExtractor, self).maybe_download_raw_asset()
