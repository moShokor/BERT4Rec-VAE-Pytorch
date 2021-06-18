import os
import shutil
import tempfile
from abc import ABCMeta, abstractmethod
from enum import Enum
from pathlib import Path

from tqdm import tqdm

from config import RAW_DATASET_ROOT_FOLDER, TMP_FOLDER

tqdm.pandas()
from datasets.utils import ungzip, download, unzip, unbz


class Extension(Enum):
    ZIP = 'zip'
    GZIP = 'GZIP'
    BZ = 'bz'


extractor_functions = {
    Extension.ZIP: unzip,
    Extension.GZIP: ungzip,
    Extension.BZ: unbz,
}


class Downloadable(metaclass=ABCMeta):

    @abstractmethod
    def url(self):
        pass

    @classmethod
    def is_compressed(cls):
        return True

    @classmethod
    def extension(cls):
        # returns the extension of the compressed file if is_compressed
        return Extension.ZIP

    @classmethod
    def compressed_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    def get_raw_asset_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def get_raw_asset_folder_path(self):
        root = self.get_raw_asset_root_path()
        return root.joinpath(self.raw_code())

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def maybe_download_raw_asset(self):
        folder_path = self.get_raw_asset_folder_path()
        if folder_path.is_dir() and all(
                folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('asset already exists. Skip downloading')
            return
        print("asset doesn't exist. Downloading...")
        if self.is_compressed():
            tmproot = Path(TMP_FOLDER)
            tmproot.mkdir(parents=True, exist_ok=True)
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            download(self.url(), tmpzip)
            extractor_functions[self.extension()](tmpzip, tmpfolder)
            if self.compressed_file_content_is_folder():
                tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, folder_path)
            shutil.rmtree(tmproot)
            print()
        else:
            # TODO check this chunk of code, is it needed
            #  especially since it has explicitly mention ratings.csv and would it work with
            #  the extractors and since all of our datasets classes have is_zip_file = True
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(self.url(), tmpfile)
            folder_path.mkdir(parents=True)
            shutil.move(tmpfile, folder_path.joinpath('ratings.csv'))
            shutil.rmtree(tmproot)
            print()
