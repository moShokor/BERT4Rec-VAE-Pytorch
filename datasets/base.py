from collections import defaultdict

import numpy as np
from tqdm import tqdm

from .downloadable import Downloadable

tqdm.pandas()

from abc import *
import pickle


class AbstractDataset(Downloadable, metaclass=ABCMeta):
    def __init__(self, args, extractors):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split
        self.additional_inputs_extractors = extractors

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    def maybe_download_raw_asset(self):
        print(f'fetching the {self.code()} dataset ')
        super(AbstractDataset, self).maybe_download_raw_asset()

    @abstractmethod
    def load_ratings_df(self):
        pass

    @abstractmethod
    def get_sid2name(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_asset()
        df = self.load_ratings_df()
        sid2name = self.get_sid2name()
        # TODO note I added this one to remove duplicates just in case
        df = df.drop_duplicates(['uid', 'sid'])
        df = self.make_implicit(df)
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        # TODO you need to use the smap to correct the coresspondance in the rest of the code
        addmap = self.index_additional_inputs(sid2name, smap)
        # train, val, test, train_add, val_add, test_add = self.split_df(df, len(umap))
        train, val, test = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   # 'train_add': train_add,
                   # 'val_add': val_add,
                   # 'test_add': test_add,
                   'umap': umap,
                   'smap': smap,
                   'addmap': addmap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def index_additional_inputs(self, sid2name, smap):
        # translate the ids to correspond to the incremental ones in the smap
        sid2name = {smap.get(sid, sid): name for sid, name in sid2name.items()}
        # then use the sid2name to get the correspondence from each of the extractors
        sid2add = defaultdict(dict)
        for code, extractor in self.additional_inputs_extractors.items():
            sid2add[code] = extractor.build_correspondence(sid2name)
        return sid2add

    def make_implicit(self, df):
        print('Turning into implicit ratings')
        df = df[df['rating'] >= self.min_rating]
        # return df[['uid', 'sid', 'timestamp']]
        return df

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]

        return df

    def densify_index(self, df):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']))}
        smap = {s: i for i, s in enumerate(set(df['sid']))}
        # TODO maybe add other columns to the df to keep the original names of sid
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
            # additional_inputs = user_group.progress_apply(lambda d:
            #                                               {key: list(d.sort_values(by='timestamp')[key])
            #                                                for key in self.additional_inputs_names()})
            train, val, test, train_add, val_add, test_add = {}, {}, {}, {}, {}, {}
            for user in range(user_count):
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
                #     TODO maybe add these if the additional inputs doesn't have a coresspondence
                # items_add = additional_inputs[user]
                # train_add[user], val_add[user], test_add[user] = items_add[:-2], items_add[-2:-1], items_add[-1:]
            # return train, val, test, train_add, val_add, test_add
            return train, val, test
        elif self.args.split == 'holdout':
            print('Splitting')
            np.random.seed(self.args.dataset_split_seed)
            eval_set_size = self.args.eval_set_size

            # Generate user indices
            permuted_index = np.random.permutation(user_count)
            train_user_index = permuted_index[:-2 * eval_set_size]
            val_user_index = permuted_index[-2 * eval_set_size:  -eval_set_size]
            test_user_index = permuted_index[-eval_set_size:]

            # Split DataFrames
            train_df = df.loc[df['uid'].isin(train_user_index)]
            val_df = df.loc[df['uid'].isin(val_user_index)]
            test_df = df.loc[df['uid'].isin(test_user_index)]

            # DataFrame to dict => {uid : list of sid's}
            # TODO modify the holdout case similar to the leave one out to include the
            #  wiki-id
            train = dict(train_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            val = dict(val_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            test = dict(test_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            return train, val, test
        else:
            raise NotImplementedError

    def _get_preprocessed_root_path(self):
        root = self.get_raw_asset_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')
