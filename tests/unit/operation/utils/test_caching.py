import os.path

import pandas as pd

from core.architecture.utils.utils import PROJECT_PATH
from core.operation.utils.caching import DataCacher

folder_path = os.path.join(PROJECT_PATH, 'cache')


class TestCaching:

    df_name = 'test_df'
    df_state = 'test_df_state'

    def basic_cacher(self):
        return DataCacher(data_type_prefix='test', cache_folder=folder_path)

    def basic_dataframe(self):
        return pd.DataFrame({'a': [1] * 100000, 'b': [4] * 100000})

    def test_hash_info(self):
        cacher = self.basic_cacher()
        hashed_info = cacher.hash_info(name=self.df_name, state=self.df_state, data=self.basic_dataframe())
        assert hashed_info is not None

    def test_load_data_from_cache(self):
        cacher = self.basic_cacher()
        hashed_info = cacher.hash_info(name=self.df_name, state=self.df_state, data=self.basic_dataframe())
        cacher.cache_data(hashed_info, self.basic_dataframe())
        data = cacher.load_data_from_cache(hashed_info)
        assert isinstance(data, pd.DataFrame)
        assert data.equals(self.basic_dataframe())

    def test_cache_data(self):
        cacher = self.basic_cacher()
        hashed_info = cacher.hash_info(name=self.df_name, state=self.df_state, data=self.basic_dataframe())
        cacher.cache_data(hashed_info, self.basic_dataframe())
        filename = hashed_info + '.pkl'
        assert os.path.isfile(os.path.join(folder_path, filename))
        os.remove(os.path.join(folder_path, filename))
