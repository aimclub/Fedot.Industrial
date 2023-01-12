import os
import unittest

import pandas as pd

from core.architecture.utils.utils import PROJECT_PATH
from core.operation.utils.caching import DataCacher


class TestDataCacher(unittest.TestCase):

    def setUp(self):
        self.cache_folder = os.path.join(PROJECT_PATH, 'cache')
        self.data_cacher = DataCacher(data_type_prefix='data', cache_folder=self.cache_folder)
        self.data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    def test_hash_info(self):
        hashed_info = self.data_cacher.hash_info(name='data', data=self.data)
        self.assertIsInstance(hashed_info, str)
        self.assertEqual(len(hashed_info), 10)

    def test_cache_data(self):
        hashed_info = self.data_cacher.hash_info(name='data', data=self.data)
        self.data_cacher.cache_data(hashed_info, self.data)
        cache_file = os.path.join(self.data_cacher.cache_folder, hashed_info + '.pkl')
        self.assertTrue(os.path.isfile(cache_file))

    def test_load_data_from_cache(self):
        hashed_info = self.data_cacher.hash_info(name='data', data=self.data)
        self.data_cacher.cache_data(hashed_info, self.data)
        loaded_data = self.data_cacher.load_data_from_cache(hashed_info)
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertTrue(loaded_data.equals(self.data))
