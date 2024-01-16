import os
import unittest

from fedot_ind.core.architecture.settings.computational import backend_methods as np
import pandas as pd

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.operation.caching import DataCacher


class TestDataCacher(unittest.TestCase):

    def setUp(self):
        self.cache_folder = os.path.join(PROJECT_PATH, 'cache')
        self.data_cacher = DataCacher(data_type_prefix='data', cache_folder=self.cache_folder)
        self.data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    def test_hash_info(self):
        hashed_info = self.data_cacher.hash_info(name='data', data=self.data, add_info=['fedot',
                                                                                        'industrial',
                                                                                        'time series'])
        self.assertIsInstance(hashed_info, str)
        self.assertEqual(len(hashed_info), 20)

    def test_cache_data(self):
        hashed_info = self.data_cacher.hash_info(name='data', data=self.data)
        self.data_cacher.cache_data(hashed_info, self.data)
        cache_file = os.path.join(self.data_cacher.cache_folder, hashed_info + '.npy')

        self.assertTrue(os.path.isfile(cache_file))

    def test_load_data_from_cache(self):
        hashed_info = self.data_cacher.hash_info(name='data', data=self.data)
        self.data_cacher.cache_data(hashed_info, self.data)
        loaded_data = self.data_cacher.load_data_from_cache(hashed_info)

        self.assertIsInstance(loaded_data, np.ndarray)
        self.assertTrue((self.data.values == loaded_data).all())
