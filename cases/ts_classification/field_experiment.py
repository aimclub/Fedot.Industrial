import pandas as pd
import pptk
import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np

block_model = pd.read_csv(r'D:\РАБОТЫ РЕПОЗИТОРИИ\Репозитории\Industiral\IndustrialTS\data\block_model\bm_ow.csv',
                          sep=';')
P = block_model.iloc[:, 1:4].values - block_model.iloc[:, -6:-3].values
block_model['AU'] = block_model['AU'].apply(lambda x: x.replace(',', '.'))
block_model['AU'] = block_model['AU'].astype(float)
block_model['AG'] = block_model['AG'].apply(lambda x: x.replace(',', '.'))
block_model['AG'] = block_model['AG'].astype(float)
#Block_AG = np.hstack((P,block_model['AG'].values.reshape(len(block_model['AG'].values),1)))
#x = np.hstack([P.reshape(-1),block_model['AU'].values])
#X = tl.tensor(x.reshape((202502, 202502, 202502,4)), dtype=tl.float32)
#weights, factors = parafac(X, rank=10)
v = pptk.viewer(P)
v.attributes(block_model['AU'])
v.attributes(block_model['AG'])
_ = 1
