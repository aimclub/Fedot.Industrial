from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.api.utils.input_data import init_input_data
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


dataset_name = 'AppliancesEnergy'
train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
train_data = init_input_data(train_data[0], train_data[1], task='regression')
test_data = init_input_data(test_data[0], test_data[1], task='regression')
with IndustrialModels():
    pipeline = PipelineBuilder().add_node('quantile_extractor').add_node('fedot_regr', params={'timeout': 2}).build()
    pipeline.fit(train_data)
    pred = pipeline.predict(test_data).predict
    print(pred)
