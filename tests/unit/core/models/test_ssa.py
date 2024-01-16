from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


def test_ssa():
    time_series = np.random.normal(size=30)
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=1))
    train_input = InputData(idx=np.arange(time_series.shape[0]),
                            features=time_series,
                            target=time_series,
                            task=task,
                            data_type=DataTypesEnum.ts)
    train_data, test_data = train_test_data_setup(train_input)

    with IndustrialModels():
        pipeline = PipelineBuilder().add_node('ssa_forecaster').build()
        pipeline.fit(train_data)
        ssa_predict = np.ravel(pipeline.predict(test_data).predict)
    assert ssa_predict is not None
