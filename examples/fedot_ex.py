import numpy as np
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from core.repository.initializer_industrial_models import initialize_industrial_models
from tests.unit.repository.test_repo import initialize_uni_data


if __name__ == '__main__':
    np.random.seed(0)
    initialize_industrial_models()
    train_data, test_data = initialize_uni_data()

    metrics = {}
    pipeline = PipelineBuilder().add_node('data_driven_basic').add_node('topological_extractor')\
        .add_node('rf').build()
    model = Fedot(problem='classification', timeout=60, initial_assumption=pipeline, n_jobs=1, metric=['f1'])
    model.fit(train_data)
    model.predict(test_data)
    print(model.get_metrics())
    model.current_pipeline.print_structure()
    print(metrics)