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
    pipeline = PipelineBuilder().add_node('data_driven_basic').add_node('quantile_extractor') \
        .add_node('logit').build()
    model = Fedot(problem='classification', logging_level=20, timeout=30, initial_assumption=pipeline, n_jobs=1,
                  metric=['roc_auc'])
    model.fit(train_data)
    model.predict(test_data)
    print(model.get_metrics())
    model.current_pipeline.print_structure()
    print(metrics)
    model.current_pipeline.show()
    _ = 1