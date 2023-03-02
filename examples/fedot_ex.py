from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from core.repository.initializer_industrial_models import initialize_industrial_models
from tests.unit.repository.test_repo import initialize_uni_data


if __name__ == '__main__':
    initialize_industrial_models()
    train_data, test_data = initialize_uni_data()

    metrics = {}
    for extractor_name in ['topological_extractor', 'quantile_extractor', 'signal_extractor', 'recurrence_extractor']:
        pipeline = PipelineBuilder().add_node('data_driven_basic').add_node(extractor_name).add_node(
            'rf').build()
        model = Fedot(problem='classification', timeout=100, initial_assumption=pipeline, n_jobs=1)
        model.fit(train_data)
        model.predict(test_data)
        print(model.get_metrics())
        model.current_pipeline.show()
    print(metrics)