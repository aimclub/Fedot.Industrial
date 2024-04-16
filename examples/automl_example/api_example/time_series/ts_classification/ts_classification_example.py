from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot_ind.tools.example_utils import industrial_common_modelling_loop


if __name__ == "__main__":
    dataset_name = 'Handwriting'
    finetune = True
    initial_assumption = PipelineBuilder().add_node('channel_filtration').add_node('quantile_extractor').add_node('rf')
    metric_names = ('f1', 'accuracy', 'precision', 'roc_auc')
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=5,
                      initial_assumption=initial_assumption,
                      n_jobs=2,
                      logging_level=20)

    model, labels, metrics = industrial_common_modelling_loop(api_config=api_config,
                                                              dataset_name=dataset_name,
                                                              finetune=finetune)
    print(metrics)
