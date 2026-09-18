import numpy as np

from benchmark.industrial import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    TaskType,
    run_tsc_benchmark_suite,
)


TRAIN_FEATURES = np.array(
    [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [1.0, 0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
        [1.0, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0],
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    ],
    dtype=float,
)
TRAIN_TARGET = np.array([0, 0, 1, 1, 0, 1], dtype=object)
TEST_FEATURES = np.array(
    [
        [0.2, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6],
        [8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5],
    ],
    dtype=float,
)
TEST_TARGET = np.array([0, 1], dtype=object)


def main() -> None:
    dataset_spec = DatasetSpec(
        benchmark='in_memory_tsc',
        dataset_name='riemann_feature_toy',
        adapter_options={
            'record': {
                'train_features': TRAIN_FEATURES,
                'train_target': TRAIN_TARGET,
                'test_features': TEST_FEATURES,
                'test_target': TEST_TARGET,
            }
        },
    )

    models = (
        ModelSpec(adapter_name='majority_class', display_name='MajorityClass'),
        ModelSpec(
            adapter_name='sklearn_classifier',
            display_name='Riemann+Logit',
            params={
                'generator_name': 'riemann_extractor',
                'generator_params': {
                    'feature_mode': 'mdm',
                    'mdm_centroid_scope': 'global',
                },
                'classifier_name': 'logistic_regression',
                'classifier_params': {'max_iter': 5000},
            },
        ),
        ModelSpec(
            adapter_name='sklearn_classifier',
            display_name='Riemann+SVC',
            params={
                'generator_name': 'riemann_extractor',
                'generator_params': {
                    'feature_mode': 'both',
                    'mdm_centroid_scope': 'global',
                },
                'classifier_name': 'svc',
            },
        ),
    )

    config = BenchmarkSuiteConfig(
        task_type=TaskType.TS_CLASSIFICATION,
        datasets=(dataset_spec,),
        models=models,
        metrics=('accuracy', 'balanced_accuracy', 'f1_macro'),
        artifact_spec=ArtifactSpec(output_dir='benchmark/results/industrial_demo/riemann_feature_suite'),
        run_spec=RunSpec(run_name='riemann_feature_suite_demo', primary_metric='accuracy'),
    )

    result = run_tsc_benchmark_suite(config)
    print(result.aggregate_report.leaderboard_rows)


if __name__ == '__main__':
    main()
