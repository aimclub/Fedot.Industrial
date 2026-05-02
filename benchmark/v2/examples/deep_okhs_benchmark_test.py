from benchmark.v2 import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    TaskType,
    run_forecasting_benchmark_suite,
)

# Назовем эксперимент тестовым
EXPERIMENT_DATE = 'test_run'

# Выберем только один сабсет для быстрого теста
SUBSET = 'Daily'

# Настраиваем датасет: берем ровно 1 временной ряд
DATASETS = (
    DatasetSpec(
        benchmark='m4',
        dataset_name=f'm4_{SUBSET}_deep_test',
        subset=SUBSET,
        sample_size=1,  # Строго 1 ряд для проверки работоспособности
        random_seed=42,
        adapter_options={'use_local_files': True},
    ),
)

# Настраиваем нашу новую модель
FORECASTING_MODELS = (
    ModelSpec(
        adapter_name='deep_okhs_fdmd_forecaster',
        display_name='Deep_OKHS_fDMD',
        params={
            # Гиперпараметры классической части
            'q': 0.7,
            'n_modes': 5,
            'window_size': 20,
            
            # Гиперпараметры нейросети (уменьшены для быстрого теста)
            'latent_dim': 16,
            'ae_epochs': 15,  # Всего 15 эпох, чтобы убедиться, что градиенты текут
            'ae_learning_rate': 1e-3,
            'hidden_layers': [32, 32],
            'alpha_adjoint': 1.0,
            'beta_rec': 1.0,
        },
    ),
    # # Опционально: добавим классический OKHS для сравнения результатов "из коробки"
    # ModelSpec(
    #     adapter_name='okhs_fdmd_forecaster',
    #     display_name='Classic_OKHS_fDMD',
    #     params={
    #         'q': 0.7,
    #         'n_modes': 5,
    #         'window_size': 20,
    #     },
    # ),
)

# Собираем конфигурацию бенчмарка
config = BenchmarkSuiteConfig(
    task_type=TaskType.FORECASTING,
    datasets=DATASETS,
    models=FORECASTING_MODELS,
    metrics=('rmse', 'mae'), # Для быстрого теста можно ограничиться базовыми метриками
    artifact_spec=ArtifactSpec(
        output_dir=f'benchmark/results/v2_demo/deep_okhs_test_{EXPERIMENT_DATE}',
        persist_on_run=True,
    ),
    run_spec=RunSpec(
        run_name='deep_okhs_test_suite',
        primary_metric='rmse',
        show_progress=True,
        progress_leave=False,
        progress_log_errors=True,
        progress_log_summaries=True,
    ),
)

if __name__ == '__main__':
    print("Запуск тестового пайплайна для Deep OKHS...")
    result = run_forecasting_benchmark_suite(config)
    print(f"\nТест завершен! Результаты сохранены в {config.artifact_spec.output_dir}")
    print(f"Успешных прогонов: {result.aggregate_report.status_counts.get('success', 0)}")