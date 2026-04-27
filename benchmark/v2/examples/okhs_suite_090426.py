from benchmark.v2 import run_local_benchmark_preset

EXPERIMENT_DATE = '090426'
PRESET_NAME = 'okhs_smoothing'
result = run_local_benchmark_preset(
    preset_name=PRESET_NAME,
    subset='daily',
    sample_size=5,
    persist_on_run=True,
    output_dir=f'benchmark/results/v2_demo/{PRESET_NAME}_{EXPERIMENT_DATE}',
)
