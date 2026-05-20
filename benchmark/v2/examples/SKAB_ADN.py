from benchmark.v2 import run_local_benchmark_preset

EXPERIMENT_DATE = '190526'
result = run_local_benchmark_preset(
    'skab',
    persist_on_run=True,
    output_dir=f'benchmark/results/v2_demo/m4_daily_{EXPERIMENT_DATE}',
)