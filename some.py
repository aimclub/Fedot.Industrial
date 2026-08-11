from benchmark.industrial import run_local_benchmark_preset

result = run_local_benchmark_preset(
    "fusion_over_raw_smoke",
    output_dir="benchmark/results/industrial_presets/fusion_over_raw_smoke",
    persist_on_run=True,
)
print(result.run_id, result.aggregate_report.primary_metric)