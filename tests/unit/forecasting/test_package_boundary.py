import ast
from pathlib import Path

import industrial_forecasting

from industrial_forecasting.regimes import analyze_regime_diagnostics, recommend_forecasting_model
from industrial_forecasting.tuning import build_forecasting_stage_search_spaces, run_sequential_stage_tuning


def test_forecasting_has_no_dependency_on_industrial_or_fedot_shells():
    root = Path(industrial_forecasting.__file__).parent
    for source in root.rglob("*.py"):
        inside_neural = "neural" in source.relative_to(root).parts[:-1]
        for node in ast.walk(ast.parse(source.read_text("utf-8"))):
            if isinstance(node, ast.Import):
                names = tuple(entry.name for entry in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = (node.module,)
            else:
                continue
            for name in names:
                dependency = name.split(".")[0]
                assert dependency not in {"fedot", "golem", "benchmark"}, (source, name)
                if dependency == "fedot_ind":
                    assert inside_neural and name.startswith("fedot_ind.neural."), (source, name)
                if dependency == "torch":
                    assert inside_neural or source.name == "series.py", source


def test_regime_rules_and_sequential_tuning_work_without_the_industrial_catalogue():
    result = analyze_regime_diagnostics(list(range(30)))
    assert recommend_forecasting_model(result).primary_adapter
    spaces = build_forecasting_stage_search_spaces("lagged_ridge_forecaster")
    assert spaces
    assert all(item.metadata["search_space_source"] == "stage_plan_fallback" for item in spaces)
    tuned = run_sequential_stage_tuning(
        "lagged_ridge_forecaster",
        objective=lambda params: abs(params["alpha"] - 0.25),
        base_params={"alpha": 1.0, "window_size": 8},
        stage_updates={"forecast_head": {"alpha": 0.25}},
    )
    assert tuned.best_parameters["alpha"] == 0.25
    assert tuned.best_score == 0.0
