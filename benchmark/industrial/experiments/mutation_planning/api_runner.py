"""Explicit end-to-end verification of the opt-in mutation planner."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping

from benchmark.industrial.experiments.evolution_observability.contracts import (
    load_default_evolution_baseline_config,
)
from benchmark.industrial.experiments.evolution_observability.runner import (
    build_fedot_industrial_config,
    run_evolution_baseline_suite,
)
from fedot_ind.core.optimizer.configuration import (
    EvolutionConfig,
    normalize_evolution_config,
)


def load_api_verification_policy() -> EvolutionConfig:
    path = Path(__file__).with_name("api_defaults.json")
    return normalize_evolution_config(json.loads(path.read_text(encoding="utf-8")))


def build_api_verification_config(
        api_config: Mapping[str, object],
        policy: EvolutionConfig,
        *,
        without_prediction_cache: bool = False,
) -> dict[str, object]:
    """Keep the scenario contract and caller-owned configuration unchanged."""
    config = deepcopy(dict(api_config))
    strategy = config["automl_config"]["optimisation_strategy"]
    diagnostics_dir = strategy["optimisation_strategy"]["diagnostics_output_dir"]
    resolved_policy = replace(policy, execution_policy=replace(
        policy.execution_policy, diagnostics_output_dir=diagnostics_dir,
    ))
    strategy["optimisation_strategy"] = resolved_policy.to_record()
    if without_prediction_cache:
        config["learning_config"]["learning_strategy_params"]["use_predictions_cache"] = False
    return config


def _industrial_model(config: Mapping[str, object]):
    from fedot_ind.api.main import FedotIndustrial

    return FedotIndustrial(**dict(config))


def run_api_verification(
        output_dir: str | Path,
        *,
        execute: bool = False,
        legacy: bool = False,
        without_prediction_cache: bool = False,
        compatibility_worktree: str | Path | None = None,
        model_factory: Callable[[Mapping[str, object]], object] = _industrial_model,
):
    """Save exact configurations; train only on an explicit execution request."""
    root = Path(output_dir)
    suite = load_default_evolution_baseline_config()
    compatibility = None
    if compatibility_worktree is not None:
        from fedot_ind.integration.fedot.compat import audit_compatibility_worktree
        compatibility = audit_compatibility_worktree(compatibility_worktree)
        if execute:
            import fedot
            expected = Path(compatibility["worktree"]) / "fedot"
            if not Path(fedot.__file__).resolve().is_relative_to(expected.resolve()):
                raise ValueError("Active FEDOT import does not belong to the audited compatibility worktree")
    if execute and any((root / scenario.name / "diagnostics" / "evolution_summary.json").exists()
                       for scenario in suite.scenarios):
        raise ValueError("Use a fresh output directory: previous diagnostics cannot verify a new training run")
    root.mkdir(parents=True, exist_ok=True)
    policy = load_api_verification_policy()
    if legacy:
        policy = replace(policy, mutation_planning=replace(policy.mutation_planning, enabled=False))
    settings = {
        "execute": execute,
        "without_prediction_cache": without_prediction_cache,
        "evolution_policy": policy.to_record(),
        "fedot_compatibility": compatibility,
    }
    (root / "verification_settings.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    for scenario in suite.scenarios:
        config = build_api_verification_config(
            build_fedot_industrial_config(suite, scenario, root), policy,
            without_prediction_cache=without_prediction_cache,
        )
        (root / f"{scenario.name}_api_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )

    def configured_factory(base_config):
        return model_factory(build_api_verification_config(
            base_config, policy, without_prediction_cache=without_prediction_cache,
        ))

    return run_evolution_baseline_suite(
        suite, root, execute=execute, model_factory=configured_factory,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify mutation planning through the Industrial API")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--without-prediction-cache", action="store_true")
    parser.add_argument("--compatibility-worktree", type=Path)
    args = parser.parse_args(argv)
    results = run_api_verification(
        args.output_dir, execute=args.execute, legacy=args.legacy,
        without_prediction_cache=args.without_prediction_cache,
        compatibility_worktree=args.compatibility_worktree,
    )
    print(json.dumps([result.to_record() for result in results], indent=2))
    return int(any(result.status not in {"planned", "succeeded"} for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
