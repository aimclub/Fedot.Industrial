"""Compare legacy and planned proposals on identical tasks, inputs and seeds."""

from collections import Counter
import json
from pathlib import Path
from time import perf_counter

from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.pipelines.verification import verifier_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from golem.core.optimisers.adaptive.operator_agent import RandomAgent
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimizer import GraphGenerationParams

from fedot_ind.core.optimizer.configuration import MutationPlanningPolicy
from fedot_ind.core.optimizer.graph_validation import IndustrialGraphVerifier
from fedot_ind.core.optimizer.mutation_runtime import PlannedMutation
from fedot_ind.core.optimizer.mutation_planning import RawInputSpec
from fedot_ind.core.optimizer.randomness import CandidateRandomness, operator_random_scope


def summarize_attempts(records):
    """Keep validity, changed-graph yield and diversity denominators explicit."""
    records = tuple(records)
    valid = [record for record in records if record["valid"]]
    changed = [record for record in valid if record["changed"]]
    unique = {record["graph_id"] for record in changed}
    operations = Counter(name for record in changed for name in record["operations"])
    reasons, stages, families = Counter(), Counter(), Counter()
    truncated = 0
    for record in records:
        execution = record.get("execution") or {}
        planning = execution.get("planning") or {}
        reasons.update(planning.get("reason_counts", {}))
        families.update(planning.get("family_counts", {}))
        stages.update(item["stage"] for item in planning.get("decisions", [])
                      if item["status"] == "rejected")
        truncated += int(planning.get("truncated", False))
    return {
        "attempts": len(records), "valid_proposals": len(valid), "changed_valid_proposals": len(changed),
        "valid_ratio": len(valid) / len(records) if records else None,
        "changed_valid_ratio": len(changed) / len(records) if records else None,
        "unique_changed_graphs": len(unique),
        "uniqueness_among_changed": len(unique) / len(changed) if changed else None,
        "distinct_operations": len(operations), "operation_frequency": dict(sorted(operations.items())),
        "status_counts": dict(Counter(record["status"] for record in records)),
        "elapsed_seconds": sum(record["elapsed_seconds"] for record in records),
        "candidate_rejection_reasons": dict(sorted(reasons.items())),
        "candidate_rejection_stages": dict(sorted(stages.items())),
        "examined_edit_families": dict(sorted(families.items())),
        "truncated_searches": truncated,
    }


def _environment(scenario, settings):
    kind = TaskTypesEnum[scenario["task"]]
    task = Task(kind, TsForecastingParams(forecast_length=4)) if kind is TaskTypesEnum.ts_forecasting else Task(kind)
    requirements = PipelineComposerRequirements(
        primary=scenario["operations"], secondary=scenario["operations"],
        max_depth=settings["max_depth"], max_arity=settings["max_arity"],
        n_jobs=1, show_progress=False, keep_history=False)
    advisor = PipelineChangeAdvisor(task)
    params = GraphGenerationParams(adapter=PipelineAdapter(), advisor=advisor,
                                   node_factory=PipelineOptNodeFactory(requirements, advisor))
    # The same verifier for both mechanisms isolates the proposal difference.
    params.verifier = IndustrialGraphVerifier(adapter=params.adapter, task_type=task.task_type,
                                              legacy_verifier=verifier_for_task(kind, params.adapter))
    node = None
    for name in scenario["initial_chain"]:
        node = PipelineNode(name, nodes_from=[node] if node else None)
    graph = params.adapter.adapt(Pipeline(node))
    if not params.verifier(graph):
        raise ValueError(f"Invalid initial chain for {scenario['name']}: {params.verifier.last_report.error_codes}")
    return requirements, params, graph


def compare_mutations(settings=None):
    """Inspectable proposal study, not a measurement of fitness or CUDA speed.

    Each attempt starts from the same graph. GOLEM gets one structural attempt,
    not an unreported retry advantage. Timing includes planning and verification.
    """
    if settings is None:
        settings = json.loads(Path(__file__).with_name("defaults.json").read_text(encoding="utf-8"))
    if settings.get("schema_version") != "industrial_mutation_comparison@1":
        raise ValueError("Unsupported mutation comparison schema")
    for field in ("attempts_per_seed", "max_candidates", "max_depth", "max_nodes", "max_arity"):
        if type(settings[field]) is not int or settings[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if not settings["seeds"] or any(type(seed) is not int or seed < 0 for seed in settings["seeds"]):
        raise ValueError("seeds must contain nonnegative integers")
    records, summaries = [], []
    actions = [MutationTypesEnum.single_change, MutationTypesEnum.single_add, MutationTypesEnum.single_drop]
    for scenario in settings["scenarios"]:
        requirements, params, graph = _environment(scenario, settings)
        for mode in ("legacy", "planned"):
            algorithm = GPAlgorithmParameters(
                mutation_types=actions, mutation_prob=1.0, variable_mutation_num=False,
                max_num_of_operator_attempts=1,
                adaptive_mutation_type=RandomAgent(actions=actions, probs=[1 / len(actions)] * len(actions)))
            operator = Mutation(algorithm, requirements, params)
            if mode == "planned":
                raw_input = RawInputSpec(**scenario["raw_input"]) if "raw_input" in scenario else None
                operator = PlannedMutation(operator, params, MutationPlanningPolicy(
                    max_candidates=settings["max_candidates"], max_nodes=settings["max_nodes"]), raw_input=raw_input)
            for seed in settings["seeds"]:
                for attempt in range(settings["attempts_per_seed"]):
                    parent = Individual(graph)
                    started = perf_counter()
                    address = CandidateRandomness(seed, "mutation_comparison", attempt)
                    with operator_random_scope(address.seed):
                        child = operator(parent)
                    elapsed = perf_counter() - started
                    candidate = child.graph if isinstance(child, Individual) else None
                    valid = candidate is not None and params.verifier(candidate)
                    changed = candidate is not None and candidate != graph
                    status = operator.last_execution.status.value if mode == "planned" else (
                        "accepted" if valid and changed else "no_changed_proposal")
                    records.append({
                        "scenario": scenario["name"], "task": scenario["task"], "mode": mode,
                        "seed": seed, "attempt": attempt, "valid": bool(valid), "changed": bool(changed),
                        "graph_id": candidate.descriptive_id if candidate is not None else None,
                        "operations": sorted(node.content["name"] for node in candidate.nodes) if candidate else [],
                        "status": status, "elapsed_seconds": elapsed,
                        "execution": operator.last_execution.to_record() if mode == "planned" else None,
                    })
            subset = [r for r in records if r["scenario"] == scenario["name"] and r["mode"] == mode]
            summaries.append({"scenario": scenario["name"], "mode": mode, **summarize_attempts(subset)})
    return {"schema_version": "industrial_mutation_comparison_report@1", "settings": settings,
            "scope": "structural_proposals_without_training", "summaries": summaries, "records": records,
            "acceptance": {"status": "not_assessed", "reason": "quality_and_compute_cost_not_measured"}}


def write_comparison(report, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = ["# Structural Mutation Diagnostics", "", "This measures graph proposals, not model quality.",
            "Acceptance: not_assessed. Graph counts are diagnostics, not an acceptance criterion.", "",
            "| Scenario | Mode | Attempts | Changed Valid | Unique Changed | Distinct Operations | Seconds |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in report["summaries"]:
        rows.append(f"| {item['scenario']} | {item['mode']} | {item['attempts']} | "
                    f"{item['changed_valid_ratio']:.3f} | {item['unique_changed_graphs']} | "
                    f"{item['distinct_operations']} | {item['elapsed_seconds']:.3f} |")
    rows += ["", "## Candidate rejection and budget diagnostics", ""]
    for item in report['summaries']:
        rows += [f"### {item['scenario']} / {item['mode']}",
                 f"Truncated searches: {item['truncated_searches']}",
                 "Examined families: " + json.dumps(item['examined_edit_families'], sort_keys=True),
                 "Rejection stages: " + json.dumps(item['candidate_rejection_stages'], sort_keys=True),
                 "Reasons: " + json.dumps(item['candidate_rejection_reasons'], sort_keys=True), ""]
    (output / "comparison.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output / "comparison.json"
