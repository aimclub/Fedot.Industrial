"""Small execution-contract probes, NOT performance or model-quality evidence."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time


def slow_value(marker):
    Path(marker).write_text('started', encoding='ascii')
    time.sleep(6)
    return 1


def short_value(_):
    time.sleep(.1)
    return 1


def fail_worker_once(marker):
    path = Path(marker)
    if not path.exists():
        path.write_text('first attempt', encoding='ascii')
        os._exit(91)
    return 42


def run_probes(output_dir):
    from fedot_ind.core.optimizer.evaluation.contracts import EvaluationPlan, EvaluationTask, LocalBackend
    from fedot_ind.core.optimizer.evaluation.executors import execute_plan
    from fedot_ind.core.optimizer.evaluation.dask_config import DaskEvaluationPlan, DaskExecutionConfig
    from fedot_ind.core.optimizer.evaluation.dask_executor import DaskEvaluationSession, execute_dask_plan

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    task = (EvaluationTask('probe', 42),)
    sequential, = execute_plan(EvaluationPlan(task, candidate_timeout_seconds=.02), [0], short_value)
    process, = execute_plan(EvaluationPlan(task, LocalBackend.PROCESS, candidate_timeout_seconds=3,
                                           total_timeout_seconds=10), [str(root / 'process_started.txt')], slow_value)
    attempts = []
    with DaskEvaluationSession(1, DaskExecutionConfig(retries=1, memory_limit=2 * 1024**3)) as session:
        client = session.client
        result, = execute_dask_plan(DaskEvaluationPlan(task, candidate_timeout_seconds=30,
                                                       total_timeout_seconds=45),
                                    [str(root / 'worker_restart.txt')], fail_worker_once,
                                    session=session, on_attempt=attempts.append)
        client_running = client.status == 'running'
        value = client.submit(short_value, 0, pure=False).result(timeout=10)
    report = {
        'scope': 'synthetic_contract_probes_not_training_or_speed_measurements',
        'sequential': {'status': sequential.status.value, 'seconds': sequential.duration_seconds,
                       'cooperative_limit_seconds': .02},
        'process': {'status': process.status.value, 'seconds': process.duration_seconds,
                    'callback_started': (root / 'process_started.txt').exists(), 'callback_sleep_seconds': 6},
        'dask': {'status': result.status.value, 'value': result.value,
                 'attempts': [{**asdict(item), 'status': item.status.value} for item in attempts],
                 'caller_client_running_after_evaluation': client_running,
                 'caller_client_still_usable': value == 1, 'closed_only_after_owner_exit': client.status == 'closed'},
    }
    (root / 'contracts.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('examples/artifacts/cloud_bundle/evolution_parallelism/contracts'))
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'action': 'execution_contract_probes', 'execute': False}))
        return
    print(json.dumps(run_probes(args.output_dir), indent=2), flush=True)


if __name__ == '__main__':
    main()
