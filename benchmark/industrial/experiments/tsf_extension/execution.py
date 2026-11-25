"""Launch the extension with separate per-stage processes and frozen B receipts."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from subprocess import TimeoutExpired

from ..cyclic_mutations.history import atomic_text
from ..evolution_parallelism.supervisor import run_guarded
from ...protocols.assets import LocalAsset, read_checked_asset
from ...protocols.json_io import parse_protocol_json, read_protocol_json
from ...protocols.schema import ProtocolError
from .inputs import _digest, _json_bytes
from .runtime_contract import ACCEPTED_PILOT, Role, load_window_request, read_manifest


@dataclass(frozen=True)
class Executor:
    python: Path
    source: Path
    receipt: Path


def _write(path, value):
    atomic_text(path, _json_bytes(value).decode())


def _require_tser(path):
    root = Path(path)
    if read_protocol_json(root / 'run.json')['device'] != 'cuda:0':
        raise ProtocolError('Expected regression acceptance on CUDA')
    allowed = ('Tecator', 'BoronConcentration', 'NaturalGasPricesSentiment', 'AppliancesEnergy')
    receipts = read_protocol_json(root / 'assess_status.json')
    completed = {(row['dataset'], row['stage']) for row in receipts if row['exit_code'] == 0}
    if completed != {(name, seed) for name in allowed for seed in ('42', '123')} or len(receipts) != 8:
        raise ProtocolError('The extended TSF protocol depends on eight complete TSER searches and assessments')
    for name, seed in completed:
        search = read_protocol_json(root / name / seed / 'search.json')
        if search['status'] != 'complete' or search['objective'] != 'rmse' or search['official_test_used'] is not False:
            raise ProtocolError('A TSER search did not complete the required TRAIN-only protocol')
        result = read_protocol_json(root / name / seed / 'assessment.json')
        results = result['results']
        if (result['official_test_used'] is not False or not result.get('train_mean_baseline')
                or not results or any(row['status'] != 'succeeded' for row in results)):
            raise ProtocolError('A TSER assessment did not complete the required controls')


def _b_profile(previous, expected):
    _, content = read_checked_asset(previous, LocalAsset('manifest.json', expected))
    previous_manifest = parse_protocol_json(content.decode())
    if previous_manifest['schema'] != 'industrial_tsf_variant_b@1' or previous_manifest['official_test_used'] is not False:
        raise ProtocolError('Expected the previous TRAIN-only variant B manifest')
    return previous_manifest['config']


def _artifact(bundle, case, seed, role):
    directory = Path(bundle) / case['collection']['id'].replace(' ', '_') / role.value
    request = load_window_request(directory, case, seed=seed, role=role)
    return dict(path=str((directory / f'{request.series_id}.npy').resolve()), length=len(request.values),
                sha256=case[f'{role.value}_sha256'])


def _stage(executor, bundle, config, case, task, work, stage, *, recipes=None, winners=()):
    work.mkdir(parents=True, exist_ok=False)
    role = Role.ASSESSMENT if stage == 'assess' else Role.DEVELOPMENT
    request = dict(
        config=config, case={'group': case['collection']['id'], 'season': case['collection']['lag'],
                             'search': case['search'], 'assessment': case['assessment']},
        task=task, pilot_receipt=str(executor.receipt.resolve()), pilot_sha256=ACCEPTED_PILOT,
        official_test_used=False, stage=stage, control_lags=case['collection']['control_lags'],
        payload=_artifact(bundle, case, task['seed'], role),
    )
    if recipes is not None:
        request['recipes'] = str(recipes.resolve())
    if stage == 'assess':
        request['winners'] = [str(path.resolve()) for path in winners]
        request['development'] = _artifact(bundle, case, task['seed'], Role.DEVELOPMENT)
    plan = work / 'stage.json'
    _write(plan, request)
    command = (
        str(executor.python.absolute()), '-B', str(Path(__file__).with_name('legacy_worker.py').resolve()),
        '--runtime-root', str(executor.source.resolve()), '--plan', str(plan.resolve()),
        '--output', str(work.resolve()),
    )
    with (work / 'stage.log').open('xb') as logs:
        try:
            code = run_guarded(command, timeout_seconds=config['process_seconds'],
                               search_seconds=config['search_seconds'], output_stream=logs)
        except TimeoutExpired:
            code = 124
    complete = code == 0 and read_protocol_json(work / 'shutdown.json')['confirmed']
    if complete and stage == 'search':
        complete = read_protocol_json(work / 'search_summary.json')['status'] == 'complete'
    if complete and stage == 'assess':
        scores = read_protocol_json(work / 'assessment.json')['scores']
        complete = bool(scores) and all(row['status'] == 'succeeded' for row in scores)
    return dict(stage=stage, task=task, exit_code=code, complete=complete)


def _paths(root, case):
    return Path(root) / case['collection']['id'].replace(' ', '_') / case['id']


def _task(case, seed):
    return dict(case=case['id'], horizon=case['collection']['horizon'], seed=seed)


def _accepted(case_results, case, seed):
    return case_results.get((case['collection']['id'], case['id'], seed), False)


def _completed(source, mode):
    rows = read_protocol_json(Path(source) / f'{mode}_status.json')
    return {(row['collection'], row['task']['case'], row['task']['seed']): row['complete']
            for row in rows if row['stage'] == 'search'}


def run(mode, *, bundle, previous, executor, tser, output, admission=None):
    _require_tser(tser)
    manifest, identity = read_manifest(bundle)
    config = _b_profile(previous, manifest['previous_manifest_sha256'])
    read_checked_asset(executor.receipt.parent, LocalAsset(executor.receipt.name, ACCEPTED_PILOT))
    output = Path(output)
    if mode == 'run':
        if read_protocol_json(Path(admission) / 'extension_run.json')['manifest_sha256'] != identity:
            raise ProtocolError('Admitted cases belong to a different extension matrix')
        prior = _completed(admission, 'admit')
        if len(prior) != 3 or not all(prior.values()):
            raise ProtocolError('One complete search per collection is required before the series')
    else:
        prior = {}
    if mode == 'assess':
        if read_protocol_json(output / 'extension_run.json')['manifest_sha256'] != identity:
            raise ProtocolError('Search results belong to a different extension matrix')
        complete = _completed(output, 'run')
        if len(complete) != 24 or not all(complete.values()):
            raise ProtocolError('All 24 searches must complete before opening late windows')
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write(output / 'extension_run.json', dict(manifest_sha256=identity, pilot_sha256=ACCEPTED_PILOT))

    records, seen = [], set()

    def record(case, row):
        records.append({'collection': case['collection']['id'], **row})
        _write(output / f'{mode}_status.json', records)

    for case in manifest['rows']:
        collection = case['collection']['id']
        home = _paths(output, case)
        old = _paths(admission, case) if admission is not None else None
        calibration_seed = config['calibration_seed']
        if mode == 'admit':
            if collection in seen:
                continue
            seen.add(collection)
        recipes = home / 'calibrate' / 'recipes.json'
        if mode == 'assess':
            if not recipes.is_file():
                recipes = old / 'calibrate' / 'recipes.json'
            winners = []
            for seed in case['seeds']:
                path = home / f'seed-{seed}' / 'winner.json'
                winners.append(path if path.is_file() else old / f'seed-{seed}' / 'winner.json')
            row = _stage(executor, bundle, config, case, _task(case, calibration_seed),
                         home / 'assessment', 'assess', recipes=recipes, winners=winners)
            record(case, row)
            continue
        reuse_calibration = old is not None and _accepted(prior, case, calibration_seed)
        if reuse_calibration:
            recipes = old / 'calibrate' / 'recipes.json'
        else:
            row = _stage(executor, bundle, config, case, _task(case, calibration_seed), home / 'calibrate', 'calibrate')
            record(case, row)
            if not row['complete']:
                continue
        seeds = (calibration_seed,) if mode == 'admit' else case['seeds']
        for seed in seeds:
            if _accepted(prior, case, seed):
                record(case, dict(stage='search', task=_task(case, seed), exit_code=0, complete=True, reused=True))
                continue
            row = _stage(executor, bundle, config, case, _task(case, seed), home / f'seed-{seed}',
                         'search', recipes=recipes)
            record(case, row)
    return 0 if records and all(row['complete'] for row in records) else 1
