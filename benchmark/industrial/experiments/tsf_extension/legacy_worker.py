"""Isolated stage worker for the accepted variant B source tree."""

from argparse import ArgumentParser
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import sys

import numpy as np


@contextmanager
def _weekly_ets(job):
    from benchmark.industrial.experiments.tsf_variant_b import runtime
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    class WeeklyForecast(runtime.StatisticalForecast):
        def fit(self, features):
            if self.kind != 'ets':
                return super().fit(features)
            history = np.asarray(features).reshape(-1)
            self.center, self.scale = float(history.mean()), float(history.std()) or 1.
            self.model = ExponentialSmoothing((history - self.center) / self.scale, trend='add', damped_trend=True,
                                              seasonal=None, initialization_method='estimated').fit()
            return self

    if job.period == 1:
        with patch.object(runtime, 'StatisticalForecast', WeeklyForecast):
            yield
    else:
        yield


def evaluate_case(job):
    from benchmark.industrial.experiments.tsf_variant_b.runtime import evaluate

    with _weekly_ets(job):
        return evaluate(job)


def evaluate_control(payload):
    from benchmark.industrial.experiments.tsf_variant_b.assessment import assess_request

    method, job = payload
    if method != 'sarima':
        return evaluate_case(job)
    if job.period != 1:
        return assess_request(payload)
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from benchmark.industrial.experiments.tsf_variant_b.assessment import DEFAULTS, read

    options = read(DEFAULTS)['sarima']
    return fit_nonseasonal_arima(job, options, SARIMAX)


def fit_nonseasonal_arima(job, options, build):
    folds = []
    for origin in job.origins:
        start = perf_counter()
        fit = build(job.history[:origin], order=tuple(options['order']),
                    trend=options['trend']).fit(disp=False, maxiter=options['maxiter'])
        values = np.asarray(fit.forecast(job.horizon), dtype=float)
        if values.shape != (job.horizon,) or not np.isfinite(values).all():
            raise ValueError('Expected a finite nonseasonal forecast')
        folds.append(dict(origin=origin, values=values.tolist(), seconds=perf_counter() - start, trace=[],
                          converged=bool(fit.mle_retvals['converged'])))
    return dict(folds=folds, device='cpu')


def read_snapshot(metadata):
    payload = Path(metadata['path']).read_bytes()
    if sha256(payload).hexdigest() != metadata['sha256']:
        raise ValueError('Frozen TRAIN input changed before the worker opened it')
    values = np.load(BytesIO(payload), allow_pickle=False)
    if (values.shape != (metadata['length'],) or values.dtype != np.dtype('<f8')
            or not np.isfinite(values).all()):
        raise ValueError('Frozen TRAIN input differs from its declared shape or type')
    return values


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('--runtime-root', type=Path, required=True)
    cli.add_argument('--plan', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    sys.path.insert(0, str(args.runtime_root.resolve(strict=True)))
    from benchmark.industrial.experiments.tsf_variant_b import admission, assessment, baselines, search
    from benchmark.industrial.experiments.tsf_variant_b.protocol import digest, read, write
    from benchmark.industrial.experiments.tsf_variant_b.series import revision
    from fedot_ind.core.optimizer.evaluation.dask_executor import DaskEvaluationSession

    plan = read(args.plan)
    gate = Path(plan['pilot_receipt'])
    if (digest(gate) != plan['pilot_sha256'] or revision() != read(gate)['code']
            or plan['official_test_used'] is not False):
        raise ValueError('Legacy B executor, receipt or data role changed')
    config, case, task = (plan[key] for key in ('config', 'case', 'task'))
    values = read_snapshot(plan['payload'])
    stage = plan['stage']
    execution = admission.pool_config(config)
    with DaskEvaluationSession(execution.device_pool.worker_count, execution) as session:
        with patch.object(search, 'evaluate', evaluate_case), patch.object(baselines, 'evaluate', evaluate_case):
            if stage == 'calibrate':
                search.calibrate(session, args.output, case, task, values, config)
            else:
                recipes = read(plan['recipes'])
                if (recipes['origins'] != case['search'] or recipes['task']['case'] != task['case']
                        or recipes['task']['horizon'] != task['horizon'] or recipes['persisted_weights'] is not False):
                    raise ValueError('Recipe is not bound to this series and horizon')
                if stage == 'search':
                    search.search(session, args.output, case, task, values, config, recipes['recipes'])
                elif stage == 'assess':
                    past = read_snapshot(plan['development'])
                    tune = assessment.tune_baselines

                    def tune_new_collection(session, output, case, task, history, config, recipes, settings, deadline):
                        settings = {**settings, 'baseline_seasonal_lags': {case['group']: plan['control_lags']}}
                        return tune(session, output, case, task, history, config, recipes, settings, deadline)

                    with patch.object(assessment, 'assess_request', evaluate_control), patch.object(
                        assessment, 'tune_baselines', tune_new_collection,
                    ):
                        assessment.assess(session, args.output, case, task, values, config, recipes['recipes'],
                                          [Path(path) for path in plan['winners']], past)
                else:
                    raise ValueError('Unknown forecast stage')
    write(args.output / 'shutdown.json', {'confirmed': session.shutdown_report.confirmed})


if __name__ == '__main__':
    main()
