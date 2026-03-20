from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from pymonad.either import Either, Left, Right


class AlignmentStatus(str, Enum):
    IMPLEMENTED = "implemented"
    PARTIAL = "partial"
    HEURISTIC = "heuristic"
    MISSING = "missing"


class ImplementationScope(str, Enum):
    EXAMPLE = "example"
    CORE = "core"
    BOTH = "example+core"


class WeaknessCategory(str, Enum):
    DATA_LEAKAGE = "data_leakage"
    API_DESIGN = "api_design"
    NUMERICAL_STABILITY = "numerical_stability"
    SIDE_EFFECTS = "side_effects"
    TESTABILITY = "testability"
    DEPENDENCY_HYGIENE = "dependency_hygiene"
    THEORY_GAP = "theory_gap"


class OKHSMethod(str, Enum):
    DMD = "dmd"
    DIRECT = "direct"
    OCCUPATION = "occupation"


@dataclass(frozen=True)
class TheoryClaim:
    claim_id: str
    theme: str
    document_section: str
    statement: str
    expected_capability: str


@dataclass(frozen=True)
class CodeEvidence:
    component: str
    path: str
    lines: str
    summary: str
    scope: ImplementationScope


@dataclass(frozen=True)
class AlignmentRow:
    claim: TheoryClaim
    status: AlignmentStatus
    summary: str
    evidence: tuple[CodeEvidence, ...]
    missing_capabilities: tuple[str, ...]


@dataclass(frozen=True)
class WeaknessRecord:
    weakness_id: str
    category: WeaknessCategory
    title: str
    summary: str
    severity: str
    evidence: tuple[CodeEvidence, ...]
    impact: str
    recommendation: str


@dataclass(frozen=True)
class RefactorTask:
    task_id: str
    title: str
    target_paths: tuple[str, ...]
    rationale: str
    planned_changes: tuple[str, ...]
    tests: tuple[str, ...]


@dataclass(frozen=True)
class RoadmapStage:
    stage: int
    title: str
    goal: str
    tasks: tuple[RefactorTask, ...]
    success_criteria: tuple[str, ...]


@dataclass(frozen=True)
class DependencyGap:
    package_name: str
    is_declared: bool
    evidence_paths: tuple[str, ...]
    recommendation: str


@dataclass(frozen=True)
class StabilityPolicy:
    threshold: float
    drop_positive_real_modes: bool
    sorting_strategy: str


@dataclass(frozen=True)
class RegularizationPolicy:
    base_jitter: float
    condition_threshold: float
    fallback_solver: str


@dataclass(frozen=True)
class TrajectorySplit:
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


@dataclass(frozen=True)
class ForecastConfig:
    method: OKHSMethod
    q: float
    forecast_horizon: int
    initial_segment_length: int
    stability_policy: StabilityPolicy
    regularization_policy: RegularizationPolicy


@dataclass(frozen=True)
class AnalysisReport:
    title: str
    summary: tuple[str, ...]
    claims: tuple[TheoryClaim, ...]
    alignments: tuple[AlignmentRow, ...]
    weaknesses: tuple[WeaknessRecord, ...]
    dependency_gaps: tuple[DependencyGap, ...]
    roadmap: tuple[RoadmapStage, ...]
    assumptions: tuple[str, ...]


def build_theory_claims() -> tuple[TheoryClaim, ...]:
    return (
        TheoryClaim(
            claim_id="representer-rkhs-okhs",
            theme="Representer theorem / RKHS / OKHS",
            document_section="1.1-1.3",
            statement="Модель должна опираться на Gram-представление траекторий и occupation-like функционалы.",
            expected_capability="Траектории сравниваются через интегральное kernel-представление, а не только через pointwise признаки.",
        ),
        TheoryClaim(
            claim_id="fractional-liouville-dmd",
            theme="Fractional Liouville + Mittag-Leffler + fractional DMD",
            document_section="2.1-2.3",
            statement="Должны существовать конечномерная аппроксимация дробного оператора Лиувилля и прогноз через функции Миттаг-Леффлера.",
            expected_capability="Есть operator fit, generalized eigensolver и прогноз в модальном базисе.",
        ),
        TheoryClaim(
            claim_id="forecaster-strategies",
            theme="OKHSForecaster strategies",
            document_section="3.2",
            statement="Форкастер должен прозрачно поддерживать recursive, multi-output и DMD режимы.",
            expected_capability="Единый, типизированный API стратегий без строковой путаницы.",
        ),
        TheoryClaim(
            claim_id="automl-rkbs-q-selection",
            theme="AutoML / RKBS / uncertainty / q-selection",
            document_section="4.1-4.3",
            statement="Верхний слой должен подбирать q, поддерживать uncertainty и связывать forecasting с более широким AutoML-замыслом.",
            expected_capability="Есть единый orchestration слой, явные q-policy и интерфейсы валидации/benchmarking.",
        ),
    )


def _evidence_index() -> dict[str, tuple[CodeEvidence, ...]]:
    return {
        "representer-rkhs-okhs": (
            CodeEvidence(
                component="OKHSTransformer",
                path="fedot_ind/core/operation/decomposition/matrix_decomposition/method_impl/okhs.py",
                lines="11-215",
                summary="Строит Gram-матрицу по траекториям через двойной интеграл и квадратуры Якоби.",
                scope=ImplementationScope.CORE,
            ),
            CodeEvidence(
                component="okhs_advanced example",
                path="examples/rkhs_okhs/temp_file/okhs_advanced.py",
                lines="140-164",
                summary="Пример реально использует OKHSTransformer как вход в дальнейший fDMD-пайплайн.",
                scope=ImplementationScope.EXAMPLE,
            ),
        ),
        "fractional-liouville-dmd": (
            CodeEvidence(
                component="FractionalLiouvilleOperator",
                path="fedot_ind/core/operation/decomposition/matrix_decomposition/method_impl/okhs.py",
                lines="218-348",
                summary="Строит матрицу Лиувилля и решает обобщённую задачу на собственные значения.",
                scope=ImplementationScope.CORE,
            ),
            CodeEvidence(
                component="FractionalDMD",
                path="fedot_ind/core/operation/decomposition/matrix_decomposition/method_impl/okhs.py",
                lines="351-725",
                summary="Использует Mittag-Leffler эволюцию и восстанавливает прогноз в модальном базисе.",
                scope=ImplementationScope.CORE,
            ),
        ),
        "forecaster-strategies": (
            CodeEvidence(
                component="OKHSForecaster",
                path="fedot_ind/core/models/kernel/okhs_forecasting.py",
                lines="8-87",
                summary="Поддерживает только строки method='dmd' и method='direct'.",
                scope=ImplementationScope.CORE,
            ),
            CodeEvidence(
                component="OKHSForecasterTorch",
                path="fedot_ind/core/models/kernel/okhs_forecasting_torch.py",
                lines="15-133",
                summary="Использует method='dmd' или method='occupation', что не совпадает с numpy-версией.",
                scope=ImplementationScope.CORE,
            ),
        ),
        "automl-rkbs-q-selection": (
            CodeEvidence(
                component="DataDrivenQSelector",
                path="fedot_ind/core/operation/transformation/representation/kernel/kernels.py",
                lines="255-360",
                summary="Есть отдельные эвристики для q, но они не встроены в единый forecasting/API слой.",
                scope=ImplementationScope.CORE,
            ),
            CodeEvidence(
                component="Theory-only scope",
                path="docs/okhs_dmd/постановка_по_okhs_dmd.md",
                lines="89-148",
                summary="AutoML, uncertainty, rolling forecast и benchmarking описаны как целевой контур, но не реализованы как связная подсистема.",
                scope=ImplementationScope.BOTH,
            ),
        ),
    }


def collect_alignment_rows(
        claims: Sequence[TheoryClaim],
) -> tuple[AlignmentRow, ...]:
    evidence_index = _evidence_index()
    rows: list[AlignmentRow] = []

    for claim in claims:
        if claim.claim_id == "representer-rkhs-okhs":
            rows.append(
                AlignmentRow(
                    claim=claim,
                    status=AlignmentStatus.IMPLEMENTED,
                    summary="На уровне core и примера OKHS-часть реализована убедительнее всего: траектории действительно входят через интегральную Gram-конструкцию.",
                    evidence=evidence_index[claim.claim_id],
                    missing_capabilities=(),
                )
            )
        elif claim.claim_id == "fractional-liouville-dmd":
            rows.append(
                AlignmentRow(
                    claim=claim,
                    status=AlignmentStatus.PARTIAL,
                    summary="Ключевая математическая цепочка присутствует, но её инженерное оформление остаётся исследовательским: print-driven flow, broad fallback и визуализация внутри модели.",
                    evidence=evidence_index[claim.claim_id],
                    missing_capabilities=(
                        "typed validation for numerical policies",
                        "separate inference API without plotting side effects",
                    ),
                )
            )
        elif claim.claim_id == "forecaster-strategies":
            rows.append(
                AlignmentRow(
                    claim=claim,
                    status=AlignmentStatus.HEURISTIC,
                    summary="Стратегии есть фрагментарно, но naming и поведение расходятся между numpy и torch реализациями.",
                    evidence=evidence_index[claim.claim_id],
                    missing_capabilities=(
                        "single typed strategy enum",
                        "consistent recursive/multi-output semantics",
                    ),
                )
            )
        else:
            rows.append(
                AlignmentRow(
                    claim=claim,
                    status=AlignmentStatus.MISSING,
                    summary="AutoML-контур, uncertainty и честная q-orchestration заявлены в постановке, но не собраны в рабочий продуктовый слой.",
                    evidence=evidence_index[claim.claim_id],
                    missing_capabilities=(
                        "unified automl entrypoint",
                        "uncertainty estimates",
                        "integrated q-selection policy",
                        "benchmarking/reporting workflow",
                    ),
                )
            )

    return tuple(rows)


def score_alignment(rows: Sequence[AlignmentRow]) -> dict[AlignmentStatus, int]:
    score = {status: 0 for status in AlignmentStatus}
    for row in rows:
        score[row.status] += 1
    return score


def detect_gaps(
        rows: Sequence[AlignmentRow],
        dependency_gaps: Sequence[DependencyGap],
) -> tuple[str, ...]:
    missing_capabilities = tuple(
        capability
        for row in rows
        for capability in row.missing_capabilities
    )
    dependency_issues = tuple(
        f"undeclared dependency: {gap.package_name}"
        for gap in dependency_gaps
        if not gap.is_declared
    )
    return missing_capabilities + dependency_issues


def build_weaknesses() -> tuple[WeaknessRecord, ...]:
    return (
        WeaknessRecord(
            weakness_id="train-test-leakage",
            category=WeaknessCategory.DATA_LEAKAGE,
            title="Train/test leakage in experimental example",
            summary="В примере в качестве test берётся траектория, которая уже участвовала в обучении.",
            severity="high",
            evidence=(
                CodeEvidence(
                    component="okhs_advanced example",
                    path="examples/rkhs_okhs/temp_file/okhs_advanced.py",
                    lines="303-318",
                    summary="`test_traj = trajectories[cfg['n_train_traj'] - 2]` выбирается из обучающего набора.",
                    scope=ImplementationScope.EXAMPLE,
                ),
            ),
            impact="Метрики optimistic и не отражают честное качество прогноза.",
            recommendation="Сделать явный holdout split и не переиспользовать train trajectory как test.",
        ),
        WeaknessRecord(
            weakness_id="implicit-segment-invariant",
            category=WeaknessCategory.API_DESIGN,
            title="Initial segment invariant encoded only in comments",
            summary="Требование `initial_segment_length >= n_train_traj` описано комментариями, но не валидируется типами или API.",
            severity="high",
            evidence=(
                CodeEvidence(
                    component="okhs_advanced example",
                    path="examples/rkhs_okhs/temp_file/okhs_advanced.py",
                    lines="237-276",
                    summary="Комментарий к конфигам предупреждает об инварианте, но код его не проверяет до позднего ValueError из core.",
                    scope=ImplementationScope.EXAMPLE,
                ),
                CodeEvidence(
                    component="FractionalDMD.fit_initial_coefficients",
                    path="fedot_ind/core/operation/decomposition/matrix_decomposition/method_impl/okhs.py",
                    lines="451-519",
                    summary="Валидация происходит поздно и выражена исключением, а не конфигом/typed result.",
                    scope=ImplementationScope.CORE,
                ),
            ),
            impact="Пользователь получает поздние сбои вместо ранней валидации конфигурации.",
            recommendation="Ввести typed config и ранний pure validator для длины начального сегмента.",
        ),
        WeaknessRecord(
            weakness_id="plotting-in-model",
            category=WeaknessCategory.SIDE_EFFECTS,
            title="Inference and plotting are mixed",
            summary="`plot_predict` держит расчёт прогноза и визуализацию в одном методе модели.",
            severity="medium",
            evidence=(
                CodeEvidence(
                    component="FractionalDMD.plot_predict",
                    path="fedot_ind/core/operation/decomposition/matrix_decomposition/method_impl/okhs.py",
                    lines="566-725",
                    summary="Метод одновременно считает прогноз, печатает диагностику и вызывает matplotlib.",
                    scope=ImplementationScope.CORE,
                ),
            ),
            impact="Усложняет тестирование, переиспользование и чистый inference path.",
            recommendation="Оставить в модели только `predict`, а plotting вынести в отдельный effect-shell модуль.",
        ),
        WeaknessRecord(
            weakness_id="numerical-policy-hidden",
            category=WeaknessCategory.NUMERICAL_STABILITY,
            title="Numerical stability policy is implicit",
            summary="Регуляризация Gram-матрицы и fallback solver зашиты непосредственно в методы без явной policy.",
            severity="medium",
            evidence=(
                CodeEvidence(
                    component="OKHSTransformer.fit",
                    path="fedot_ind/core/operation/decomposition/matrix_decomposition/method_impl/okhs.py",
                    lines="165-186",
                    summary="Добавляется безусловный jitter `1e-8 * I` без проверки condition number и без конфигурации.",
                    scope=ImplementationScope.CORE,
                ),
                CodeEvidence(
                    component="FractionalLiouvilleOperator.fit",
                    path="fedot_ind/core/operation/decomposition/matrix_decomposition/method_impl/okhs.py",
                    lines="323-330",
                    summary="Есть broad exception fallback с переходом на pseudo-inverse path.",
                    scope=ImplementationScope.CORE,
                ),
            ),
            impact="Численные решения трудно воспроизводимы и плохо объяснимы на boundary API.",
            recommendation="Вынести регуляризацию, threshold и fallback solver в typed policy objects.",
        ),
        WeaknessRecord(
            weakness_id="missing-okhs-tests",
            category=WeaknessCategory.TESTABILITY,
            title="No dedicated OKHS/fDMD tests",
            summary="В репозитории нет выделенного набора тестов для OKHS/fDMD core.",
            severity="medium",
            evidence=(
                CodeEvidence(
                    component="tests coverage gap",
                    path="tests/",
                    lines="n/a",
                    summary="Поиск по `tests` не показывает таргетированных unit/integration тестов для `okhs.py` и forecasting wrappers.",
                    scope=ImplementationScope.BOTH,
                ),
            ),
            impact="Регрессии в математическом ядре и API остаются незамеченными.",
            recommendation="Добавить pure-core tests на invariants и thin integration tests на честный holdout scenario.",
        ),
        WeaknessRecord(
            weakness_id="missing-dependency-declaration",
            category=WeaknessCategory.DEPENDENCY_HYGIENE,
            title="Required package is imported but not declared",
            summary="`pymittagleffler` импортируется в codebase, но не найден в явных зависимостях.",
            severity="medium",
            evidence=(
                CodeEvidence(
                    component="FractionalDMD dependency",
                    path="fedot_ind/core/operation/decomposition/matrix_decomposition/method_impl/okhs.py",
                    lines="1-7",
                    summary="Core OKHS implementation импортирует `pymittagleffler.mittag_leffler`.",
                    scope=ImplementationScope.CORE,
                ),
                CodeEvidence(
                    component="okhs_advanced example dependency",
                    path="examples/rkhs_okhs/temp_file/okhs_advanced.py",
                    lines="1-10",
                    summary="Пример тоже зависит от `pymittagleffler`.",
                    scope=ImplementationScope.EXAMPLE,
                ),
            ),
            impact="Окружение может собраться неполно, а ошибка проявится только во время импорта.",
            recommendation="Добавить пакет в `requirements.txt` и `pyproject.toml`.",
        ),
    )


def detect_dependency_gaps() -> tuple[DependencyGap, ...]:
    return (
        DependencyGap(
            package_name="pymittagleffler",
            is_declared=False,
            evidence_paths=(
                "fedot_ind/core/operation/decomposition/matrix_decomposition/method_impl/okhs.py",
                "examples/rkhs_okhs/temp_file/okhs_advanced.py",
            ),
            recommendation="Declare the dependency explicitly in packaging metadata.",
        ),
    )


def build_roadmap() -> tuple[RoadmapStage, ...]:
    return (
        RoadmapStage(
            stage=1,
            title="Stabilize the experimental boundary",
            goal="Make the example honest, reproducible and easy to inspect.",
            tasks=(
                RefactorTask(
                    task_id="split-example-runner",
                    title="Turn okhs_advanced into a thin experiment runner",
                    target_paths=("examples/rkhs_okhs/temp_file/okhs_advanced.py",),
                    rationale="The example currently mixes data generation, split selection, training, plotting and metric calculation.",
                    planned_changes=(
                        "extract data generation into pure helpers",
                        "introduce explicit holdout split",
                        "move plotting to dedicated functions",
                        "validate config before model fitting",
                    ),
                    tests=(
                        "honest holdout split has no train/test overlap",
                        "config validation fails early for insufficient initial segment",
                    ),
                ),
            ),
            success_criteria=(
                "No train/test leakage in example pipeline.",
                "Metrics are computed on holdout trajectories only.",
            ),
        ),
        RoadmapStage(
            stage=2,
            title="Harden the numerical core",
            goal="Separate pure numerical logic from effectful presentation and expose explicit policies.",
            tasks=(
                RefactorTask(
                    task_id="core-policy-split",
                    title="Introduce stability and regularization policies",
                    target_paths=(
                        "fedot_ind/core/operation/decomposition/matrix_decomposition/method_impl/okhs.py",
                    ),
                    rationale="Numerical choices are currently hidden inside methods and enforced via implicit constants and fallbacks.",
                    planned_changes=(
                        "replace hidden jitter with RegularizationPolicy",
                        "replace implicit stability filtering with StabilityPolicy",
                        "move plot_predict logic into external visualization helper",
                        "replace print-driven flow with explicit result objects",
                    ),
                    tests=(
                        "Gram matrix symmetry and deterministic construction",
                        "Liouville/DMD shape invariants",
                        "predict path free of plotting side effects",
                    ),
                ),
            ),
            success_criteria=(
                "Predict and plot are separate APIs.",
                "Numerical fallbacks are explicit in config, not hidden in implementation.",
            ),
        ),
        RoadmapStage(
            stage=3,
            title="Unify forecasting APIs",
            goal="Align numpy and torch wrappers around one typed strategy model.",
            tasks=(
                RefactorTask(
                    task_id="unify-forecaster-methods",
                    title="Normalize method naming and q-selection hooks",
                    target_paths=(
                        "fedot_ind/core/models/kernel/okhs_forecasting.py",
                        "fedot_ind/core/models/kernel/okhs_forecasting_torch.py",
                        "fedot_ind/core/operation/transformation/representation/kernel/kernels.py",
                    ),
                    rationale="Current wrappers use inconsistent string modes and leave q-selection detached from forecasting orchestration.",
                    planned_changes=(
                        "introduce OKHSMethod enum",
                        "normalize direct/occupation semantics",
                        "wire q-selection as explicit policy input",
                        "preserve backwards-compatible aliases at entrypoints",
                    ),
                    tests=(
                        "legacy string aliases remain accepted",
                        "new enum-based config resolves to consistent runtime behavior",
                    ),
                ),
            ),
            success_criteria=(
                "Torch and numpy wrappers share one strategy vocabulary.",
                "q-selection is invoked as policy, not as hidden heuristic.",
            ),
        ),
    )


def normalize_okhs_method(value: str | OKHSMethod) -> Either:
    if isinstance(value, OKHSMethod):
        return Right(value)

    normalized = value.strip().lower()
    aliases = {
        "dmd": OKHSMethod.DMD,
        "direct": OKHSMethod.DIRECT,
        "occupation": OKHSMethod.OCCUPATION,
    }
    return Right(aliases[normalized]) if normalized in aliases else Left(
        f"Unsupported OKHS method: {value}"
    )


def validate_initial_segment_length(
        initial_segment_length: int,
        n_modes: int,
        n_features: int,
) -> Either:
    available_equations = initial_segment_length * n_features
    return (
        Right(initial_segment_length)
        if available_equations >= n_modes
        else Left(
            "Insufficient initial segment length: "
            f"{initial_segment_length} * {n_features} < {n_modes}"
        )
    )


def choose_holdout_split(
        total_trajectories: int,
        holdout_size: int = 1,
) -> Either:
    if total_trajectories <= holdout_size:
        return Left(
            f"Need more trajectories than holdout_size: {total_trajectories} <= {holdout_size}"
        )

    train_indices = tuple(range(0, total_trajectories - holdout_size))
    test_indices = tuple(range(total_trajectories - holdout_size, total_trajectories))
    return Right(TrajectorySplit(train_indices=train_indices, test_indices=test_indices))


def choose_q_policy(
        autocorrelation_slope: float,
        dominant_frequency: float,
) -> float:
    if autocorrelation_slope > -0.3 or dominant_frequency < 0.1:
        return 0.9
    if autocorrelation_slope > -0.7 or dominant_frequency < 0.3:
        return 0.7
    if autocorrelation_slope > -1.2:
        return 0.5
    return 0.3


def split_predict_and_plot_api(
        predict_name: str = "predict",
        plot_name: str = "plot_forecast_diagnostics",
) -> dict[str, tuple[str, ...]]:
    return {
        "predict_api": (predict_name, "returns forecast only"),
        "plot_api": (plot_name, "accepts forecast artefacts and renders charts"),
    }


def validate_square_matrix_shape(rows: int, cols: int, label: str) -> Either:
    return Right((rows, cols)) if rows == cols else Left(f"{label} must be square: got {rows}x{cols}")


def validate_liouville_shapes(
        gram_shape: tuple[int, int],
        liouville_shape: tuple[int, int],
        eigen_count: int,
) -> Either:
    gram_ok = validate_square_matrix_shape(gram_shape[0], gram_shape[1], "Gram matrix")
    if gram_ok.is_left():
        return gram_ok

    liouville_ok = validate_square_matrix_shape(liouville_shape[0], liouville_shape[1], "Liouville matrix")
    if liouville_ok.is_left():
        return liouville_ok

    if gram_shape != liouville_shape:
        return Left(
            f"Gram/Liouville mismatch: gram={gram_shape}, liouville={liouville_shape}"
        )
    if eigen_count > gram_shape[0]:
        return Left(
            f"Eigen count {eigen_count} cannot exceed matrix size {gram_shape[0]}"
        )
    return Right((gram_shape, liouville_shape, eigen_count))


def render_refactor_recommendations(stages: Sequence[RoadmapStage]) -> str:
    rendered_stages = []
    for stage in stages:
        rendered_tasks = []
        for task in stage.tasks:
            rendered_tasks.append(
                "\n".join(
                    (
                        f"- `{task.task_id}`: {task.title}",
                        f"  Paths: {', '.join(task.target_paths)}",
                        f"  Why: {task.rationale}",
                        f"  Changes: {', '.join(task.planned_changes)}",
                        f"  Tests: {', '.join(task.tests)}",
                    )
                )
            )
        rendered_stages.append(
            "\n".join(
                (
                    f"### Stage {stage.stage}. {stage.title}",
                    stage.goal,
                    *rendered_tasks,
                    "Success criteria: " + "; ".join(stage.success_criteria),
                )
            )
        )
    return "\n\n".join(rendered_stages)


def render_markdown_report(report: AnalysisReport) -> str:
    score = score_alignment(report.alignments)

    alignment_lines = [
        "| Theme | Status | What matches | Missing | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report.alignments:
        evidence = "<br>".join(
            f"`{item.path}:{item.lines}` - {item.summary}"
            for item in row.evidence
        )
        missing = "<br>".join(row.missing_capabilities) if row.missing_capabilities else "-"
        alignment_lines.append(
            f"| {row.claim.theme} | {row.status.value} | {row.summary} | {missing} | {evidence} |"
        )

    weakness_lines = []
    for weakness in report.weaknesses:
        evidence = "; ".join(f"`{item.path}:{item.lines}`" for item in weakness.evidence)
        weakness_lines.append(
            "\n".join(
                (
                    f"### {weakness.title}",
                    f"- Category: `{weakness.category.value}`",
                    f"- Severity: `{weakness.severity}`",
                    f"- Summary: {weakness.summary}",
                    f"- Evidence: {evidence}",
                    f"- Impact: {weakness.impact}",
                    f"- Recommendation: {weakness.recommendation}",
                )
            )
        )

    dependency_lines = [
        f"- `{gap.package_name}` declared={gap.is_declared}: {gap.recommendation}"
        for gap in report.dependency_gaps
    ]

    assumptions = "\n".join(f"- {item}" for item in report.assumptions)
    summary = "\n".join(f"- {item}" for item in report.summary)

    return "\n".join(
        (
            f"# {report.title}",
            "",
            "## Executive Summary",
            summary,
            "",
            "## Alignment Scorecard",
            f"- implemented: {score[AlignmentStatus.IMPLEMENTED]}",
            f"- partial: {score[AlignmentStatus.PARTIAL]}",
            f"- heuristic: {score[AlignmentStatus.HEURISTIC]}",
            f"- missing: {score[AlignmentStatus.MISSING]}",
            "",
            "## Alignment Matrix",
            *alignment_lines,
            "",
            "## Weaknesses",
            "\n\n".join(weakness_lines),
            "",
            "## Dependency Gaps",
            *dependency_lines,
            "",
            "## Development Roadmap",
            render_refactor_recommendations(report.roadmap),
            "",
            "## Why This Is Better Than A Naive Scientific Script",
            "- The domain model is typed: strategy, stability and regularization become explicit data rather than comments and string flags.",
            "- The pure core can be tested without matplotlib, pycaputo runtime or filesystem access.",
            "- Effect boundaries are narrow: reading sources and writing artefacts are isolated from analysis and planning logic.",
            "- The design makes future refactoring incremental: example runner, numerical core and wrapper APIs can evolve independently.",
            "",
            "## Assumptions",
            assumptions,
            "",
        )
    )


def build_analysis_report() -> AnalysisReport:
    claims = build_theory_claims()
    alignments = collect_alignment_rows(claims)
    weaknesses = build_weaknesses()
    dependency_gaps = detect_dependency_gaps()
    return AnalysisReport(
        title="OKHS-DMD Alignment Analysis And Reference Refactoring Design",
        summary=(
            "The strongest match between theory and implementation is the OKHS/fractional Liouville mathematical core in `okhs.py`.",
            "The main engineering gaps are data leakage in the example, hidden numerical policies, plotting inside the model and inconsistent wrapper APIs.",
            "AutoML, uncertainty estimation, benchmarking and integrated q-selection remain mostly theory-level intentions rather than delivered product behavior.",
        ),
        claims=claims,
        alignments=alignments,
        weaknesses=weaknesses,
        dependency_gaps=dependency_gaps,
        roadmap=build_roadmap(),
        assumptions=(
            "Analysis is derived only from local repository artefacts.",
            "`okhs_advanced.py` is treated as an experiment consumer, not the system-of-record implementation.",
            "The short-term goal is alignment and testability, not a full scientific rewrite.",
        ),
    )


def read_text(path: Path) -> Either:
    try:
        return Right(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return Left(f"Failed to read {path}: {exc}")


def write_text(path: Path, content: str) -> Either:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return Right(path)
    except OSError as exc:
        return Left(f"Failed to write {path}: {exc}")


def write_analysis_bundle(output_dir: Path) -> Either:
    report = build_analysis_report()
    report_path = output_dir / "okhs_alignment_analysis.md"
    write_result = write_text(report_path, render_markdown_report(report))
    if write_result.is_left():
        return write_result
    return Right((report_path, output_dir / "okhs_development_reference.py"))


DEFAULT_STABILITY_POLICY = StabilityPolicy(
    threshold=0.0,
    drop_positive_real_modes=True,
    sorting_strategy="abs_eigenvalue_desc",
)

DEFAULT_REGULARIZATION_POLICY = RegularizationPolicy(
    base_jitter=1e-8,
    condition_threshold=1e10,
    fallback_solver="pinv",
)

DEFAULT_FORECAST_CONFIG = ForecastConfig(
    method=OKHSMethod.DMD,
    q=0.7,
    forecast_horizon=10,
    initial_segment_length=16,
    stability_policy=DEFAULT_STABILITY_POLICY,
    regularization_policy=DEFAULT_REGULARIZATION_POLICY,
)

if __name__ == "__main__":
    report = build_analysis_report()
    print(render_markdown_report(report))
