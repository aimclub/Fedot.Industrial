import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from unittest.mock import patch
from dataclasses import dataclass
from typing import Optional, Any

from pyriemann.estimation import Covariances, BlockCovariances, Shrinkage, CoSpectra
from pyriemann.tangentspace import TangentSpace


from fedot_ind.core.kernel_learning import (
    BudgetedRepositoryFeatureGeneratorAdapter,
    GeneratorBudgetPolicy,
    OperationSpec,
    RepositoryFeatureGeneratorAdapter,
    ShapeletFeatureGenerator,
    SummaryFeatureGenerator,
    build_generator_registry,
    create_feature_generator,
    resolve_torch_device,
)
from fedot_ind.core.kernel_learning.generators import adapters
from fedot_ind.core.operation.transformation.representation.manifold.riemann_embeding import RiemannExtractor

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@dataclass
class MockInputData:
    features: np.ndarray
    target: Optional[np.ndarray] = None


def generate_spd_matrices(n_samples: int, channels: int) -> np.ndarray:
    A = np.random.randn(n_samples, channels, channels)
    return A @ A.transpose(0, 2, 1) + np.eye(channels) * 1e-4

@pytest.fixture
def spd_data():
    X = generate_spd_matrices(10, 3)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return InputData(
        idx=np.arange(10),
        features=X,
        target=y,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table
    )


def _raw_view(name: str = "raw") -> dict:
    """Return a raw-covariance view used by extractor tests."""
    return {
        "name": name,
        "builder": "covariance",
        "weight": 1.0,
        "shrinkage": 0.1,
        "params": {"estimator": "scm", "estimator_params": {}},
    }


def _grouped_view(group_sizes: list[int]) -> dict:
    """Return a grouped-covariance view used by extractor tests."""
    return {
        "name": "grouped",
        "builder": "grouped_covariance",
        "weight": 1.0,
        "shrinkage": 0.1,
        "params": {
            "group_sizes": group_sizes,
            "estimator": "scm",
            "estimator_params": {},
        },
    }


def test_riemann_extractor_handles_short_series_without_nan_or_inf():
    """Default class MDM must remain finite for short labelled series."""
    generator = create_feature_generator("riemann_extractor")
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ]
    )
    y = np.array([0, 1, 1])

    features = generator.fit_transform(X, y).features

    assert features.shape[0] == X.shape[0]
    assert np.all(np.isfinite(features))


def test_riemann_extractor_sanitizes_nan_and_inf_inputs():
    """The registry path must sanitise non-finite labelled input values."""
    generator = create_feature_generator("riemann_extractor")
    X = np.array(
        [
            [0.0, np.nan, 1.0],
            [np.inf, -1.0, 0.0],
            [1.0, 0.0, 0.5],
        ]
    )
    y = np.array([0, 1, 1])

    features = generator.fit_transform(X, y).features

    assert features.shape[0] == X.shape[0]
    assert np.all(np.isfinite(features))


def test_fitted_riemann_transform_does_not_require_target():
    """Only class-centroid fitting requires target; fitted inference must not."""
    pytest.importorskip("fedot")
    pytest.importorskip("torch")

    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    y = np.array([0, 1])

    generator = create_feature_generator("riemann_extractor")

    train = generator.fit_transform(X, y).features
    inference = generator.transform(X).features

    assert np.all(np.isfinite(train))
    assert np.all(np.isfinite(inference))
    assert train.shape == inference.shape


def test_riemann_extractor_uses_task_specific_effective_feature_mode():
    """Default both-mode keeps MDM for classification but not regression."""
    pytest.importorskip("fedot")
    pytest.importorskip("torch")

    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    y = np.array([0, 1])

    gen_clf = create_feature_generator("riemann_extractor")
    out_clf = gen_clf.fit_transform(X, y, task_type="classification").features

    gen_reg = create_feature_generator("riemann_extractor")
    out_reg = gen_reg.fit_transform(X, y, task_type="regression").features

    gen_ts = create_feature_generator("riemann_extractor")
    out_ts = gen_ts.fit_transform(X, y, task_type="ts_forecasting").features

    assert out_clf.shape == out_ts.shape
    assert out_clf.shape[1] == out_reg.shape[1] + 2
    assert np.all(np.isfinite(out_clf))
    assert np.all(np.isfinite(out_reg))
    assert np.all(np.isfinite(out_ts))


@pytest.mark.parametrize("invalid_params, expected_error_match", [
    (
        {"views": [_raw_view()], "feature_mode": "magic_method"},
        "feature_mode must be"
    ),
    (
        {"views": [_raw_view()], "mdm_metric": "manhattan"},
        "Unsupported mdm_metric: 'manhattan'"
    ),
    (
        {"views": [_raw_view()], "tangent_metric": "cosine"},
        "Unsupported tangent_metric: 'cosine'"
    ),
    (
        {"extraction_strategy": "tangent"},
        "Legacy RiemannExtractor parameters are not supported"
    ),
])
def test_riemann_extractor_incorrect_params_raise_value_error(invalid_params, expected_error_match):

    with pytest.raises(ValueError, match=expected_error_match):
        RiemannExtractor(invalid_params)


def test_riemann_extractor_standalone_mdm_strategy_fit_without_target_raises_error(spd_data):
    extractor = RiemannExtractor({
        "views": [_raw_view()],
        "feature_mode": "mdm",
        "mdm_centroid_scope": "class",
    })

    data_without_target = MockInputData(features=spd_data.features, target=None)

    with pytest.raises(ValueError, match="Target data is required to fit class MDM centroids"):
        extractor.fit(data_without_target)


def test_riemann_extractor_mdm_centroid_scope_behavioral_difference(spd_data):
    ext_class_wise = RiemannExtractor({
        "views": [_raw_view()],
        "feature_mode": "mdm",
        "mdm_centroid_scope": "class",
    }).fit(spd_data)

    ext_global = RiemannExtractor({
        "views": [_raw_view()],
        "feature_mode": "mdm",
        "mdm_centroid_scope": "global",
    }).fit(spd_data)

    assert len(ext_class_wise.class_product_centroids_) == 2
    assert len(ext_global.class_product_centroids_) == 1

    out_class_wise = ext_class_wise._transform(spd_data)
    out_global = ext_global._transform(spd_data)

    assert out_class_wise.shape[1] == 2
    assert out_global.shape[1] == 1
    assert not np.allclose(out_class_wise[:, 0], out_global[:, 0])


@pytest.mark.parametrize("feature_mode, expected_dim", [
    ('mdm', 2),
    ('tangent', 6),
    ('both', 8)
])
def test_riemann_extractor_output_shape_matches_strategy(feature_mode, expected_dim, spd_data):
    extractor = RiemannExtractor({
        "views": [_raw_view()],
        "feature_mode": feature_mode,
        "mdm_centroid_scope": "class",
    })


    features = extractor.fit(spd_data)._transform(spd_data)

    assert features.shape == (10, expected_dim)


def test_riemann_extractor_robust_centroid_dispatching(spd_data):
    """Use the configured robust Torch centroid for class-wise MDM."""
    extractor = RiemannExtractor({
        "views": [_raw_view()],
        "feature_mode": "mdm",
        "mdm_centroid_scope": "global",
        'centroid_type': 'median',
        "mdm_metric": "riemann",
    })
    extractor.fit(spd_data)
    assert extractor.class_product_centroids_ is not None
    assert extractor.mdm_distances_["raw"].metric == 'riemann'


def test_riemann_extractor_logeuclid_median_is_supported():
    """Allow the native Torch Log-Euclidean geometric median."""
    extractor = RiemannExtractor({
        "views": [_raw_view()],
        "centroid_type": "median",
        "mdm_metric": "logeuclid",
    })
    assert extractor.centroid_type == 'median'


def test_riemann_extractor_tangent_with_median_is_supported():
    """Use a robust centroid for tangent projection instead of ignoring it."""
    extractor = RiemannExtractor({
        "views": [_raw_view()],
        "feature_mode": "tangent",
        "centroid_type": "median",
    })
    assert extractor.centroid_type == 'median'


def test_riemann_extractor_transform_before_fit_raises_warning(spd_data):
    extractor = RiemannExtractor({"views": [_raw_view()], "feature_mode": "mdm"})

    with pytest.warns(UserWarning, match="RiemannExtractor is not fitted"):
        extractor._transform(spd_data)


def test_riemann_extractor_distance_uses_mdm_metric(spd_data):
    """Configure the Torch MDM distance transformer with ``mdm_metric``."""
    extractor = RiemannExtractor({
        "views": [_raw_view()],
        "feature_mode": "mdm",
        "mdm_centroid_scope": "global",
        "mdm_metric": "logeuclid",
        "tangent_metric": "riemann",
    })

    extractor.fit(spd_data)
    extractor._transform(spd_data)
    assert extractor.mdm_distances_["raw"].metric == 'logeuclid'

def test_riemann_extractor_returns_warning_for_one_dimensional_input():

    extractor = RiemannExtractor({
        "views": [_raw_view()],
        "feature_mode": "mdm",
        "mdm_centroid_scope": "global",
    })
    X = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0]
    ])

    with pytest.warns(UserWarning):
        extractor._prepare_tensor(X)

def test_riemann_extractor_uses_default_registry_path_for_small_input():

    pytest.importorskip("fedot")
    pytest.importorskip("torch")

    generator = create_feature_generator("riemann_extractor")

    X = np.random.randn(2, 3, 4)
    y = np.array([0, 1])

    bundle = generator.fit_transform(X, y)

    assert bundle.diagnostics.get("skipped") is not True, \
        f"Extractor failed and used fallback. Reason: {bundle.diagnostics.get('budget', {}).get('skip_reason')}"

    assert bundle.diagnostics.get("source") == "fedot_industrial_operation", \
        f"Expected 'fedot_industrial_operation', got '{bundle.diagnostics.get('source')}'"

    assert np.all(np.isfinite(bundle.features))


@pytest.fixture
def ts_multichannel_data():
    """Генерирует временные ряды: 10 сэмплов, 4 канала, 350 временных отсчетов."""
    np.random.seed(42)
    X = np.random.randn(10, 4, 350)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return InputData(
        idx=np.arange(10),
        features=X,
        target=y,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table
    )


def test_riemann_extractor_invalid_representation_type():
    with pytest.raises(ValueError, match="Unknown SPD builder: 'magic'"):
        RiemannExtractor({
            "views": [{**_raw_view(), "builder": "magic"}],
        })


def test_riemann_extractor_block_missing_sizes():
    view = _raw_view()
    view.update({"builder": "grouped_covariance", "params": {}})
    with pytest.raises(ValueError, match="requires 'group_sizes'"):
        RiemannExtractor({"views": [view]})

def test_riemann_extractor_block_matrix_structure(ts_multichannel_data):
    extractor = RiemannExtractor({
        "views": [_grouped_view([2, 2])],
        "feature_mode": "tangent",
    })

    # Извлекаем тензор в том виде, в котором он пойдет в Torch SPD builder.
    X = extractor._prepare_tensor(ts_multichannel_data.features)

    # Инициализируем пайплайн, чтобы обучить builder.
    extractor.fit(ts_multichannel_data)

    # Получаем промежуточные матрицы ковариации.
    # Для 4 каналов это должны быть матрицы размерности 4x4
    SPD = extractor.view_builders_["grouped"].transform_spd(X).to_dense().cpu().numpy()

    assert SPD.shape == (10, 4, 4)

    # Проверяем структуру:
    # При block_sizes = [2, 2] матрица разбита на 4 квадранта 2x2.
    # Внедиагональные квадранты должны состоять из нулей.
    off_diagonal_block_1 = SPD[:, :2, 2:] # Верхний правый угол
    off_diagonal_block_2 = SPD[:, 2:, :2] # Нижний левый угол

    np.testing.assert_allclose(off_diagonal_block_1, 0.0, atol=1e-7)
    np.testing.assert_allclose(off_diagonal_block_2, 0.0, atol=1e-7)

    # Диагональные квадранты должны содержать значимые (ненулевые) значения
    diagonal_block_1 = SPD[:, :2, :2]
    diagonal_block_2 = SPD[:, 2:, 2:]

    assert np.any(np.abs(diagonal_block_1) > 1e-7)
    assert np.any(np.abs(diagonal_block_2) > 1e-7)

def test_riemann_extractor_exact_numerical_values(ts_multichannel_data):
    # 1. Запускаем тестируемый пайплайн для блочного представления
    extractor = RiemannExtractor({
        "views": [_grouped_view([2, 2])],
        "feature_mode": "tangent",
    })

    # Извлекаем тензор X в формате [n_samples, channels, time]
    X = extractor._prepare_tensor(ts_multichannel_data.features)
    n_samples, C, T = X.shape

    # Пропускаем через pyriemann, чтобы получить эмпирические результаты
    extractor.fit(ts_multichannel_data)
    builder = extractor.view_builders_["grouped"]
    shrinkage = extractor.view_shrinkages_["grouped"]
    SPD_block = builder.transform_spd(X).to_dense().cpu().numpy()
    structured_spd = builder.transform_spd(X)
    SPD_shrunk = shrinkage.transform(structured_spd).to_dense().cpu().numpy()

    # 2. Аналитическое вычисление ожидаемых матриц
    cov_estimator = Covariances(estimator="scm")

    # Вычисляем ковариации для первого (каналы 0,1) и второго (каналы 2,3) блоков абсолютно независимо
    signals = X.cpu().numpy()
    spd_1 = cov_estimator.fit_transform(signals[:, :2, :])
    spd_2 = cov_estimator.fit_transform(signals[:, 2:, :])

    expected_spd = np.zeros((n_samples, 4, 4))
    expected_shrunk = np.zeros((n_samples, 4, 4))
    alpha = shrinkage.shrinkage

    for i in range(n_samples):
        # Собираем прямую сумму матриц (размещаем независимые ковариации на диагонали)
        expected_spd[i, :2, :2] = spd_1[i]
        expected_spd[i, 2:, 2:] = spd_2[i]

        # Аналитически применяем shrinkage
        mu = np.trace(expected_spd[i]) / C
        expected_shrunk[i] = (1 - alpha) * expected_spd[i] + alpha * mu * np.eye(C)

    # 3. Строгая численная сверка до машинной точности
    np.testing.assert_allclose(SPD_block, expected_spd, atol=1e-12,
                               err_msg="Базовая блочная матрица не совпадает с аналитической суммой блоков")

    np.testing.assert_allclose(SPD_shrunk, expected_shrunk, atol=1e-12,
                               err_msg="Матрица после Shrinkage не совпадает с аналитическим расчетом")

def test_riemann_extractor_block_analytical_features(ts_multichannel_data):
    """
    Тест проверяет аналитическое извлечение признаков для block-представления.
    Использует фикстуру ts_multichannel_data (4 канала).
    """
    extractor = RiemannExtractor({
        "views": [_grouped_view([2, 2])],
        "feature_mode": "tangent",
    })

    # 1. Прогоняем данные через пайплайн
    features = extractor.fit(ts_multichannel_data)._transform(ts_multichannel_data)

    # 2. Аналитическое вычисление ожидаемого результата
    X = ts_multichannel_data.features

    spd_block = BlockCovariances(block_size=[2, 2], estimator='scm').fit_transform(X)
    spd_shrunk = Shrinkage().fit_transform(spd_block)
    full_tangent_features = TangentSpace(metric='riemann').fit_transform(spd_shrunk)

    # 3. Оставляем только координаты диагональных блоков матрицы 4x4.
    analytical_indices = np.array([0, 1, 4, 7, 8, 9])
    expected_features = full_tangent_features[:, analytical_indices] / np.sqrt(2.0)

    # Выравниваем размерности (на случай если декоратор сделал массив трехмерным)
    features = features.reshape(expected_features.shape)

    # 4. Строгая поэлементная сверка
    np.testing.assert_allclose(
        features,
        expected_features,
        rtol=1e-6,
        atol=1e-9,
        err_msg="Извлеченные признаки не совпадают с аналитически отфильтрованным касательным вектором"
    )

def test_riemann_extractor_cospectra_analytical_features(ts_multichannel_data):
    """
    Тест проверяет конструирование блочно-диагональной матрицы из ко-спектра
    и аналитическое отсечение структурных нулей.
    """
    fmin, fmax, fs = 1.0, 3.0, 100.0

    extractor = RiemannExtractor({
        "views": [{
            "name": "spectral",
            "builder": "cospectra",
            "weight": 1.0,
            "shrinkage": 0.1,
            "params": {"fmin": fmin, "fmax": fmax, "fs": fs},
        }],
        "feature_mode": "tangent",
    })

    # 1. Прогоняем данные через пайплайн
    features = extractor.fit(ts_multichannel_data)._transform(ts_multichannel_data)

    # 2. Математическое дублирование пайплайна для сверки
    X = ts_multichannel_data.features

    co_spectra = CoSpectra(fmin=fmin, fmax=fmax, fs=fs).fit_transform(X)
    N, C, _, F = co_spectra.shape
    D = C * F

    spd_block = np.zeros((N, D, D))
    for f in range(F):
        spd_block[:, f*C:(f+1)*C, f*C:(f+1)*C] = co_spectra[:, :, :, f]

    spd_shrunk = Shrinkage().fit_transform(spd_block)
    full_tangent_features = TangentSpace(metric='riemann').fit_transform(spd_shrunk)

    # 3. Аналитический расчет индексов
    expected_indices = []
    offset = 0
    for _ in range(F):
        for i in range(offset, offset + C):
            for j in range(i, offset + C):
                idx = i * D - (i * (i + 1)) // 2 + j
                expected_indices.append(idx)
        offset += C

    expected_features = full_tangent_features[:, expected_indices] / np.sqrt(F)
    features = features.reshape(expected_features.shape)

    # 4. Строгая сверка индексов и самих признаков
    np.testing.assert_allclose(
        features,
        expected_features,
        rtol=1e-6,
        atol=1e-9,
        err_msg="Касательные векторы для cospectra рассчитаны неверно по отношению к эталону"
    )
