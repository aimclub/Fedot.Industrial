"""Typed TSC graph grammar, deterministic mutation and selection."""

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from random import Random

from .data import canonical_bytes, digest


@dataclass(frozen=True, order=True)
class Branch:
    operations: tuple[str, ...]
    parameter: int = 0

    def __post_init__(self):
        if self.operations not in ANCHOR_OPERATIONS + REGRESSION_OPERATIONS:
            raise ValueError('Unsupported transformation or estimator sequence')
        if type(self.parameter) is not int or self.parameter not in range(4):
            raise ValueError('Unsupported node parameter')
        if self.parameter and self.head in NEURAL:
            raise ValueError('Neural recipes are frozen after generation zero')

    @property
    def head(self):
        return self.operations[-1]


NEURAL = frozenset(('cnn2d_classifier', 'tsc_inception', 'tsc_resnet', 'cnn2d_regressor',
                    'tser_inception', 'tser_resnet'))
ANCHOR_OPERATIONS = (
    ('quantile_extractor_torch', 'logreg'),
    ('fourier_basis', 'quantile_extractor_torch', 'logreg'),
    ('wavelet_basis', 'quantile_extractor_torch', 'logreg'),
    ('minirocket_extractor', 'logreg'),
    ('recurrence_rqa', 'logreg'),
    ('recurrence_rqa', 'pilot_rf'),
    ('recurrence_image', 'cnn2d_classifier'),
    ('tsc_inception',),
    ('tsc_resnet',),
    ('quantile_extractor_torch', 'pilot_rf'),
)
REGRESSION_OPERATIONS = (
    ('quantile_extractor_torch', 'pilot_ridge'),
    ('fourier_basis', 'quantile_extractor_torch', 'pilot_ridge'),
    ('wavelet_basis', 'quantile_extractor_torch', 'pilot_ridge'),
    ('minirocket_extractor', 'pilot_ridge'),
    ('recurrence_rqa', 'pilot_ridge'),
    ('recurrence_rqa', 'pilot_regression_forest'),
    ('recurrence_image', 'cnn2d_regressor'),
    ('tser_inception',),
    ('tser_resnet',),
    ('quantile_extractor_torch', 'pilot_regression_forest'),
)


@dataclass(frozen=True)
class Graph:
    branches: tuple[Branch, ...]

    def __post_init__(self):
        if (not 1 <= len(self.branches) <= 3 or self.branches != tuple(sorted(self.branches))
                or len({branch.operations for branch in self.branches}) != len(self.branches)):
            raise ValueError('A graph must have one to three distinct, canonically sorted branches')
        if len({branch.operations in REGRESSION_OPERATIONS for branch in self.branches}) != 1:
            raise ValueError('A graph cannot combine classification and regression heads')

    @classmethod
    def of(cls, branches):
        return cls(tuple(sorted(branches)))

    @property
    def key(self):
        return digest(canonical_bytes(asdict(self)))

    @property
    def requires_cuda(self):
        return any(branch.head in NEURAL for branch in self.branches)

    @property
    def regression(self):
        return self.branches[0].operations in REGRESSION_OPERATIONS


def parse_graph(value):
    return Graph(tuple(Branch(tuple(row['operations']), row['parameter']) for row in value['branches']))


def initial(*, regression=False):
    return tuple(Graph((Branch(ops),)) for ops in (REGRESSION_OPERATIONS if regression else ANCHOR_OPERATIONS))


def proposals(population, seen, count, *, tune, rng: Random, regression=False):
    anchors = tuple(candidate.branches[0] for candidate in initial(regression=regression))
    choices = []
    for parent in population:
        if tune:
            choices.extend(
                Graph.of((*parent.branches[:index], replace(branch, parameter=parameter), *parent.branches[index + 1:]))
                for index, branch in enumerate(parent.branches) if branch.head not in NEURAL
                for parameter in range(4) if parameter != branch.parameter
            )
        elif len(parent.branches) < 3:
            choices.extend(
                Graph.of((*parent.branches, anchor)) for anchor in anchors
                if anchor.operations not in {branch.operations for branch in parent.branches}
            )
    if not tune:
        choices.extend(Graph.of(pair) for pair in combinations(anchors, 2))
        choices.extend(Graph.of(group) for group in combinations(anchors, 3))
    candidates = list({graph.key: graph for graph in choices if graph.key not in seen}.values())
    rng.shuffle(candidates)
    return tuple(candidates[:count])


def rank(values):
    return tuple(sorted(values, key=lambda entry: (entry[1], entry[0].key)))
