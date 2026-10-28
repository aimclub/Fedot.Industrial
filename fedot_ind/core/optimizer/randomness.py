"""Explicit coordinator-owned randomness for existing GOLEM operators."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256

from golem.core.optimisers.opt_history_objects.individual import Individual

from fedot_ind.core.optimizer.random_scope import operator_random_scope


@dataclass(frozen=True)
class CandidateRandomness:
    """Portable seed address; never includes a UUID, process or worker number."""

    run_seed: int
    stream: str
    attempt: int

    def __post_init__(self):
        for name, value in (("run_seed", self.run_seed), ("attempt", self.attempt)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.stream, str) or not self.stream:
            raise ValueError("stream must be a non-empty string")

    @property
    def seed(self) -> int:
        payload = f"industrial-candidate@1:{self.run_seed}:{len(self.stream)}:{self.stream}:{self.attempt}"
        return int.from_bytes(sha256(payload.encode("utf-8")).digest()[:4], "big")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": "industrial_candidate_randomness@1",
            "run_seed": self.run_seed,
            "stream": self.stream,
            "attempt": self.attempt,
            "seed": self.seed,
        }


class SeededOperator:
    """Keep one explicit seed sequence for one coordinator-side operator."""

    def __init__(self, operator, run_seed: int, stream: str):
        CandidateRandomness(run_seed, stream, 0)
        self.operator = operator
        self.run_seed = run_seed
        self.stream = stream
        self.attempt = 0

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "operator"), name)

    def next_randomness(self) -> CandidateRandomness:
        address = CandidateRandomness(self.run_seed, self.stream, self.attempt)
        self.attempt += 1
        return address

    def __call__(self, *args, **kwargs):
        address = self.next_randomness()
        with operator_random_scope(address.seed):
            return self.operator(*args, **kwargs)


class SeededMutation(SeededOperator):
    """Seed each candidate separately, including unsuccessful mutation attempts."""

    def __call__(self, population):
        individuals = [population] if isinstance(population, Individual) else list(population)
        results = []
        for parent in individuals:
            address = self.next_randomness()
            with operator_random_scope(address.seed):
                candidate = self.operator(parent)
            # GOLEM may return the parent when mutation is skipped, or [] when
            # attempts are exhausted. Do not overwrite a parent's existing seed.
            if isinstance(candidate, Individual):
                if candidate is not parent:
                    candidate = replace(candidate, metadata={
                        **candidate.metadata,
                        "candidate_randomness": address.to_record(),
                    })
                results.append(candidate)
            elif candidate is not None and not (isinstance(candidate, list) and not candidate):
                raise TypeError("Mutation must return an Individual or an empty result")
        if len(individuals) == 1:
            return results[0] if results else []
        return results


def install_seeded_operators(optimizer, run_seed: int | None) -> None:
    """Wire seeded operators into every existing reference without changing search."""
    if run_seed is None:
        return
    previous = (optimizer.selection, optimizer.crossover, optimizer.mutation)
    optimizer.selection = SeededOperator(previous[0], run_seed, "selection")
    optimizer.crossover = SeededOperator(previous[1], run_seed, "crossover")
    optimizer.mutation = SeededMutation(previous[2], run_seed, "mutation")
    replacements = dict(zip(map(id, previous), (optimizer.selection, optimizer.crossover, optimizer.mutation)))
    optimizer.operators = [replacements.get(id(operator), operator) for operator in optimizer.operators]
    optimizer.reproducer.selection = optimizer.selection
    optimizer.reproducer.crossover = optimizer.crossover
    optimizer.reproducer.mutation = optimizer.mutation
    optimizer.inheritance.selection = optimizer.selection
