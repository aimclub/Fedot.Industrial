from __future__ import annotations

from examples.real_world_examples.current_api import eeg_classification_context


def build_neiry_eeg_context() -> dict:
    return {"scenario": "neiry_eeg_classification", "context": eeg_classification_context()}


if __name__ == "__main__":
    print(build_neiry_eeg_context())
