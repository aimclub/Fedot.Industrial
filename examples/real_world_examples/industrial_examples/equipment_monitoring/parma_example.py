from __future__ import annotations

from pathlib import Path


def build_parma_equipment_monitoring_context(data_dir: str | Path = "dataset") -> dict:
    data_root = Path(data_dir)
    return {
        "scenario": "parma_equipment_monitoring_classification",
        "context": {
            "task_type": "ts_classification",
            "data_root": str(data_root),
            "expected_files": ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"],
        },
    }


if __name__ == "__main__":
    print(build_parma_equipment_monitoring_context())
