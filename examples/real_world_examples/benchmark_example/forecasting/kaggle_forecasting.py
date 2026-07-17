from __future__ import annotations

from examples.real_world_examples.current_api import kaggle_forecasting_context


def build_kaggle_forecasting_context(data_dir="examples/utils/data/forecasting/kaggle_inventory") -> dict:
    return {"scenario": "kaggle_multi_warehouse_forecasting", "context": kaggle_forecasting_context(data_dir)}


if __name__ == "__main__":
    print(build_kaggle_forecasting_context())
