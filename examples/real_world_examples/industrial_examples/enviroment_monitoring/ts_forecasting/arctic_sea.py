from __future__ import annotations

from pathlib import Path

from fedot_ind.tools.serialisation.path_lib import EXAMPLES_DATA_PATH


def build_arctic_sea_forecasting_context() -> dict:
    data_path = Path(EXAMPLES_DATA_PATH) / "forecasting" / "ice_forecasting" / "ices_areas_ts.csv"
    return {
        "scenario": "arctic_sea_forecasting",
        "context": {
            "task_type": "forecasting",
            "data_path": str(data_path),
            "recommended_visualization": "benchmark.industrial.evaluation.render_publication_pack",
        },
    }


if __name__ == "__main__":
    print(build_arctic_sea_forecasting_context())
