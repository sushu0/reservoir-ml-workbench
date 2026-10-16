"""Small SYNTHETIC run of the extracted original model comparison code."""
from __future__ import annotations

import numpy as np
import pandas as pd

from reservoir_workbench.model_compare import run_same_cohort_lowo


def main() -> None:
    rng = np.random.default_rng(42)
    wells = ("SYNTHETICA", "SYNTHETICB", "SYNTHETICC")
    rows = []
    for index, well in enumerate(wells):
        for step in range(12):
            gr = 40.0 + index * 3.0 + step
            rows.append(
                {
                    "well_name": well,
                    "sample_depth": 100.0 + step * 0.125,
                    "porosity_train_eligible": True,
                    "porosity_label_lineage": "measured",
                    "porosity_interpolated": False,
                    "GR": gr,
                    "RTCAL": 5.0 + rng.random(),
                    "DEN": 2.3 + rng.normal(0, 0.01),
                    "porosity": 30.0 - gr * 0.2 + rng.normal(0, 0.1),
                }
            )
    result = run_same_cohort_lowo(pd.DataFrame(rows), "porosity", wells)
    print("SYNTHETIC ONLY: generated values and lineage flags are test fixtures.")
    print("No actual measurements, field accuracy, or production acceptance is claimed.")
    print(f"Synthetic samples: {len(rows)}; held-out groups: {len(wells)}")
    print(result.metrics[["model_family", "lowo_well_count", "train_rows"]].to_string(index=False))
    assert set(result.predictions["well_name"]) == set(wells)
    assert len(result.predictions) == len(rows) * 2
    print("SYNTHETIC workflow check: PASS")


if __name__ == "__main__":
    main()
