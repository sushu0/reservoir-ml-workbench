from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from reservoir_workbench.resampling import (
    ResamplingConfig,
    resample_frozen_segment,
)


class PersonalReservoirResamplingTests(unittest.TestCase):
    def test_resampling_records_actual_log_depth_and_does_not_cross_large_core_gap(self) -> None:
        log = pd.DataFrame(
            {
                "DEPT": np.arange(100.0, 102.0, 0.125),
                "GR": np.arange(16.0),
                "GR_NORM": np.linspace(0, 1, 16),
            }
        )
        layers = pd.DataFrame(
            {
                "well_name": ["SYNTHETICA", "SYNTHETICA"],
                "layer": ["L1", "L2"],
                "top_md": [100.0, 101.0],
                "bottom_md": [101.0, 102.0],
            }
        )
        anchors = pd.DataFrame(
            {
                "well_name": ["SYNTHETICA"] * 4,
                "depth_md": [100.0, 100.5, 101.0, 101.75],
                "depth_md_raw": [99.875, 100.375, 100.875, 101.625],
                "porosity": [10.0, 20.0, 30.0, 60.0],
                "porosity_label_lineage": ["measured"] * 4,
            }
        )
        manifest = pd.Series(
            {
                "well_name": "SYNTHETICA",
                "segment_label": "全井",
                "layer_names": "L1;L2",
                "status": "FROZEN",
            }
        )
        out = resample_frozen_segment(
            log, anchors, layers, manifest, ResamplingConfig()
        )
        at_100_25 = out.loc[np.isclose(out["sample_depth"], 100.25)].iloc[0]
        at_101_375 = out.loc[np.isclose(out["sample_depth"], 101.375)].iloc[0]
        self.assertAlmostEqual(float(at_100_25["porosity"]), 15.0)
        self.assertTrue(bool(at_100_25["porosity_interpolated"]))
        self.assertEqual(at_100_25["porosity_label_lineage"], "measured")
        self.assertAlmostEqual(float(at_100_25["log_match_depth"]), 100.25)
        self.assertAlmostEqual(float(at_100_25["depth_delta"]), 0.0)
        self.assertTrue(pd.isna(at_101_375["porosity"]))
        self.assertEqual(at_101_375["porosity_label_lineage"], "unknown")
        self.assertEqual(at_101_375["layer"], "L2")
        self.assertFalse(bool(np.isclose(out["sample_depth"], 102.0).any()))

    def test_formula_endpoint_makes_interpolated_target_formula_lineage(self) -> None:
        log = pd.DataFrame({"DEPT": [1.0, 1.125, 1.25], "GR": [10.0, 11.0, 12.0]})
        anchors = pd.DataFrame(
            {
                "well_name": ["SYNTHETICA", "SYNTHETICA"],
                "depth_md": [1.0, 1.25],
                "porosity": [10.0, 20.0],
                "porosity_label_lineage": ["measured", "formula"],
            }
        )
        layers = pd.DataFrame(
            {
                "well_name": ["SYNTHETICA"],
                "layer": ["L1"],
                "top_md": [1.0],
                "bottom_md": [1.375],
            }
        )
        out = resample_frozen_segment(
            log,
            anchors,
            layers,
            pd.Series(
                {
                    "well_name": "SYNTHETICA",
                    "segment_label": "全井",
                    "layer_names": "L1",
                    "status": "FROZEN",
                }
            ),
        )
        middle = out.loc[np.isclose(out["sample_depth"], 1.125)].iloc[0]
        self.assertEqual(middle["porosity_label_lineage"], "formula")

    def test_non_frozen_segment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "segment is not FROZEN"):
            resample_frozen_segment(
                pd.DataFrame({"DEPT": [1.0], "GR": [10.0]}),
                pd.DataFrame({"well_name": ["SYNTHETICB"], "depth_md": [1.0]}),
                pd.DataFrame(
                    {
                        "well_name": ["SYNTHETICB"],
                        "layer": ["L1"],
                        "top_md": [1.0],
                        "bottom_md": [2.0],
                    }
                ),
                pd.Series(
                    {
                        "well_name": "SYNTHETICB",
                        "segment_label": "下部",
                        "status": "NEEDS_CONFIRMATION",
                    }
                ),
            )

    def test_non_multiple_interval_keeps_every_bottom_exclusive_grid_point(self) -> None:
        out = resample_frozen_segment(
            pd.DataFrame(
                {"DEPT": [0.0, 0.125, 0.25], "GR": [10.0, 11.0, 12.0]}
            ),
            pd.DataFrame(columns=["well_name", "depth_md"]),
            pd.DataFrame(
                {
                    "well_name": ["SYNTHETICA"],
                    "layer": ["L1"],
                    "top_md": [0.0],
                    "bottom_md": [0.3],
                }
            ),
            pd.Series(
                {
                    "well_name": "SYNTHETICA",
                    "segment_label": "全井",
                    "layer_names": "L1",
                    "status": "FROZEN",
                }
            ),
        )
        np.testing.assert_allclose(out["sample_depth"], [0.0, 0.125, 0.25])

    def test_thinner_than_step_interval_still_contains_its_top(self) -> None:
        out = resample_frozen_segment(
            pd.DataFrame({"DEPT": [1.0], "GR": [10.0]}),
            pd.DataFrame(columns=["well_name", "depth_md"]),
            pd.DataFrame(
                {
                    "well_name": ["SYNTHETICA"],
                    "layer": ["L1"],
                    "top_md": [1.0],
                    "bottom_md": [1.1],
                }
            ),
            pd.Series(
                {
                    "well_name": "SYNTHETICA",
                    "segment_label": "全井",
                    "layer_names": "L1",
                    "status": "FROZEN",
                }
            ),
        )
        np.testing.assert_allclose(out["sample_depth"], [1.0])


if __name__ == "__main__":
    unittest.main()
