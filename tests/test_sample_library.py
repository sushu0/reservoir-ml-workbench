from __future__ import annotations

import unittest

import pandas as pd

from reservoir_workbench.sample_library import (
    build_standard_sample_library as _build_standard_sample_library,
)



def build_standard_sample_library(frames, manifest, summary):
    return _build_standard_sample_library(
        frames, manifest, summary,
        training_wells=("SYNTHETICA", "SYNTHETICC"),
        target_excluded_wells={"original_oil_saturation": ("SYNTHETICC",)},
    )

class PersonalReservoirSampleLibraryTests(unittest.TestCase):
    def _frames(self) -> dict[str, pd.DataFrame]:
        frames = {
            well: pd.DataFrame(
                {
                    "well_name": [well],
                    "segment_label": ["全井"],
                    "sample_depth": [100.0],
                    "layer": ["L1"],
                    "log_match_depth": [100.0],
                    "depth_delta": [0.0],
                    "GR": [50.0],
                    "GR_NORM": [0.5],
                    "porosity": [12.0],
                    "permeability": [20.0],
                    "original_oil_saturation": [60.0],
                    "porosity_label_lineage": ["measured"],
                    "permeability_label_lineage": ["measured"],
                    "original_oil_saturation_label_lineage": ["measured"],
                }
            )
            for well in ("SYNTHETICA", "SYNTHETICC", "SYNTHETICREVIEWA", "SYNTHETICREVIEWB")
        }
        frames["SYNTHETICA"].loc[0, "permeability_label_lineage"] = "formula"
        frames["SYNTHETICA"].loc[
            0, "original_oil_saturation_label_lineage"
        ] = "unknown"
        return frames

    def test_review_wells_are_prediction_only_and_syntheticc_saturation_is_excluded(self) -> None:
        frames = self._frames()
        manifest = pd.DataFrame(
            {
                "well_name": ["SYNTHETICA", "SYNTHETICC", "SYNTHETICREVIEWA", "SYNTHETICREVIEWB"],
                "segment_label": ["全井"] * 4,
                "status": ["FROZEN"] * 4,
            }
        )
        summary = pd.DataFrame(
            {
                "well_name": ["SYNTHETICA", "SYNTHETICC", "SYNTHETICREVIEWA", "SYNTHETICREVIEWB"],
                "status": ["OK", "OK", "REVIEW", "REVIEW"],
                "den_status": ["has_den"] * 4,
            }
        )
        out = build_standard_sample_library(frames, manifest, summary).set_index(
            "well_name"
        )
        self.assertTrue(bool(out.loc["SYNTHETICA", "train_eligible"]))
        self.assertTrue(bool(out.loc["SYNTHETICA", "porosity_train_eligible"]))
        self.assertFalse(bool(out.loc["SYNTHETICA", "permeability_train_eligible"]))
        self.assertFalse(
            bool(out.loc["SYNTHETICA", "original_oil_saturation_train_eligible"])
        )
        self.assertFalse(bool(out.loc["SYNTHETICREVIEWA", "train_eligible"]))
        self.assertTrue(bool(out.loc["SYNTHETICREVIEWA", "predict_eligible"]))
        self.assertFalse(bool(out.loc["SYNTHETICREVIEWB", "train_eligible"]))
        self.assertFalse(
            bool(out.loc["SYNTHETICC", "original_oil_saturation_train_eligible"])
        )

    def test_duplicate_sample_keys_are_rejected(self) -> None:
        frame = pd.concat([self._frames()["SYNTHETICA"]] * 2, ignore_index=True)
        manifest = pd.DataFrame(
            {
                "well_name": ["SYNTHETICA"],
                "segment_label": ["全井"],
                "status": ["FROZEN"],
            }
        )
        summary = pd.DataFrame({"well_name": ["SYNTHETICA"], "status": ["OK"]})
        with self.assertRaisesRegex(ValueError, "duplicate sample keys"):
            build_standard_sample_library(
                {"SYNTHETICA": frame}, manifest, summary
            )

    def test_empty_input_returns_a_headered_sample_contract(self) -> None:
        out = build_standard_sample_library(
            {},
            pd.DataFrame(columns=["well_name", "segment_label", "status"]),
            pd.DataFrame(columns=["well_name", "status"]),
        )
        self.assertTrue(out.empty)
        self.assertTrue(
            {
                "well_name",
                "segment_label",
                "sample_depth",
                "train_eligible",
                "predict_eligible",
                "porosity_label_lineage",
                "porosity_train_eligible",
            }.issubset(out.columns)
        )


if __name__ == "__main__":
    unittest.main()
