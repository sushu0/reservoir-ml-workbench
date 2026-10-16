from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from typing import Sequence
import numpy as np
import pandas as pd
from reservoir_workbench import model_compare


class ModelCompareTests(unittest.TestCase):
    def test_string_false_and_review_wells_never_enter_training_cohort(self) -> None:
        sample = pd.DataFrame(
            [
                {
                    "well_name": "SYNTHETICA",
                    "sample_depth": 100.0,
                    "porosity": 10.0,
                    "porosity_train_eligible": "False",
                    "GR": 80.0,
                },
                {
                    "well_name": "SYNTHETICREVIEWA",
                    "sample_depth": 101.0,
                    "porosity": 11.0,
                    "porosity_train_eligible": True,
                    "GR": 81.0,
                },
            ]
        )

        cohort = model_compare.build_target_cohort(
            sample,
            "porosity",
            ("SYNTHETICA", "SYNTHETICREVIEWA"),
            review_wells=("SYNTHETICREVIEWA",),
        )

        self.assertTrue(cohort.empty)

    def test_lowo_prediction_contract_is_immutable(self) -> None:
        prediction = model_compare.LowoPrediction(
            sample_id=1,
            well_name="SYNTHETICA",
            heldout_well="SYNTHETICA",
            target="porosity",
            model_family="M0",
            y_true=20.0,
            y_pred=19.5,
        )

        self.assertEqual(prediction.model_family, "M0")
        with self.assertRaises(FrozenInstanceError):
            prediction.y_pred = 20.0

    def make_synthetic_sample(self, wells: Sequence[str], rows_per_well: int) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        rows: list[dict[str, object]] = []
        for well_index, well in enumerate(wells):
            for row_index in range(rows_per_well):
                gr = 45.0 + well_index * 2.0 + row_index * 0.2
                rows.append(
                    {
                        "well_name": well,
                        "sample_depth": 100.0 + row_index * 0.125,
                        "train_eligible": True,
                        "porosity_train_eligible": True,
                        "permeability_train_eligible": True,
                        "original_oil_saturation_train_eligible": True,
                        "SP": -20.0 + rng.normal(),
                        "GR": gr,
                        "RTCAL": 10.0 + rng.random(),
                        "AC": 220.0 + rng.normal(),
                        "DEN": 2.4 + rng.normal(scale=0.02),
                        "CNL": 18.0 + rng.normal(),
                        "layer_group": "L1" if row_index < rows_per_well / 2 else "L2",
                        "lithology_group": "sandstone",
                        "facies_group": "",
                        "facies_source": "",
                        "facies_review_status": "",
                        "porosity": 30.0 - 0.15 * gr + rng.normal(scale=0.1),
                        "permeability": max(
                            0.01, 10 ** (2.0 - 0.015 * gr + rng.normal(scale=0.02))
                        ),
                        "original_oil_saturation": 65.0
                        - 0.2 * gr
                        + rng.normal(scale=0.2),
                    }
                )
        return pd.DataFrame(rows)

    def test_three_families_use_identical_heldout_wells_and_rows(self) -> None:
        sample = self.make_synthetic_sample(wells=("A", "B", "C", "D"), rows_per_well=12)
        sample["facies_group"] = np.where(
            sample.groupby("well_name").cumcount().mod(2).eq(0), "channel", "bar"
        )
        sample["facies_source"] = "manual_review"
        sample["facies_review_status"] = "APPROVED"
        result = model_compare.run_same_cohort_lowo(
            sample, "porosity", ("A", "B", "C", "D"), random_seed=42
        )
        self.assertEqual({"M0", "M1", "M2"}, set(result.metrics["model_family"]))
        heldout = result.predictions.groupby("model_family")["heldout_well"].apply(
            lambda x: tuple(sorted(x.unique()))
        )
        self.assertEqual(heldout["M0"], heldout["M1"])
        self.assertEqual(heldout["M1"], heldout["M2"])
        row_sets = result.predictions.groupby("model_family")["sample_id"].apply(
            lambda x: set(x)
        )
        self.assertEqual(row_sets["M0"], row_sets["M1"])
        self.assertEqual(row_sets["M1"], row_sets["M2"])
        self.assertTrue(result.metrics["r2_random"].isna().all())
        self.assertEqual(set(result.metrics["lowo_well_count"]), {4})
        for column in ("r2_lowo", "mae_lowo", "rmse_lowo", "nrmse_lowo"):
            self.assertIn(column, result.metrics.columns)

    def test_inverse_target_then_applies_physical_bounds(self) -> None:
        permeability = model_compare._inverse_target_with_physical_bounds(
            "permeability", np.array([-10.0, 0.0, 10.0])
        )
        porosity = model_compare._inverse_target_with_physical_bounds(
            "porosity", np.array([-5.0, 20.0, 50.0])
        )
        saturation = model_compare._inverse_target_with_physical_bounds(
            "original_oil_saturation", np.array([-1.0, 50.0, 101.0])
        )

        np.testing.assert_allclose(permeability, [1e-6, 1.0, 20000.0])
        np.testing.assert_allclose(porosity, [0.0, 20.0, 45.0])
        np.testing.assert_allclose(saturation, [0.0, 50.0, 100.0])

    def test_target_cohort_uses_only_finite_sample_depth_when_column_is_available(self) -> None:
        sample = pd.DataFrame(
            {
                "well_name": ["SYNTHETICA", "SYNTHETICA"],
                "sample_depth": [100.0, "bad-depth"],
                "porosity_train_eligible": [True, True],
                "GR": [50.0, 51.0],
                "porosity": [20.0, 21.0],
            }
        )

        cohort = model_compare.build_target_cohort(sample, "porosity", ["SYNTHETICA"])

        self.assertEqual(cohort["sample_depth"].tolist(), [100.0])

    def test_m2_uses_confirmed_facies_but_never_data_cluster(self) -> None:
        sample = pd.DataFrame(
            {
                "facies_group": ["channel", "bar"],
                "facies_source": ["人工复核", "聚类"],
                "facies_review_status": ["APPROVED", "DRAFT"],
                "layer_group": ["L1", "UNKNOWN_LAYER"],
                "lithology_group": ["sandstone", "unknown_lithology"],
                "data_cluster": [0, 1],
            }
        )

        prepared, columns, audit = model_compare.resolve_m2_constraints(sample)

        self.assertIn("trusted_facies_group", columns)
        self.assertNotIn("data_cluster", columns)
        self.assertEqual(
            prepared["trusted_facies_group"].tolist(), ["channel", "UNKNOWN_FACIES"]
        )
        self.assertEqual(audit["confirmed_facies_rows"], 1)

    def test_target_cohort_excludes_review_wells_and_nonpositive_permeability(self) -> None:
        sample = pd.DataFrame(
            {
                "well_name": ["SYNTHETICA", "SYNTHETICB", "SYNTHETICREVIEWA", "SYNTHETICB"],
                "train_eligible": [True, True, True, True],
                "GR": [50.0, 60.0, 70.0, 80.0],
                "permeability": [1.0, 10.0, 100.0, 0.0],
            }
        )

        cohort = model_compare.build_target_cohort(
            sample, "permeability", ["SYNTHETICA", "SYNTHETICB"]
        )

        self.assertEqual(cohort["well_name"].tolist(), ["SYNTHETICA", "SYNTHETICB"])
        self.assertTrue((cohort["permeability"] > 0).all())

    def test_target_specific_eligibility_excludes_syntheticc_saturation_even_when_value_exists(
        self,
    ) -> None:
        sample = pd.DataFrame(
            {
                "well_name": ["SYNTHETICA", "SYNTHETICC"],
                "sample_depth": [1.0, 1.0],
                "train_eligible": [True, True],
                "original_oil_saturation_train_eligible": [True, False],
                "GR": [50.0, 60.0],
                "original_oil_saturation": [40.0, 55.0],
            }
        )
        cohort = model_compare.build_target_cohort(
            sample, "original_oil_saturation", ["SYNTHETICA", "SYNTHETICC"]
        )
        self.assertEqual(["SYNTHETICA"], cohort["well_name"].tolist())

    def test_all_null_numeric_features_are_excluded_from_model_schema(self) -> None:
        frame = pd.DataFrame(
            {
                "SP": [np.nan, np.nan],
                "GR": [50.0, 60.0],
                "RTCAL": [10.0, 11.0],
                "DEN": [2.3, 2.4],
                "CNL": [np.nan, np.nan],
            }
        )

        features = model_compare.resolve_usable_numeric_features(
            frame, ("SP", "GR", "RTCAL", "DEN", "CNL")
        )

        self.assertEqual(("GR", "RTCAL", "DEN"), features)

    def test_m2_is_not_applicable_without_approved_category(self) -> None:
        sample = self.make_synthetic_sample(
            wells=("A", "B", "C", "D"), rows_per_well=12
        ).drop(columns=["facies_group", "facies_source", "facies_review_status"])

        result = model_compare.run_same_cohort_lowo(
            sample, "porosity", ("A", "B", "C", "D"), random_seed=42
        )

        self.assertEqual({"M0", "M1"}, set(result.metrics["model_family"]))
        self.assertNotIn("M2", set(result.predictions["model_family"]))

    def test_hgb_explicitly_disables_internal_early_stopping(self) -> None:
        m1 = model_compare._model_for_family(
            "M1", ["GR", "RTCAL", "DEN"], [], 42
        )

        self.assertIs(m1.named_steps["model"].early_stopping, False)

if __name__ == "__main__":
    unittest.main()
