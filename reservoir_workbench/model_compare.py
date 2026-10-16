from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



TARGETS = ("porosity", "permeability", "original_oil_saturation")
NUMERIC_FEATURES = ("SP", "GR", "RTCAL", "AC", "DEN", "CNL")
FORMAL_NUMERIC_FEATURES = ("GR", "RTCAL", "DEN")
CONFIRMED_FACIES_STATUSES = {"CONFIRMED", "APPROVED"}
PHYSICAL_RANGES = {
    "porosity": (0.0, 45.0),
    "permeability": (1e-6, 20000.0),
    "original_oil_saturation": (0.0, 100.0),
}


@dataclass(frozen=True)
class ModelFamilySpec:
    code: str
    display_name: str
    estimator: str
    uses_geology: bool


MODEL_FAMILIES = (
    ModelFamilySpec("M0", "传统线性回归基准", "linear", False),
    ModelFamilySpec("M1", "全样本机器学习", "hgb", False),
    ModelFamilySpec("M2", "分类/相控机器学习", "hgb", True),
)


def resolve_usable_numeric_features(
    frame: pd.DataFrame, requested: Sequence[str] = NUMERIC_FEATURES
) -> tuple[str, ...]:
    """Return requested numeric columns that contain at least one finite value."""

    usable: list[str] = []
    for name in requested:
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).any():
            usable.append(str(name))
    return tuple(usable)


def prepare_model_frame(sample_df: pd.DataFrame) -> pd.DataFrame:
    frame = sample_df.copy()
    if "well_name" not in frame.columns:
        raise KeyError("model frame requires well_name")
    frame["well_name"] = frame["well_name"].astype(str).str.strip()
    frame["train_eligible"] = frame.get("train_eligible", False)
    return frame


def resolve_m2_constraints(
    sample_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    frame = sample_df.copy()
    columns: list[str] = []
    facies = (
        frame.get("facies_group", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    source = (
        frame.get("facies_source", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    status = (
        frame.get("facies_review_status", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.upper()
    )
    confirmed = facies.ne("") & source.ne("") & status.isin(CONFIRMED_FACIES_STATUSES)
    frame["trusted_facies_group"] = facies.where(confirmed, "UNKNOWN_FACIES")
    if confirmed.any():
        columns.append("trusted_facies_group")
    audit = {
        "constraint_columns": columns,
        "applicable": bool(columns),
        "status": "APPLICABLE" if columns else "M2_NOT_APPLICABLE",
        "reason_codes": [] if columns else ["NO_APPROVED_GEOLOGIC_CATEGORY"],
        "confirmed_facies_rows": int(confirmed.sum()),
        "unconfirmed_facies_rows": int((facies.ne("") & ~confirmed).sum()),
        "data_cluster_formal_use": False,
    }
    return frame, columns, audit


def build_target_cohort(
    sample_df: pd.DataFrame, target: str, training_wells: Sequence[str],
    *, review_wells: Sequence[str] = (),
) -> pd.DataFrame:
    frame = prepare_model_frame(sample_df)
    if target not in frame.columns:
        return frame.iloc[0:0].copy()
    allowed = {str(well).strip() for well in training_wells}.difference(review_wells)
    target_values = pd.to_numeric(frame[target], errors="coerce")
    numeric = frame.reindex(columns=NUMERIC_FEATURES).apply(pd.to_numeric, errors="coerce")
    eligibility_col = (
        f"{target}_train_eligible"
        if f"{target}_train_eligible" in frame.columns
        else "train_eligible"
    )
    strict_eligibility = frame[eligibility_col].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    )
    mask = (
        strict_eligibility
        & frame["well_name"].isin(allowed)
        & ~frame["well_name"].isin(review_wells)
        & target_values.notna()
        & numeric.notna().any(axis=1)
    )
    sample_depth = None
    if "sample_depth" in frame.columns:
        sample_depth = pd.to_numeric(frame["sample_depth"], errors="coerce")
        mask &= sample_depth.notna() & np.isfinite(sample_depth)
    if target == "permeability":
        mask &= target_values.gt(0)
    cohort = frame.loc[mask].copy()
    cohort[target] = target_values.loc[mask]
    if sample_depth is not None:
        cohort["sample_depth"] = sample_depth.loc[mask]
    sort_columns = ["well_name", "sample_depth"] if "sample_depth" in cohort.columns else ["well_name"]
    return cohort.sort_values(sort_columns).reset_index(drop=True)


def _transform_target(target: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.log10(values) if target == "permeability" else values


def _inverse_target(target: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if target == "permeability":
        with np.errstate(over="ignore", invalid="ignore"):
            return np.power(10.0, values)
    return values


def _inverse_target_with_physical_bounds(target: str, values: np.ndarray) -> np.ndarray:
    restored = _inverse_target(target, values)
    lower, upper = PHYSICAL_RANGES[target]
    return np.clip(restored, lower, upper)


@dataclass(frozen=True)
class LowoPrediction:
    sample_id: int
    well_name: str
    heldout_well: str
    target: str
    model_family: str
    y_true: float
    y_pred: float


@dataclass(frozen=True)
class ModelComparisonResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    cohort_manifest: pd.DataFrame


def _model_for_family(
    family: str, numeric: list[str], categorical: list[str], random_seed: int
) -> Pipeline:
    if family == "M0":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
    if family == "M1":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        random_state=random_seed,
                        max_iter=200,
                        early_stopping=False,
                    ),
                ),
            ]
        )
    if family != "M2":
        raise ValueError(f"unknown model family: {family}")
    preprocessor = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=random_seed,
                    max_iter=200,
                    early_stopping=False,
                ),
            ),
        ]
    )


def _metric_row(
    target: str,
    family: str,
    truth: np.ndarray,
    pred: np.ndarray,
    lowo_well_count: int,
) -> dict[str, object]:
    rmse = float(np.sqrt(mean_squared_error(truth, pred)))
    mae = float(mean_absolute_error(truth, pred))
    r2 = float(r2_score(truth, pred))
    scale = float(np.nanpercentile(truth, 95) - np.nanpercentile(truth, 5))
    row = {
        "target": target,
        "model_family": family,
        "r2_lowo": r2,
        "r2_random": np.nan,
        "mae_lowo": mae,
        "rmse_lowo": rmse,
        "nrmse_lowo": rmse / scale if scale > 0 else np.nan,
        "nmae_lowo": mae / scale if scale > 0 else np.nan,
        "train_rows": int(len(truth)),
        "lowo_well_count": int(lowo_well_count),
        "top3_feature_importance_sum": np.nan,
        "feature_importance_status": "NOT_COMPUTED",
    }
    if target == "permeability":
        safe_pred = np.clip(pred, 1e-12, None)
        safe_truth = np.clip(truth, 1e-12, None)
        log_error = np.abs(np.log10(safe_pred) - np.log10(safe_truth))
        ratio = np.maximum(safe_pred / safe_truth, safe_truth / safe_pred)
        row.update(
            {
                "mean_abs_log10_error": float(np.mean(log_error)),
                "factor2_accuracy": float(np.mean(ratio <= 2.0)),
                "factor3_accuracy": float(np.mean(ratio <= 3.0)),
            }
        )
    return row


def run_same_cohort_lowo(
    sample_df: pd.DataFrame,
    target: str,
    training_wells: Sequence[str],
    random_seed: int = 42,
) -> ModelComparisonResult:
    cohort = build_target_cohort(sample_df, target, training_wells).copy()
    if cohort.empty or cohort["well_name"].nunique() < 2:
        raise ValueError("LOWO requires eligible samples from at least two wells")
    cohort["sample_id"] = np.arange(len(cohort), dtype=int)
    constrained, categorical, _ = resolve_m2_constraints(cohort)
    numeric = list(resolve_usable_numeric_features(constrained))
    families = ("M0", "M1", "M2") if categorical else ("M0", "M1")
    prediction_rows: list[pd.DataFrame] = []
    transformed_target = _transform_target(
        target, constrained[target].to_numpy(dtype=float)
    )
    for heldout in sorted(constrained["well_name"].unique()):
        test_mask = constrained["well_name"].eq(heldout).to_numpy()
        train_mask = ~test_mask
        for family in families:
            cat = categorical if family == "M2" else []
            columns = numeric + cat
            model = _model_for_family(family, numeric, cat, random_seed)
            model.fit(constrained.loc[train_mask, columns], transformed_target[train_mask])
            transformed_pred = model.predict(constrained.loc[test_mask, columns])
            raw_pred = _inverse_target(target, transformed_pred)
            lower, upper = PHYSICAL_RANGES[target]
            clipped_pred = np.clip(raw_pred, lower, upper)
            violations = (
                ~np.isfinite(raw_pred) | (raw_pred < lower) | (raw_pred > upper)
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "sample_id": constrained.loc[test_mask, "sample_id"].to_numpy(),
                        "well_name": constrained.loc[test_mask, "well_name"].to_numpy(),
                        "heldout_well": heldout,
                        "target": target,
                        "model_family": family,
                        "y_true": constrained.loc[test_mask, target].to_numpy(dtype=float),
                        "y_pred": raw_pred,
                        "y_pred_raw": raw_pred,
                        "y_pred_clipped": clipped_pred,
                        "physical_violation_raw": violations,
                    }
                )
            )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics = pd.DataFrame(
        [
            _metric_row(
                target,
                family,
                group["y_true"].to_numpy(),
                group["y_pred"].to_numpy(),
                group["heldout_well"].nunique(),
            )
            for family, group in predictions.groupby("model_family", sort=True)
        ]
    )
    manifest = constrained[["sample_id", "well_name", "sample_depth", target]].copy()
    return ModelComparisonResult(metrics, predictions, manifest)


def fit_selected_models(
    sample_df: pd.DataFrame,
    target: str,
    training_wells: Sequence[str],
    random_seed: int = 42,
) -> dict[str, object]:
    cohort = build_target_cohort(sample_df, target, training_wells)
    constrained, categorical, _ = resolve_m2_constraints(cohort)
    models: dict[str, object] = {}
    families = ("M0", "M1", "M2") if categorical else ("M0", "M1")
    for family in families:
        models[family] = fit_model_family(
            sample_df,
            target,
            training_wells,
            family,
            random_seed=random_seed,
        )
    return models


def fit_model_family(
    sample_df: pd.DataFrame,
    target: str,
    training_wells: Sequence[str],
    family: str,
    *,
    random_seed: int = 42,
) -> object:
    cohort = build_target_cohort(sample_df, target, training_wells)
    constrained, categorical, _ = resolve_m2_constraints(cohort)
    numeric = list(resolve_usable_numeric_features(constrained))
    selected_categories = categorical if family == "M2" else []
    if family == "M2" and not selected_categories:
        raise ValueError("M2_NOT_APPLICABLE: no approved geologic category")
    model = _model_for_family(family, numeric, selected_categories, random_seed)
    y = _transform_target(target, constrained[target].to_numpy(dtype=float))
    model.fit(constrained[numeric + selected_categories], y)
    return model
