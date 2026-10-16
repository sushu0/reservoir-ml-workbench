from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .identifiers import canonicalize_well_name

LOG_COLUMNS = ("SP", "GR", "RTCAL", "AC", "DEN", "CNL")
TARGET_COLUMNS = (
    "porosity",
    "permeability",
    "original_oil_saturation",
    "shale_content",
)
LABEL_LINEAGES = frozenset({"measured", "formula", "unknown"})


@dataclass(frozen=True)
class ResamplingConfig:
    step_m: float = 0.125
    max_log_delta_m: float = 0.200
    max_core_gap_m: float = 0.500


def _interpolate_one_target(
    depths: np.ndarray,
    values: np.ndarray,
    query: float,
    max_gap: float,
) -> tuple[float, bool, float, float]:
    valid = np.isfinite(depths) & np.isfinite(values)
    x, y = depths[valid], values[valid]
    order = np.argsort(x)
    x, y = x[order], y[order]
    exact = np.flatnonzero(np.isclose(x, query, atol=1e-9, rtol=0.0))
    if exact.size:
        depth = float(x[exact[0]])
        return float(y[exact[0]]), False, depth, depth
    right = int(np.searchsorted(x, query, side="right"))
    left = right - 1
    if left < 0 or right >= len(x) or float(x[right] - x[left]) > max_gap:
        return np.nan, False, np.nan, np.nan
    weight = (query - x[left]) / (x[right] - x[left])
    return (
        float(y[left] + weight * (y[right] - y[left])),
        True,
        float(x[left]),
        float(x[right]),
    )


def _normalize_lineage(value: object) -> str:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in LABEL_LINEAGES else "unknown"


def _target_lineages(frame: pd.DataFrame, target: str) -> pd.Series:
    lineage_col = f"{target}_label_lineage"
    measured_col = f"{target}_label_is_measured"
    formula_col = f"{target}_label_is_formula"
    if lineage_col in frame.columns:
        return frame[lineage_col].map(_normalize_lineage)
    if measured_col in frame.columns or formula_col in frame.columns:
        measured = frame.get(measured_col, pd.Series(False, index=frame.index)).fillna(
            False
        ).astype(bool)
        formula = frame.get(formula_col, pd.Series(False, index=frame.index)).fillna(
            False
        ).astype(bool)
        result = pd.Series("unknown", index=frame.index, dtype=object)
        result.loc[measured & ~formula] = "measured"
        result.loc[formula] = "formula"
        return result
    if "label_lineage" in frame.columns:
        return frame["label_lineage"].map(_normalize_lineage)
    if "label_is_measured" in frame.columns or "label_is_formula" in frame.columns:
        measured = frame.get(
            "label_is_measured", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        formula = frame.get(
            "label_is_formula", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        result = pd.Series("unknown", index=frame.index, dtype=object)
        result.loc[measured & ~formula] = "measured"
        result.loc[formula] = "formula"
        return result
    return pd.Series("unknown", index=frame.index, dtype=object)


def _interpolated_lineage(
    frame: pd.DataFrame,
    target: str,
    left_depth: float,
    right_depth: float,
) -> str:
    if not np.isfinite(left_depth) or not np.isfinite(right_depth):
        return "unknown"
    depths = pd.to_numeric(frame["depth_md"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(
        frame.get(target, pd.Series(np.nan, index=frame.index)), errors="coerce"
    ).to_numpy(dtype=float)
    lineages = _target_lineages(frame, target).to_numpy(dtype=object)

    def lineage_at(depth: float) -> str:
        matches = np.flatnonzero(
            np.isfinite(values)
            & np.isclose(depths, depth, atol=1e-9, rtol=0.0)
        )
        return _normalize_lineage(lineages[matches[0]]) if matches.size else "unknown"

    left_lineage = lineage_at(left_depth)
    right_lineage = lineage_at(right_depth)
    if left_lineage == "formula" or right_lineage == "formula":
        return "formula"
    if left_lineage == right_lineage == "measured":
        return "measured"
    return "unknown"


def resample_frozen_segment(
    log_df: pd.DataFrame,
    aligned_anchors: pd.DataFrame,
    layer_intervals: pd.DataFrame,
    manifest_row: pd.Series,
    config: ResamplingConfig = ResamplingConfig(),
) -> pd.DataFrame:
    if str(manifest_row.get("status", "")) != "FROZEN":
        raise ValueError("segment is not FROZEN")
    well = canonicalize_well_name(manifest_row["well_name"])
    selected_layers = tuple(
        filter(None, str(manifest_row.get("layer_names", "")).split(";"))
    )
    intervals = layer_intervals.copy()
    intervals["well_name"] = intervals["well_name"].map(canonicalize_well_name)
    intervals = intervals.loc[
        intervals["well_name"].eq(well)
        & intervals["layer"].astype(str).isin(selected_layers)
    ].copy()
    intervals = intervals.sort_values(["top_md", "bottom_md"])
    if intervals.empty:
        raise ValueError(f"no selected layer intervals for {well}")

    depth_col = "DEPT" if "DEPT" in log_df.columns else str(log_df.columns[0])
    log_depth = pd.to_numeric(log_df[depth_col], errors="coerce").to_numpy(
        dtype=float
    )
    curve_columns = [
        column
        for column in log_df.columns
        if column in LOG_COLUMNS or str(column).endswith("_NORM")
    ]
    anchors = aligned_anchors.copy()
    anchors["well_name"] = anchors["well_name"].map(canonicalize_well_name)
    anchors = anchors.loc[anchors["well_name"].eq(well)].copy()

    rows: list[dict[str, object]] = []
    for _, interval in intervals.iterrows():
        top = float(interval["top_md"])
        bottom = float(interval["bottom_md"])
        layer = str(interval["layer"])
        if not np.isfinite(top) or not np.isfinite(bottom) or bottom <= top:
            raise ValueError(f"invalid layer interval for {well}:{layer}")
        if not np.isfinite(config.step_m) or config.step_m <= 0:
            raise ValueError("step_m must be positive")
        grid_count = max(
            1,
            int(np.ceil((bottom - top) / config.step_m - 1e-12)),
        )
        grid = top + np.arange(grid_count) * config.step_m
        anchor_depth = pd.to_numeric(anchors["depth_md"], errors="coerce")
        layer_anchors = anchors.loc[anchor_depth.ge(top) & anchor_depth.lt(bottom)]

        for depth in grid:
            rec: dict[str, object] = {
                "well_name": well,
                "segment_label": manifest_row["segment_label"],
                "sample_depth": float(depth),
                "layer": layer,
            }
            diff = np.abs(log_depth - depth)
            finite_log = np.isfinite(diff)
            if bool(finite_log.any()):
                idx = int(np.flatnonzero(finite_log)[np.argmin(diff[finite_log])])
            else:
                idx = -1
            if idx >= 0 and float(diff[idx]) <= config.max_log_delta_m:
                rec["log_match_depth"] = float(log_depth[idx])
                rec["depth_delta"] = float(diff[idx])
                for curve in curve_columns:
                    value = pd.to_numeric(log_df.iloc[idx].get(curve), errors="coerce")
                    rec[curve] = float(value) if np.isfinite(value) else np.nan
            else:
                rec["log_match_depth"] = np.nan
                rec["depth_delta"] = np.nan
                rec.update({curve: np.nan for curve in curve_columns})

            anchor_depths = pd.to_numeric(
                layer_anchors.get("depth_md"), errors="coerce"
            ).to_numpy(dtype=float)
            for target in TARGET_COLUMNS:
                target_values = pd.to_numeric(
                    layer_anchors.get(
                        target, pd.Series(np.nan, index=layer_anchors.index)
                    ),
                    errors="coerce",
                ).to_numpy(dtype=float)
                value, interpolated, left, right = _interpolate_one_target(
                    anchor_depths,
                    target_values,
                    float(depth),
                    config.max_core_gap_m,
                )
                rec[target] = value
                rec[f"{target}_interpolated"] = interpolated
                rec[f"{target}_source_left_md"] = left
                rec[f"{target}_source_right_md"] = right
                rec[f"{target}_label_lineage"] = _interpolated_lineage(
                    layer_anchors, target, left, right
                )
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["sample_depth"]).reset_index(drop=True)
