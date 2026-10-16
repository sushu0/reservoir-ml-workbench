from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from .identifiers import canonicalize_well_name

TARGETS = (
    "porosity",
    "permeability",
    "original_oil_saturation",
    "shale_content",
)
LABEL_LINEAGES = frozenset({"measured", "formula", "unknown"})
EMPTY_SAMPLE_COLUMNS = (
    "well_name",
    "segment_label",
    "sample_depth",
    "layer",
    "log_match_depth",
    "depth_delta",
    "segment_status",
    "alignment_status",
    "train_eligible",
    "predict_eligible",
    *(
        column
        for target in TARGETS
        for column in (
            target,
            f"{target}_interpolated",
            f"{target}_source_left_md",
            f"{target}_source_right_md",
            f"{target}_label_lineage",
            f"{target}_label_is_measured",
            f"{target}_label_is_formula",
            f"{target}_train_eligible",
        )
    ),
    "sample_excluded_reason",
    "sample_library_version",
)


def _canonical_lineage(value: object) -> str:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in LABEL_LINEAGES else "unknown"


def build_standard_sample_library(
    resampled_by_well: Mapping[str, pd.DataFrame],
    segment_manifest: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    *,
    training_wells: Sequence[str],
    target_excluded_wells: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    frames = [
        frame.copy()
        for frame in resampled_by_well.values()
        if frame is not None and not frame.empty
    ]
    if not frames:
        return pd.DataFrame(columns=EMPTY_SAMPLE_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out["well_name"] = out["well_name"].map(canonicalize_well_name)

    manifest_tags = segment_manifest[
        ["well_name", "segment_label", "status"]
    ].copy()
    manifest_tags["well_name"] = manifest_tags["well_name"].map(
        canonicalize_well_name
    )
    manifest_tags = manifest_tags.rename(columns={"status": "segment_status"})

    summary_cols = [
        column
        for column in (
            "well_name",
            "status",
            "den_status",
            "review_reason_code",
            "core_shift_md",
            "log_shift_md",
        )
        if column in alignment_summary.columns
    ]
    alignment_tags = alignment_summary[summary_cols].copy()
    alignment_tags["well_name"] = alignment_tags["well_name"].map(
        canonicalize_well_name
    )
    alignment_tags = alignment_tags.rename(columns={"status": "alignment_status"})

    out = out.merge(
        manifest_tags,
        on=["well_name", "segment_label"],
        how="left",
        validate="many_to_one",
    )
    out = out.merge(
        alignment_tags,
        on="well_name",
        how="left",
        validate="many_to_one",
    )
    key = ["well_name", "segment_label", "sample_depth"]
    if bool(out.duplicated(key).any()):
        raise ValueError("duplicate sample keys")

    formal = out["well_name"].isin({canonicalize_well_name(well) for well in training_wells})
    frozen = out["segment_status"].eq("FROZEN")
    aligned_ok = out["alignment_status"].eq("OK")
    out["train_eligible"] = formal & frozen & aligned_ok
    out["predict_eligible"] = frozen & out["alignment_status"].isin(
        ["OK", "REVIEW"]
    )

    for target in TARGETS:
        if target not in out.columns:
            out[target] = pd.NA
        lineage_col = f"{target}_label_lineage"
        if lineage_col not in out.columns:
            out[lineage_col] = "unknown"
        else:
            out[lineage_col] = out[lineage_col].map(_canonical_lineage)
        out[f"{target}_label_is_measured"] = out[lineage_col].eq("measured")
        out[f"{target}_label_is_formula"] = out[lineage_col].eq("formula")
        out[f"{target}_train_eligible"] = (
            out["train_eligible"]
            & pd.to_numeric(out[target], errors="coerce").notna()
            & out[f"{target}_label_is_measured"]
        )

    for target, wells in (target_excluded_wells or {}).items():
        if target not in TARGETS:
            raise ValueError(f"unknown target: {target}")
        excluded = {canonicalize_well_name(well) for well in wells}
        out.loc[out["well_name"].isin(excluded), f"{target}_train_eligible"] = False
    out["sample_excluded_reason"] = ""
    out.loc[~frozen, "sample_excluded_reason"] = "segment_not_frozen"
    out.loc[
        frozen & ~aligned_ok & out["alignment_status"].eq("REVIEW"),
        "sample_excluded_reason",
    ] = "alignment_review_prediction_only"
    out.loc[
        frozen & ~out["alignment_status"].isin(["OK", "REVIEW"]),
        "sample_excluded_reason",
    ] = "alignment_not_usable"
    out["sample_library_version"] = "personal_reservoir_foundation_v1"
    return out.sort_values(key).reset_index(drop=True)
