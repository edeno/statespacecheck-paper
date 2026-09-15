"""Figure-3 sensitivity analyses: coverage, sparse-regime composition, trajectory law.

Each setting reruns the complete Figure-3 pipeline (independent matched-null
calibration followed by evaluation on disjoint seeds) with one configuration
change, so its results are comparable with the main summary and never share
calibration with it:

- **HPD coverage** 50% and 80% (the main setting is 95%): thresholds are
  recalibrated at each coverage because the HPD-overlap null distribution
  changes with the region definition.
- **Residual ordinary activity** in the sparse regime (ordinary place-cell rate
  scale 0.005, 0.01, 0.05 in both generation and decoding) and a **lower-rate**
  sparse population: the sparse column's all-event and sparse-cell
  denominators are reported separately.
- **Trajectory law**: the approximate reflected continuous walk, whose
  reflected increments only approximate the decoder's grid transition, to
  quantify how far that legacy benchmark sits from the exact null.

The main setting is fixed in advance (``Figure3Config()``); these runs report
how the operating points move, not which setting is most favorable.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from statespacecheck_paper.figure03_generation import (
    N_CALIBRATION_REALIZATIONS,
    N_REALIZATIONS,
    figure03_summary_payload,
)
from statespacecheck_paper.figure03_protocol import Figure3Config
from statespacecheck_paper.figure03_summary import (
    CONDITION_IDS,
    Figure3RealizationSummary,
    estimate_calibration_thresholds,
    estimate_realization_summary,
)
from statespacecheck_paper.scientific_artifacts import (
    scientific_source_provenance,
    write_json_artifact,
)

FIGURE03_SENSITIVITY_SUMMARY_PATH = Path(
    "manuscript/figures/supplementary/figure03_sensitivity_summary.json"
)


@dataclasses.dataclass(frozen=True)
class SensitivitySetting:
    """One sensitivity configuration: an identifier, a description, and the config."""

    setting_id: str
    description: str
    config: Figure3Config


def sensitivity_settings(base: Figure3Config | None = None) -> tuple[SensitivitySetting, ...]:
    """Return the fixed set of sensitivity settings evaluated around ``base``."""
    if base is None:
        base = Figure3Config()
    settings = [
        SensitivitySetting(
            "hpd_coverage_0.5",
            "HPD regions at 50% coverage (thresholds recalibrated)",
            dataclasses.replace(base, hpd_coverage=0.5),
        ),
        SensitivitySetting(
            "hpd_coverage_0.8",
            "HPD regions at 80% coverage (thresholds recalibrated)",
            dataclasses.replace(base, hpd_coverage=0.8),
        ),
    ]
    for scale in (0.005, 0.01, 0.05):
        settings.append(
            SensitivitySetting(
                f"sparse_ordinary_scale_{scale:g}",
                f"Sparse regime with the ordinary ensemble at {scale:g} of its rate "
                "(generation and decoding)",
                dataclasses.replace(base, sparse_control_ordinary_rate_scale=scale),
            )
        )
    settings.append(
        SensitivitySetting(
            "sparse_lower_rate",
            "Sparse regime with the sparse cells at half their reference peak rate",
            dataclasses.replace(
                base, sparse_cell_peak_rate_per_step=0.5 * base.sparse_cell_peak_rate_per_step
            ),
        )
    )
    settings.append(
        SensitivitySetting(
            "continuous_reflected_trajectory",
            "Approximate benchmark: reflected continuous Gaussian walk instead of the "
            "decoder's grid transition (calibration and evaluation)",
            dataclasses.replace(base, trajectory_model="continuous_reflected"),
        )
    )
    return tuple(settings)


def _compact_summary(
    config: Figure3Config, summary: Figure3RealizationSummary
) -> dict[str, object]:
    """Return the subset of the Figure-3 payload a sensitivity comparison needs."""
    payload = figure03_summary_payload(config, summary)
    keep = (
        "realizations",
        "calibration",
        "metric_order",
        "flag_rules",
        "condition_order",
        "median_flag_percentages",
        "pooled_flag_percentages",
        "event_counts_by_realization",
        "accuracy_metric_order",
        "median_decoding_accuracy",
        "sparse_population",
        "matched_null_rank_pvalue_tail_percentages",
        "median_flag_percentage_standard_errors",
    )
    compact = {key: payload[key] for key in keep}
    compact["flag_percentages_by_realization"] = payload["flag_percentages_by_realization"]
    compact["interquartile_flag_percentages"] = {
        "q25": np.nanpercentile(summary.flag_percentages_by_realization, 25, axis=0),
        "q75": np.nanpercentile(summary.flag_percentages_by_realization, 75, axis=0),
    }
    return compact


def run_figure03_sensitivity(
    *,
    base: Figure3Config | None = None,
    n_realizations: int = N_REALIZATIONS,
    n_calibration_realizations: int = N_CALIBRATION_REALIZATIONS,
    n_jobs: int = -1,
    summary_path: Path = FIGURE03_SENSITIVITY_SUMMARY_PATH,
) -> dict[str, object]:
    """Run every sensitivity setting and write one combined JSON summary."""
    if base is None:
        base = Figure3Config()
    results: dict[str, object] = {}
    for setting in sensitivity_settings(base):
        print(f"[sensitivity] {setting.setting_id}: {setting.description}")
        calibration = estimate_calibration_thresholds(
            setting.config, n_calibration_realizations=n_calibration_realizations, n_jobs=n_jobs
        )
        summary = estimate_realization_summary(
            setting.config, calibration=calibration, n_realizations=n_realizations, n_jobs=n_jobs
        )
        print(
            f"  thresholds={calibration.diagnostic_thresholds} "
            f"null={np.array2string(calibration.pooled_null_flag_percentages, precision=2)} "
            f"medians=\n{np.array2string(summary.median_flag_percentages, precision=2)}"
        )
        changed = {
            field.name: getattr(setting.config, field.name)
            for field in dataclasses.fields(Figure3Config)
            if field.name != "place_field_centers"
            and getattr(setting.config, field.name) != getattr(base, field.name)
        }
        results[setting.setting_id] = {
            "description": setting.description,
            "changed_configuration": changed,
            "summary": _compact_summary(setting.config, summary),
        }
    payload = {
        "schema_version": 1,
        "figure": "figure03_sensitivity",
        "base_configuration": dataclasses.asdict(base),
        "condition_order": list(CONDITION_IDS),
        "settings": results,
        "provenance": {"source": scientific_source_provenance()},
    }
    written = write_json_artifact(summary_path, payload)
    print(f"Saved sensitivity summary to {written}")
    return payload
