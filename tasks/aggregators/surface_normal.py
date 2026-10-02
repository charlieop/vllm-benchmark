"""Image-level mean of the per-image surface-normal metrics (Marigold / RINO convention).

Samples with NO successfully scored trial (generation failed after retries, or scoring
raised) are not silently dropped. ``failed_sample_policy`` decides, before evaluation:

  * "penalize" (default): the sample counts as fully degenerate — every valid pixel at
    ``degenerate_error_deg`` (default 90 deg), so mean = median = 90, all threshold
    percentages = 0, degenerate_pixel_pct = 100.
  * "exclude": drop it (the coverage numbers below still show how many were dropped).

Both policies also report ``samples_scored`` and ``samples_failed`` so results can be
compared fairly.
"""

from statistics import mean

METRIC_NAMES = ("mean_angular_error", "median_angular_error", "pct_within_11_25",
                "pct_within_22_5", "pct_within_30", "degenerate_pixel_pct")


def _penalty(error_deg: float) -> dict[str, float]:
    thresholds = {"pct_within_11_25": 11.25, "pct_within_22_5": 22.5, "pct_within_30": 30.0}
    row = {"mean_angular_error": error_deg, "median_angular_error": error_deg, "degenerate_pixel_pct": 100.0}
    row.update({name: (100.0 if error_deg < t else 0.0) for name, t in thresholds.items()})
    return row


def aggregate(records: list[dict], config: dict) -> dict[str, float]:
    policy = config.get("failed_sample_policy", "penalize")
    if policy not in {"penalize", "exclude"}:
        raise ValueError("failed_sample_policy must be 'penalize' or 'exclude'")
    penalty = _penalty(float(config.get("degenerate_error_deg", 90.0)))
    rows, failed = [], 0
    for record in records:
        metrics = record["metrics"]
        if all(name in metrics for name in METRIC_NAMES):
            rows.append(metrics)
        else:
            failed += 1
            if policy == "penalize":
                rows.append({name: metrics.get(name, penalty[name]) for name in METRIC_NAMES})
    result = {name: mean(row[name] for row in rows) for name in METRIC_NAMES} if rows else {}
    result["samples_scored"] = float(len(records) - failed)
    result["samples_failed"] = float(failed)
    return result
