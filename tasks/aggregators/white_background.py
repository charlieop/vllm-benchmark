"""Optional global aggregator; default aggregation already gives this mean."""

from statistics import mean


def aggregate(records: list[dict], config: dict) -> dict[str, float]:
    values = [row["metrics"]["white_fraction"] for row in records if "white_fraction" in row["metrics"]]
    return {"white_fraction": mean(values)} if values else {}
