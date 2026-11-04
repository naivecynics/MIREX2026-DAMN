import math
from itertools import combinations
from typing import Sequence

import numpy as np


def aggregate_prompt_scores(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["cid"]), str(row["source"]))
        grouped.setdefault(key, []).append(float(row["cosine_similarity"]))
    result: list[dict[str, object]] = []
    for (cid, source), raw_values in sorted(grouped.items()):
        values = np.asarray(raw_values, dtype=np.float64)
        mean = float(values.mean())
        result.append(
            {
                "cid": cid,
                "source": source,
                "sample_count": int(values.size),
                "cosine_similarity": mean,
                "cosine_distance": 1.0 - mean,
                "within_prompt_std_similarity": (
                    float(values.std(ddof=1)) if values.size > 1 else 0.0
                ),
            }
        )
    return result


def summarize(
    rows: Sequence[dict[str, object]],
    *,
    seed: int = 42,
    bootstrap_samples: int = 10_000,
) -> list[dict[str, object]]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    grouped: dict[str, list[float]] = {}
    within_prompt: dict[str, list[float]] = {}
    for row in rows:
        source = str(row["source"])
        grouped.setdefault(source, []).append(
            float(row["cosine_similarity"])
        )
        if "within_prompt_std_similarity" in row:
            within_prompt.setdefault(source, []).append(
                float(row["within_prompt_std_similarity"])
            )
    rng = np.random.default_rng(seed)
    result: list[dict[str, object]] = []
    for source, raw_values in grouped.items():
        values = np.asarray(raw_values, dtype=np.float64)
        resampled = rng.choice(
            values, size=(bootstrap_samples, values.size), replace=True
        ).mean(axis=1)
        ci_low = float(np.quantile(resampled, 0.025))
        ci_high = float(np.quantile(resampled, 0.975))
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        result.append(
            {
                "source": source,
                "count": int(values.size),
                "mean_similarity": float(values.mean()),
                "std_similarity": std,
                "sem_similarity": std / math.sqrt(values.size),
                "median_similarity": float(np.median(values)),
                "mean_distance": 1.0 - float(values.mean()),
                "mean_within_prompt_std_similarity": (
                    float(np.mean(within_prompt[source]))
                    if source in within_prompt
                    else 0.0
                ),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "bootstrap_distance_ci95_low": 1.0 - ci_high,
                "bootstrap_distance_ci95_high": 1.0 - ci_low,
            }
        )
    return sorted(result, key=lambda row: str(row["source"]))


def paired_comparisons(
    rows: Sequence[dict[str, object]],
    sources: Sequence[str],
    *,
    seed: int = 42,
    bootstrap_samples: int = 10_000,
) -> list[dict[str, object]]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    by_source: dict[str, dict[str, float]] = {source: {} for source in sources}
    for row in rows:
        source = str(row["source"])
        if source in by_source:
            by_source[source][str(row["cid"])] = float(row["cosine_similarity"])
    rng = np.random.default_rng(seed)
    result: list[dict[str, object]] = []
    for source_a, source_b in combinations(sources, 2):
        cids = sorted(set(by_source[source_a]) & set(by_source[source_b]))
        if not cids:
            raise ValueError(f"no paired pieces for {source_a} and {source_b}")
        delta = np.asarray(
            [by_source[source_a][cid] - by_source[source_b][cid] for cid in cids],
            dtype=np.float64,
        )
        resampled = rng.choice(
            delta, size=(bootstrap_samples, delta.size), replace=True
        ).mean(axis=1)
        result.append(
            {
                "source_a": source_a,
                "source_b": source_b,
                "count": len(cids),
                "mean_delta_similarity_a_minus_b": float(delta.mean()),
                "bootstrap_ci95_low": float(np.quantile(resampled, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(resampled, 0.975)),
                "win_rate_a": float(np.mean(delta > 0)),
                "tie_rate": float(np.mean(delta == 0)),
            }
        )
    return result



