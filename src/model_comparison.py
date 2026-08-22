from __future__ import annotations

from typing import Any

import pandas as pd


def compare_models(
    kmeans_out: dict[str, Any],
    gmm_out: dict[str, Any],
    hmm_out: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Compare clustering and temporal models with a compact evaluation view."""
    rows: list[dict[str, Any]] = [
        {
            "model": "KMeans",
            "silhouette": kmeans_out["metrics"].get("silhouette_score"),
            "davies_bouldin": kmeans_out["metrics"].get("davies_bouldin_index"),
            "transition_rate": None,
            "risk_monotonicity": kmeans_out["metrics"].get("risk_monotonicity"),
        },
        {
            "model": "GMM",
            "silhouette": gmm_out["metrics"].get("silhouette"),
            "davies_bouldin": gmm_out["metrics"].get("davies_bouldin"),
            "transition_rate": None,
            "risk_monotonicity": gmm_out["metrics"].get("risk_monotonicity"),
        },
    ]

    for mode_name, result in hmm_out.items():
        rows.append(
            {
                "model": f"HMM ({mode_name})",
                "silhouette": None,
                "davies_bouldin": None,
                "transition_rate": result.get("transition_rate"),
                "risk_monotonicity": None,
            }
        )

    comparison = pd.DataFrame(rows)
    return comparison.round(4)
