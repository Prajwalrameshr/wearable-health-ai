from sklearn.preprocessing import StandardScaler

from src.gmm import DEFAULT_GMM_FEATURES, run_gmm_pipeline
from src.hmm_model import run_all_hmm_modes
from src.kmeans import DEFAULT_KMEANS_FEATURES, run_kmeans_pipeline
from src.model_comparison import compare_models
from src.preprocessing import CLUSTER_FEATURE_COLUMNS, preprocess_for_modeling


CLUSTER_FEATURES = [column for column in CLUSTER_FEATURE_COLUMNS if column in DEFAULT_KMEANS_FEATURES and column in DEFAULT_GMM_FEATURES]


def main() -> None:
    print("Starting full pipeline...\n")

    outputs = preprocess_for_modeling()
    feature_df = outputs["feature_df"]
    print("Preprocessing done")

    scaler = StandardScaler()
    scaled_cluster_df = feature_df.copy()
    scaled_cluster_df[CLUSTER_FEATURES] = scaler.fit_transform(feature_df[CLUSTER_FEATURES])
    print(f"Scaling done with features: {CLUSTER_FEATURES}")

    kmeans_out = run_kmeans_pipeline(scaled_cluster_df, feature_columns=CLUSTER_FEATURES)
    print("KMeans done")

    gmm_out = run_gmm_pipeline(scaled_cluster_df, feature_columns=CLUSTER_FEATURES)
    print("GMM done")

    hmm_input = gmm_out["labeled_df"].copy()
    if "kmeans_cluster" in kmeans_out["labeled_df"].columns:
        hmm_input["kmeans_cluster"] = kmeans_out["labeled_df"]["kmeans_cluster"].astype(int).to_numpy()

    hmm_out = run_all_hmm_modes(hmm_input)
    print("HMM done")

    comparison_df = compare_models(kmeans_out, gmm_out, hmm_out)

    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))

    print("\nGMM Cluster Summary:")
    print(gmm_out["cluster_summary"].to_string(index=False))

    for model_name, result in hmm_out.items():
        print(f"\nHMM Summary ({model_name}):")
        print("Transition matrix:")
        print(result["transition_matrix"].to_string())
        print("State distribution:")
        print(result["state_distribution"].to_string())
        print(f"Transition rate: {result['transition_rate']:.4f}")


if __name__ == "__main__":
    main()
