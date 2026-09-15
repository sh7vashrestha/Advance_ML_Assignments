import numpy as np
from sklearn.metrics import silhouette_score

from model.KMeans import kmeans


def evaluate_clusters(embeddings, min_k=2, max_k=10):
    results = []

    for k in range(min_k, max_k + 1):
        labels, centroids = kmeans(
            embeddings,
            k=k
        )

        # Sum of squared distances to assigned centroids
        inertia = np.sum(
            (embeddings - centroids[labels]) ** 2
        )

        silhouette = silhouette_score(
            embeddings,
            labels
        )

        results.append({
            "k": k,
            "labels": labels,
            "centroids": centroids,
            "inertia": inertia,
            "silhouette": silhouette,
        })

        print(
            f"k={k}: "
            f"inertia={inertia:.4f}, "
            f"silhouette={silhouette:.4f}"
        )

    return results