# Source Code File: clustering_agent.py
# Student Name: Shiva Shrestha
# Date: September 16, 2026

import numpy as np
from sklearn.metrics import silhouette_score

from model.KMeans import kmeans


def evaluate_k_values(embeddings, k_values):
    """Evaluate selected values of k using clustering metrics."""
    results = []

    for k in k_values:
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
        cluster_sizes = np.bincount(labels, minlength=k)

        results.append({
            "k": k,
            "labels": labels,
            "centroids": centroids,
            "inertia": inertia,
            "silhouette": silhouette,
            "cluster_sizes": cluster_sizes,
        })

        print(
            f"k={k}: "
            f"inertia={inertia:.4f}, "
            f"silhouette={silhouette:.4f}, "
            f"smallest cluster={cluster_sizes.min()}, "
            f"largest cluster={cluster_sizes.max()}"
        )

    return results


def evaluate_clusters(embeddings, min_k=2, max_k=10):
    """Evaluate every value of k within an inclusive range."""
    return evaluate_k_values(
        embeddings,
        range(min_k, max_k + 1),
    )


def find_representatives(chunks, embeddings, labels, centroids=None):
    """Find the chunk closest to each cluster centroid."""
    representatives = []
    cluster_count = (
        len(centroids)
        if centroids is not None
        else int(labels.max()) + 1
    )

    for cluster_id in range(cluster_count):
        indices = np.where(labels == cluster_id)[0]

        if len(indices) == 0:
            continue

        distances = np.sqrt(
            np.sum(
                (
                    embeddings[indices]
                    - centroids[cluster_id]
                ) ** 2,
                axis=1,
            )
        )
        medoid_index = int(indices[np.argmin(distances)])

        representatives.append({
            "cluster": cluster_id,
            "cluster_size": len(indices),
            "medoid_index": medoid_index,
            "chunk": chunks[medoid_index],
        })

    return representatives
