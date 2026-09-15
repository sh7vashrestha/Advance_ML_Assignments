import numpy as np


def kmeans(data, k, max_iterations=100):
    rng = np.random.default_rng(30)
    indices = rng.choice(len(data), k, replace=False)
    centroids = data[indices].copy()

    for _ in range(max_iterations):
        differences = data[:, np.newaxis] - centroids
        distances = np.sqrt(
            np.sum(differences ** 2, axis=2)
        )

        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            data[labels == cluster].mean(axis=0)
            if np.any(labels == cluster)
            else centroids[cluster]
            for cluster in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    return labels, centroids