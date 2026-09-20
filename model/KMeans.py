# Source Code File: KMeans.py
# Student Name: Shiva Shrestha
# Date: September 15, 2026

import numpy as np


def kmeans(data, k, max_iterations=100):
    rng = np.random.default_rng(30)
    indices = rng.choice(len(data), k, replace=False)
    centroids = data[indices].copy()

    for _ in range(max_iterations):
        labels = []

        for point in data:
            distances = []
            for centroid in centroids:
                distance = np.sqrt(np.sum((point - centroid) ** 2))
                distances.append(distance)

            labels.append(np.argmin(distances))

        labels = np.array(labels)

        new_centroids = []

        for cluster in range(k):
            cluster_points = data[labels == cluster]

            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[cluster])

        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            break

        centroids = new_centroids
    return labels, centroids
