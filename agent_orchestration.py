# Source Code File: agent_orchestration.py
# Student Name: Shiva Shrestha
# Date: September 15, 2026

from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from sklearn.metrics import silhouette_score

from agents.clustering_agent import find_representatives
from agents.document_agent import process_documents
from agents.embedding_agent import create_embeddings
from agents.summary_agent import create_final_summary
from agents.topic_agent import create_topics
from model.KMeans import kmeans


class SummaryState(TypedDict, total=False):
    """Shared information passed between the agents."""

    collection_name: str
    pdf_folder: str
    k: int
    chunk_size: int
    overlap: int
    chunks: list[dict[str, Any]]
    embeddings: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray
    cluster_sizes: np.ndarray
    inertia: float
    silhouette: float
    medoids: list[dict[str, Any]]
    topics: list[dict[str, Any]]
    final_summary: str


def document_node(state: SummaryState):
    """Load and split the documents unless chunks were supplied."""
    if "chunks" in state:
        return {}

    return {
        "chunks": process_documents(
            state["pdf_folder"],
            chunk_size=state.get("chunk_size", 1000),
            overlap=state.get("overlap", 200),
        )
    }


def embedding_node(state: SummaryState):
    """Create embeddings unless they were supplied."""
    if "embeddings" in state:
        return {}

    return {
        "embeddings": create_embeddings(state["chunks"])
    }


def clustering_node(state: SummaryState):
    """Cluster the embeddings and calculate evaluation metrics."""
    labels, centroids = kmeans(
        state["embeddings"],
        k=state["k"],
    )

    inertia = np.sum(
        (state["embeddings"] - centroids[labels]) ** 2
    )
    cluster_sizes = np.bincount(
        labels,
        minlength=state["k"],
    )

    return {
        "labels": labels,
        "centroids": centroids,
        "cluster_sizes": cluster_sizes,
        "inertia": float(inertia),
        "silhouette": float(
            silhouette_score(state["embeddings"], labels)
        ),
    }


def medoid_node(state: SummaryState):
    """Select the chunk closest to each cluster centroid."""
    return {
        "medoids": find_representatives(
            state["chunks"],
            state["embeddings"],
            state["labels"],
            state["centroids"],
        )
    }


def topic_node(state: SummaryState):
    """Generate a topic label and summary for every medoid."""
    return {
        "topics": create_topics(state["medoids"])
    }


def summary_node(state: SummaryState):
    """Combine the topic components into the final summary."""
    return {
        "final_summary": create_final_summary(state["topics"])
    }


def build_graph():
    """Build and compile the sequential agent graph."""
    builder = StateGraph(SummaryState)

    builder.add_node("documents", document_node)
    builder.add_node("embeddings", embedding_node)
    builder.add_node("clustering", clustering_node)
    builder.add_node("medoids", medoid_node)
    builder.add_node("topics", topic_node)
    builder.add_node("summary", summary_node)

    builder.add_edge(START, "documents")
    builder.add_edge("documents", "embeddings")
    builder.add_edge("embeddings", "clustering")
    builder.add_edge("clustering", "medoids")
    builder.add_edge("medoids", "topics")
    builder.add_edge("topics", "summary")
    builder.add_edge("summary", END)

    return builder.compile()


agent_graph = build_graph()


def run_collection(
    collection_name,
    k,
    pdf_folder=None,
    chunks=None,
    embeddings=None,
    chunk_size=1000,
    overlap=200,
):
    """Run the graph for one document collection and one value of k."""
    initial_state = {
        "collection_name": collection_name,
        "k": k,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }

    if pdf_folder is not None:
        initial_state["pdf_folder"] = pdf_folder
    if chunks is not None:
        initial_state["chunks"] = chunks
    if embeddings is not None:
        initial_state["embeddings"] = embeddings

    return agent_graph.invoke(initial_state)


def run_k_study(
    collection_name,
    chunks,
    embeddings,
    k_values=(3, 10, 30),
):
    """Run the complete agent graph for several values of k."""
    reports = {}

    for k in k_values:
        print(f"\nRunning {collection_name} with k={k}...")

        result = run_collection(
            collection_name=collection_name,
            k=k,
            chunks=chunks,
            embeddings=embeddings,
        )

        reports[k] = {
            "inertia": result["inertia"],
            "silhouette": result["silhouette"],
            "cluster_sizes": result["cluster_sizes"],
            "medoids": result["medoids"],
            "topics": result["topics"],
            "final_summary": result["final_summary"],
        }

        print(f"\n{'=' * 60}")
        print(f"{collection_name}: k={k}")
        print("=" * 60)
        print(f"Inertia: {result['inertia']:.4f}")
        print(f"Silhouette: {result['silhouette']:.4f}")

        for medoid, topic in zip(
            result["medoids"],
            result["topics"],
        ):
            chunk = medoid["chunk"]
            excerpt = chunk["text"][:200].replace("\n", " ")

            print(
                f"\nCluster {medoid['cluster']}"
                f" - size {medoid['cluster_size']}"
            )
            print(
                f"Medoid: {chunk['source']}, "
                f"page {chunk['page']}"
            )
            print(f"Excerpt: {excerpt}...")
            print(f"Topic: {topic['label']}")
            print(f"Topic summary: {topic['summary']}")

        print("\nFINAL SUMMARY")
        print(result["final_summary"])

    return reports


def print_assignment_results(
    reports,
    collection_name,
    k_values=(3, 10, 30),
):
    """Print the required assignment results for each value of k."""
    for k in k_values:
        report = reports[k]

        print("\n" + "=" * 70)
        print(f"{collection_name.upper()}: k = {k}")
        print("=" * 70)
        print(f"Inertia: {report['inertia']:.4f}")
        print(
            f"Silhouette score: "
            f"{report['silhouette']:.4f}"
        )

        print("\n1. CLUSTER SIZES")
        for cluster_id, size in enumerate(
            report["cluster_sizes"]
        ):
            print(f"Cluster {cluster_id}: {size} chunks")

        topics_by_cluster = {
            topic["cluster"]: topic
            for topic in report["topics"]
        }

        print("\n2. CLUSTER MEDOIDS")
        print("3. GENERATED TOPIC LABELS")

        for medoid in report["medoids"]:
            cluster_id = medoid["cluster"]
            chunk = medoid["chunk"]
            topic = topics_by_cluster[cluster_id]

            print("\n" + "-" * 60)
            print(f"Cluster: {cluster_id}")
            print(f"Cluster size: {medoid['cluster_size']}")
            print(f"Medoid source: {chunk['source']}")
            print(f"Medoid page: {chunk['page']}")
            print(f"Medoid chunk: {chunk['chunk']}")
            print(f"Topic label: {topic['label']}")
            print(f"Topic summary: {topic['summary']}")
            print("Medoid text:")
            print(chunk["text"])

        print("\n4. FINAL SUMMARY")
        print("-" * 60)
        print(report["final_summary"])
