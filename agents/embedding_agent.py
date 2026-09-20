# Source Code File: embedding_agent.py
# Student Name: Shiva Shrestha
# Date: September 16, 2026

import numpy as np
import ollama


def create_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]

    response = ollama.embed(
        model="embeddinggemma",
        input=texts
    )

    return np.array(
        response["embeddings"],
        dtype=float
    )
