# Source Code File: topic_agent.py
# Student Name: Shiva Shrestha
# Date: September 16, 2026

from ollama import chat


def create_topics(representatives, model="gemma4:e2b"):
    topics = []

    for representative in representatives:
        cluster_id = representative["cluster"]
        chunk = representative["chunk"]

        prompt = f"""
Analyze the following representative document chunk.

Return exactly:
Topic: <short topic name>
Summary: <one concise sentence>

Text:
{chunk["text"]}
"""

        response = chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.2
            }
        )

        result = response.message.content.strip()
        topic_label = "Untitled topic"
        topic_summary = result

        for line in result.splitlines():
            if line.lower().startswith("topic:"):
                topic_label = line.split(":", 1)[1].strip()
            elif line.lower().startswith("summary:"):
                topic_summary = line.split(":", 1)[1].strip()

        topics.append({
            "cluster": cluster_id,
            "cluster_size": representative["cluster_size"],
            "medoid_index": representative["medoid_index"],
            "source": chunk["source"],
            "page": chunk["page"],
            "label": topic_label,
            "summary": topic_summary,
            "result": result,
        })

    return topics
