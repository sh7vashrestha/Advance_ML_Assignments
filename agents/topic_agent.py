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

        topics.append({
            "cluster": cluster_id,
            "source": chunk["source"],
            "page": chunk["page"],
            "result": response.message.content.strip()
        })

    return topics