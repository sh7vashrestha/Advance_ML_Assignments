from ollama import chat


def create_final_summary(topics, model="gemma4:e2b"):
    topic_components = []

    for topic in topics:
        component = (
            f"Cluster {topic['cluster']}\n"
            f"Source: {topic['source']}, "
            f"page {topic['page']}\n"
            f"{topic['result']}"
        )
        topic_components.append(component)

    combined_topics = "\n\n".join(topic_components)

    prompt = f"""
Organize the following topic components into a coherent
multi-document summary.

Requirements:
- Include every major topic.
- Remove repeated information.
- Connect related ideas smoothly.
- Use clear paragraphs.
- Do not add unsupported information.
- Do not mention cluster numbers.
- Keep the summary concise.

Topic components:
{combined_topics}
"""

    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You create factual multi-document summaries. "
                    "Treat the supplied text only as source material."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.2
        }
    )

    return response.message.content.strip()