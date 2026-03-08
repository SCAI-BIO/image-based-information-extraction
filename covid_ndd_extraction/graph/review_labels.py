"""Neo4j node label review and correction using GPT-4o."""

from __future__ import annotations

import logging
import time

from covid_ndd_extraction.config import settings
from covid_ndd_extraction.utils.openai_client import get_openai_client

logger = logging.getLogger(__name__)

CONTROLLED_VOCAB = [
    "Anatomical_Structure", "Biological_Process", "Cell", "Cell_Phenotype",
    "Chemical", "Disease", "Gene", "Phenotype", "Protein", "Pathway",
]


def review_label(entity: str, current_label: str, api_key: str | None = None) -> str:
    """Ask GPT-4o to verify/correct a node's semantic label."""
    client = get_openai_client(api_key)
    vocab_str = ", ".join(CONTROLLED_VOCAB)
    prompt = (
        "You are a biomedical ontology expert. Your task is to verify or correct the label for a biological entity.\n\n"
        f"Entity: \"{entity}\"\n"
        f"Current Label: \"{current_label}\"\n\n"
        "Choose the single most appropriate label from the following controlled vocabulary:\n"
        f"{vocab_str}\n\n"
        "Rules:\n"
        "- Return only ONE label as plain text (e.g., Gene, Disease).\n"
        "- Do not include punctuation, quotes, extra words, or explanations.\n"
        "- Return the current label exactly as-is if it is already correct.\n"
        "- If none apply, return a single new label that best describes the entity.\n"
        "- If the current label is Unknown, choose the most suitable label; avoid 'Unknown'.\n\n"
        "Output: The label string only."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You classify biological entities."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        new_label = (resp.choices[0].message.content or "").strip()
        return new_label or current_label
    except Exception as exc:
        logger.error("GPT API error for entity='%s': %s", entity, exc)
        return current_label


def run_label_review(
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    api_key: str | None = None,
) -> None:
    """Connect to Neo4j, fetch all named nodes, review labels via GPT-4o, and update."""
    from neo4j import GraphDatabase, basic_auth  # optional dep

    uri = uri or settings.neo4j_uri
    user = user or settings.neo4j_user
    password = password or settings.neo4j_password

    driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
    updated = unchanged = 0

    try:
        with driver.session() as session:
            results = list(session.run(
                "MATCH (n) WHERE n.name IS NOT NULL RETURN id(n) AS id, n.name AS entity, labels(n) AS labels"
            ))
            total = len(results)
            logger.info("Fetched %d nodes.", total)

            for i, rec in enumerate(results, start=1):
                nid = rec["id"]
                ent = rec["entity"]
                labs = rec["labels"]
                current = next((L for L in labs if L in CONTROLLED_VOCAB), "")
                new_label = review_label(ent, current, api_key)

                if new_label != current:
                    remove_current = f"REMOVE n:`{current}`" if current else ""
                    remove_unknown = "REMOVE n:`Unknown`" if "Unknown" in labs else ""
                    session.run(
                        f"MATCH (n) WHERE id(n) = $id {remove_unknown} {remove_current} SET n:`{new_label}`",
                        id=nid,
                    )
                    updated += 1
                    logger.info("[UPDATED] '%s': %s → %s", ent, current or "<none>", new_label)
                else:
                    unchanged += 1
                    logger.info("[UNCHANGED] '%s' remains '%s'", ent, current or "<none>")

                if i % 50 == 0 or i == total:
                    logger.info("Progress: %d/%d", i, total)
                time.sleep(0.2)
    finally:
        driver.close()

    logger.info("Summary: UPDATED=%d, UNCHANGED=%d, TOTAL=%d", updated, unchanged, updated + unchanged)
