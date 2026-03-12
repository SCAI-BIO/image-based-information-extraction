"""Triple extraction from full-text biomedical articles using GPT-4o."""

from __future__ import annotations

import json
import logging
import os

import pandas as pd

from covid_ndd_extraction.utils.openai_client import get_openai_client

logger = logging.getLogger(__name__)

FULLTEXT_PROMPT = """Describe the following scientific paragraph from an article on comorbidity between COVID-19 and Neurodegeneration.
1. Name potential mechanisms (pathophysiological processes) of Covid-19's impact on the brain described in the text.
2. Describe each process described in the text as semantic triples (subject–predicate–object).
Example:
Pathophysiological Process: Astrocyte_Activation
Triples:
SARS-CoV-2_infection|triggers|astrocyte_activation

If the paragraph does not contain relevant biological content (e.g. acknowledgments, funding, conflicts of interest, publisher notes, metadata, disclaimers, or any non-scientific content), or if no such mechanisms or valid triples are present in the paragraph, return exactly:
Pathophysiological Process: Not_found
Triples:
Not_found|Not_found|Not_found

Use ONLY the information provided in the text! Follow the structure precisely and don't write anything else! Replace spaces in names with _ sign, make sure that words "Pathophysiology Process:" and "Triples:" are presented, don't use bold font and margins. Each triple must contain ONLY THREE elements separated by a | sign, four and more are not allowed!"""


def gpt_extract_from_text(paragraph: str, api_key: str | None = None) -> str:
    """Send a text paragraph to GPT-4o for triple extraction."""
    client = get_openai_client(api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FULLTEXT_PROMPT},
                    {"type": "text", "text": paragraph},
                ],
            }
        ],
        max_tokens=2000,
        temperature=0.0,
        top_p=0.0,
    )
    return response.choices[0].message.content or ""


def triples_extraction_from_articles(
    input_path: str,
    output_dir: str,
    api_key: str | None = None,
) -> None:
    """Extract triples from full-text articles and save to CSV/Excel.

    Parameters
    ----------
    input_path:
        Path to JSON file with article data.
    output_dir:
        Directory for output files.
    api_key:
        OpenAI API key.
    """
    with open(input_path, encoding="utf-8") as f:
        articles = json.load(f)

    parsed_data: list[list] = []

    for article_id, article in articles.items():
        title = article.get("title", "")
        paragraphs = article.get("paragraphs", [])

        for paragraph in paragraphs:
            try:
                content = gpt_extract_from_text(paragraph, api_key)
                logger.info("Processed paragraph from article %s | Title: %s\n%s", article_id, title, content)

                not_found_marker = "Pathophysiological Process: Not_found\nTriples:\nNot_found|Not_found|Not_found"
                if content.strip() == not_found_marker:
                    parsed_data.append([article_id, title, paragraph, "Not_found", "Not_found", "Not_found", "Not_found"])
                    continue

                mechanisms = content.strip().split("Pathophysiological Process: ")
                for mechanism_block in mechanisms[1:]:
                    lines = mechanism_block.strip().split("\n")
                    mechanism_name = lines[0].strip()
                    triples = lines[2:]  # skip "Triples:" line

                    malformed = any(len(t.strip().split("|")) != 3 for t in triples if t.strip())
                    if malformed:
                        parsed_data.append([article_id, title, paragraph, "Not_found", "Not_found", "Not_found", "Not_found"])
                        break
                    for triple in triples:
                        if not triple.strip():
                            continue
                        subject, predicate, obj = triple.strip().split("|")
                        parsed_data.append([article_id, title, paragraph, mechanism_name, subject, predicate, obj])

            except Exception as exc:
                logger.error("Error processing article %s: %s", article_id, exc)

    parsed_df = pd.DataFrame(
        parsed_data,
        columns=["PMID", "Title", "Paragraph", "Pathophysiological Process", "Subject", "Predicate", "Object"],
    )
    parsed_df = parsed_df[parsed_df["Pathophysiological Process"] != "Not_found"].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    parsed_df.to_csv(os.path.join(output_dir, "Triples_Full_Text_GPT_for_comp.csv"), index=False)
    parsed_df.to_excel(os.path.join(output_dir, "Triples_Full_Text_GPT_for_comp.xlsx"), index=False)
    logger.info("Triples saved to %s", output_dir)
