"""GPT-4o fallback categorisation for triples labelled 'Uncategorized' by BERT."""

from __future__ import annotations

import logging
import os

import pandas as pd

from covid_ndd_extraction.utils.openai_client import get_openai_client

logger = logging.getLogger(__name__)

CATEGORIES = [
    "Viral Entry and Neuroinvasion",
    "Immune and Inflammatory Response",
    "Neurodegenerative Mechanisms",
    "Vascular Effects",
    "Psychological and Neurological Symptoms",
    "Systemic Cross-Organ Effects",
]

PROMPT_TEMPLATE = """\
You are a biomedical expert. Categorize the following pathophysiological process into exactly one of the six predefined mechanistic categories listed below. Respond only with the category name — do not explain your reasoning.

Available categories:
{categories}

If the input does not clearly fit any category, return: Uncategorized

Process: "{process}"

Your answer:
"""


def gpt_categorize(process_text: str, api_key: str | None = None) -> str:
    """Ask GPT-4o to assign a category to a pathophysiological process."""
    client = get_openai_client(api_key)
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(CATEGORIES))
    prompt = PROMPT_TEMPLATE.format(categories=numbered, process=process_text)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=0.0,
            max_tokens=50,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Error categorising '%s': %s", process_text, exc)
        return "GPT_Error"


def categorize_uncategorized(
    input_file: str,
    output_file: str,
    api_key: str | None = None,
) -> None:
    """Fill 'Uncategorized' entries with GPT-4o labels and write consolidated output.

    Parameters
    ----------
    input_file:
        Path to XLSX with a ``Category`` column (output of BERT categorisation).
    output_file:
        Path stem for output ``.xlsx`` and ``.csv`` files.
    api_key:
        OpenAI API key.
    """
    df = pd.read_excel(input_file)
    df["Category_GPT"] = ""

    for idx, row in df.iterrows():
        if row["Category"] == "Uncategorized":
            pp_text = str(row["Pathophysiological Process"])
            gpt_cat = gpt_categorize(pp_text, api_key)
            df.at[idx, "Category_GPT"] = gpt_cat
            logger.info("'%s' → %s", pp_text, gpt_cat)

    df["Final_Category"] = df["Category"]
    df.loc[df["Final_Category"] == "Uncategorized", "Final_Category"] = df["Category_GPT"]

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df.to_csv(output_file + ".csv", index=False)
    df.to_excel(output_file + ".xlsx", index=False)
    logger.info("Saved to %s", output_file)
