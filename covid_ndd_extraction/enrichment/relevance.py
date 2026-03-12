"""GPT-4o-based relevance filtering for biomedical image URLs."""

from __future__ import annotations

import logging
import os
import time

import pandas as pd
import requests
from http.client import RemoteDisconnected

from covid_ndd_extraction.utils.openai_client import get_openai_client

logger = logging.getLogger(__name__)

RELEVANCE_PROMPT = """\
Image URL is given. Analyze this image and assign relevance to it: "Yes" for relevant images, "No" for irrelevant images, and "Uncertain" if the relevance cannot be assessed. Follow this classification:
Relevant:
Images which:
Clearly demonstrate relationship between Covid-19 and neurodegeneration (any neurological impacts).
Don't contain lots of text (no more than 500 characters).
Don't just depict a research outline.
Don't be just graphs or represent photos derived from scientific tools (microscopic, histological images) and data visualization.
Are likely to be cartoons drawn by article authors.

Irrelevant:
Unrelated images, for example just an image of a virus particle or a sick person.
Images where correct interpretation of the data is impossible.
Images which display insights into Covid-19 OR Neurodegeneration, if one is present and the other is missing.

Uncertain:
The relevance of the image cannot be confidently determined based on the visual content.

Your answer should contain only a final decision in the following format: No/Yes/Uncertain (without dots)
Don't write anything else!"""


def check_image_url(url: str, retries: int = 3) -> bool:
    """Return True if URL is accessible and returns image content."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
                return True
            return False
        except (requests.ConnectionError, requests.Timeout, RemoteDisconnected) as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(2)
            else:
                return False
        except Exception as exc:
            logger.warning("Error checking %s: %s", url, exc)
            return False
    return False


def gpt_classify_relevance(url: str, api_key: str | None = None) -> str:
    """Ask GPT-4o whether an image is relevant to COVID-19/neurodegeneration research."""
    client = get_openai_client(api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": RELEVANCE_PROMPT},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
            max_tokens=900,
        )
        content = (response.choices[0].message.content or "").strip().rstrip(".")
        return content
    except Exception as exc:
        logger.error("Error classifying %s: %s", url, exc)
        return "Error"


def relevance_check_main(
    input_path: str,
    output_dir: str = "data/URL_relevance_analysis",
    api_key: str | None = None,
) -> None:
    """Run GPT-4o relevance filtering on an Excel file of image URLs.

    Parameters
    ----------
    input_path:
        Excel file with an ``image_url`` column.
    output_dir:
        Directory for output files.
    api_key:
        OpenAI API key.
    """
    df = pd.read_excel(input_path)
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    results = []
    for _, row in df.iterrows():
        url = row["image_url"]
        try:
            label = gpt_classify_relevance(url, api_key)
            logger.info("%s → %s", url, label)
            results.append([url, label])
        except Exception as exc:
            logger.error("Error processing %s: %s", url, exc)
            results.append([url, "Error"])

    relevance_df = pd.DataFrame(results, columns=["URL", "Relevance_GPT"])
    os.makedirs(output_dir, exist_ok=True)
    relevance_df.to_excel(os.path.join(output_dir, "Relevance_assignment_GPT_4o.xlsx"), index=False)

    relevant = relevance_df[relevance_df["Relevance_GPT"] == "Yes"].copy()
    relevant.reset_index(drop=True, inplace=True)
    relevant[["URL"]].to_excel(os.path.join(output_dir, "Relevant_URLs_only_GPT_4o.xlsx"), index=False)

    logger.info("Total URLs: %d", len(df))
    logger.info("Errors: %d", len(relevance_df[relevance_df["Relevance_GPT"] == "Error"]))
    logger.info("Relevant: %d", len(relevant))
