"""Triple extraction from biomedical images using GPT-4o."""

from __future__ import annotations

import csv
import logging
import os
import re
import time
from datetime import datetime

import pandas as pd
import requests

from covid_ndd_extraction.utils.openai_client import get_openai_client
from covid_ndd_extraction.utils.parsing import (
    REFUSAL_KEYWORDS,
    parse_mechanisms_and_triples,  # noqa: F401 (re-exported)
)

__all__ = ["parse_mechanisms_and_triples", "triples_extraction_from_urls", "gpt_extract"]

logger = logging.getLogger(__name__)

PROMPT_TEXT = (
    "Describe the image (Figure/Graphical abstract) from an article on comorbidity between "
    "COVID-19 and Neurodegeneration.\n"
    "1. Name potential mechanisms (pathophysiological processes) of Covid-19's impact on the brain "
    "depicted in the image.\n"
    "2. Describe each process depicted in the image as semantic triples (subject–predicate–object).\n"
    "Example:\n"
    "Pathophysiological Process: Astrocyte_Activation\n"
    "Triples:\n"
    "SARS-CoV-2_infection|triggers|astrocyte_activation\n\n"
    "Use ONLY the information shown in the image! Follow the structure precisely and don't write "
    "anything else!\n"
    "Replace spaces in names with _ sign, make sure that words \"Pathophysiological Process:\" and "
    "\"Triples:\" are presented,\n"
    "don't use bold font and margins. Each triple must contain ONLY THREE elements separated by a "
    "| sign; four or more are not allowed!"
)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def to_raw_github(url: str) -> str:
    """Convert GitHub /blob/ URLs to raw.githubusercontent.com."""
    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$", url)
    if m:
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url


def is_url_accessible(url: str, timeout: int = 10) -> bool:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        )
    }
    try:
        trial_urls = [to_raw_github(url)]
        if trial_urls[0] != url:
            trial_urls.append(url)
        for u in trial_urls:
            r = requests.head(u, headers=headers, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                return True
            r = requests.get(u, headers=headers, stream=True, timeout=timeout)
            if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                return True
        return False
    except requests.RequestException as exc:
        logger.warning("URL access error for %s: %s", url, exc)
        return False


# ---------------------------------------------------------------------------
# GPT call
# ---------------------------------------------------------------------------


def gpt_extract(url: str, api_key: str | None = None) -> str:
    """Send an image URL to GPT-4o and return the raw text response."""
    client = get_openai_client(api_key)
    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEXT},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        max_tokens=2000,
        temperature=0.25,
        top_p=0.25,
        seed=42,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def triples_extraction_from_urls(
    input_path: str,
    output_path_base: str,
    api_key: str | None = None,
) -> None:
    """Extract semantic triples from image URLs and save to CSV/Excel.

    Parameters
    ----------
    input_path:
        Path to Excel file with image URLs.
    output_path_base:
        Output file stem (no extension); ``.csv`` and ``.xlsx`` are written.
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` env var.
    """
    out_dir = os.path.dirname(output_path_base)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    log_dir = "data/triples_output"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "triple_extraction_runlog.csv")
    with open(log_path, "w", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Image_Number", "URL", "Outcome", "Finish_Reason", "HTTP_Status", "Timestamp"])

        df = pd.read_excel(input_path)
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "image_number"})
        else:
            df["image_number"] = df.index + 1
        if "URL" not in df.columns:
            raise ValueError(f"Input file missing 'URL' column. Found: {list(df.columns)}")
        df = df.rename(columns={"URL": "url"})
        df["github_url"] = df["url"]

        parsed_data: list[list] = []

        for _, row in df.iterrows():
            image_number = row["image_number"]
            original_url = row["url"]
            github_url = to_raw_github(row["github_url"])

            outcome = "unknown"
            finish_reason = "N/A"
            http_status: int | str = "N/A"

            if not is_url_accessible(github_url):
                logger.warning("Skipping inaccessible URL: %s", github_url)
                outcome = "inaccessible"
                http_status = "Non-200 or timeout"
                log_writer.writerow([image_number, github_url, outcome, finish_reason, http_status, datetime.now().isoformat()])
                continue

            try:
                time.sleep(1.5)
                http_status = 200
                content = gpt_extract(github_url, api_key)
                finish_reason = "stop"

                if any(k in content.lower() for k in REFUSAL_KEYWORDS):
                    outcome = "refused"
                    logger.warning("GPT refused to process: %s", github_url)
                else:
                    blocks = parse_mechanisms_and_triples(content)
                    if not blocks:
                        outcome = "no_parse"
                        logger.warning("No mechanisms/triples parsed for: %s\n--- RAW ---\n%s\n-----------", github_url, content)
                    else:
                        outcome = "ok"
                        logger.info("%s\n%s", github_url, content)
                        for b in blocks:
                            mech = b["mechanism"]
                            for subj, pred, obj in b["triples"]:
                                parsed_data.append([image_number, original_url, github_url, mech, subj, pred, obj])

                log_writer.writerow([image_number, github_url, outcome, finish_reason, http_status, datetime.now().isoformat()])

            except Exception as exc:
                logger.error("Error processing %s: %s", github_url, exc)
                log_writer.writerow([image_number, github_url, "error", str(exc), http_status, datetime.now().isoformat()])

    parsed_df = pd.DataFrame(
        parsed_data,
        columns=["Image_number", "URL", "GitHub_URL", "Pathophysiological Process", "Subject", "Predicate", "Object"],
    )
    parsed_df.to_csv(f"{output_path_base}.csv", index=False)
    parsed_df.to_excel(f"{output_path_base}.xlsx", index=False)
    logger.info("Triples saved to %s.csv / .xlsx", output_path_base)
    logger.info("Run log saved to %s", log_path)
