"""Random subset selector for CBM evaluation datasets."""

from __future__ import annotations

import logging
import os
import random

import pandas as pd

logger = logging.getLogger(__name__)


def select_url_subset(
    cbm_metadata_path: str = "data/CBM_data/Data_CBM_with_GitHub_URLs.xlsx",
    cbm_triples_path: str = "data/CBM_data/Triples_CBM_Gold_Standard.xlsx",
    output_dir: str = "data/prompt_engineering/cbm_files",
    n: int = 50,
    seed: int | None = None,
) -> None:
    """Randomly select *n* image URLs from the CBM dataset and save the subset.

    Parameters
    ----------
    cbm_metadata_path:
        Path to the CBM metadata Excel file (must have ``URL`` and ``Image_number`` columns).
    cbm_triples_path:
        Path to the full CBM gold-standard triples Excel file.
    output_dir:
        Directory for the two output files.
    n:
        Number of unique URLs to sample.
    seed:
        Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    df_metadata = pd.read_excel(cbm_metadata_path)
    unique_urls = df_metadata["URL"].dropna().unique().tolist()
    if len(unique_urls) < n:
        logger.warning("Only %d unique URLs available; sampling all.", len(unique_urls))
        n = len(unique_urls)
    sampled_urls = random.sample(unique_urls, n)

    df_subset_urls = (
        df_metadata[df_metadata["URL"].isin(sampled_urls)][["Image_number", "URL", "GitHub_URL"]]
        .drop_duplicates()
    )
    os.makedirs(output_dir, exist_ok=True)
    urls_out = os.path.join(output_dir, "CBM_subset_50_URLs.xlsx")
    df_subset_urls.to_excel(urls_out, index=False)
    logger.info("Saved %d URL subset to %s", len(df_subset_urls), urls_out)

    df_triples = pd.read_excel(cbm_triples_path)
    selected_images = df_subset_urls["Image_number"].unique().tolist()
    df_subset_triples = df_triples[df_triples["Image_number"].isin(selected_images)]
    triples_out = os.path.join(output_dir, "CBM_subset_50_URL_triples.xlsx")
    df_subset_triples.to_excel(triples_out, index=False)
    logger.info("Saved corresponding triples to %s", triples_out)
