"""Semantic triple comparison using BioBERT + Hungarian matching vs a gold standard."""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from covid_ndd_extraction.utils.embeddings import (
    batch_embed,
    build_sim_matrix,
    format_triple,
    load_biobert,
)
from covid_ndd_extraction.utils.matching import hungarian_match

logger = logging.getLogger(__name__)


def group_triples(df: pd.DataFrame) -> dict[str, list[tuple[str, str, str]]]:
    """Group (Subject, Predicate, Object) tuples by Image_number."""
    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for _, row in df.iterrows():
        if all(pd.notna([row["Subject"], row["Predicate"], row["Object"]])):
            grouped.setdefault(row["Image_number"], []).append(
                (row["Subject"], row["Predicate"], row["Object"])
            )
    return grouped


def evaluate_images(
    df_gold: pd.DataFrame,
    df_eval: pd.DataFrame,
    threshold: float = 0.85,
    output_dir: str = "data/gold_standard_comparison",
    figures_dir: str = "data/figures_output",
) -> dict[str, float]:
    """Compare GPT triples to gold standard using BioBERT + Hungarian matching.

    Returns
    -------
    dict with keys ``precision``, ``recall``, ``f1``.
    """
    tokenizer, model = load_biobert()
    gold_dict = group_triples(df_gold)
    eval_dict = group_triples(df_eval)

    common_images = sorted(
        set(gold_dict) & set(eval_dict),
        key=lambda x: int(x.split("_")[-1]),
    )
    total_TP = total_FP = total_FN = 0
    comparison_records: list[dict] = []

    for image_id in common_images:
        logger.info("Comparing triples for image: %s", image_id)
        gold_triples = gold_dict[image_id]
        eval_triples = eval_dict[image_id]

        gold_sentences = [format_triple(*t) for t in gold_triples]
        eval_sentences = [format_triple(*t) for t in eval_triples]

        emb_gold = batch_embed(gold_sentences, tokenizer, model)
        emb_eval = batch_embed(eval_sentences, tokenizer, model)

        sim_matrix = build_sim_matrix(emb_eval, emb_gold)

        for i, es in enumerate(eval_sentences):
            for j, gs in enumerate(gold_sentences):
                comparison_records.append({
                    "Image": image_id,
                    "GPT_Triple": es,
                    "CBM_Triple": gs,
                    "Similarity": sim_matrix[i, j],
                })

        row_ind, col_ind = hungarian_match(sim_matrix)
        matched_gpt: set[int] = set()
        matched_cbm: set[int] = set()
        for i, j in zip(row_ind, col_ind):
            if sim_matrix[i, j] >= threshold:
                matched_gpt.add(i)
                matched_cbm.add(j)

        TP = len(matched_gpt)
        FP = len(eval_triples) - TP
        FN = len(gold_triples) - len(matched_cbm)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        logger.info("Image %s — TP=%d, FP=%d, FN=%d", image_id, TP, FP, FN)

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    logger.info("Overall — TP=%d FP=%d FN=%d | P=%.3f R=%.3f F1=%.3f",
                total_TP, total_FP, total_FN, precision, recall, f1)

    # Histogram
    os.makedirs(figures_dir, exist_ok=True)
    plt.figure(figsize=(6, 4), dpi=600)
    plt.hist(
        [r["Similarity"] for r in comparison_records],
        bins=np.linspace(0.5, 1.0, 11),
        edgecolor="black",
        color="teal",
    )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "Fig_FullTriple_Semantic.tiff"), dpi=600)
    plt.close()

    # Save log
    os.makedirs(output_dir, exist_ok=True)
    threshold_str = str(int(threshold * 100))
    log_path = os.path.join(output_dir, f"CBM_comparison_Report_Threshold_{threshold_str}.xlsx")
    pd.DataFrame(comparison_records).to_excel(log_path, index=False)
    logger.info("Comparison log saved to %s", log_path)

    return {"precision": precision, "recall": recall, "f1": f1}
