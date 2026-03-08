"""Threshold optimisation for BioBERT triple matching."""

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

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]


def _group_full_triples(df: pd.DataFrame) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        if all(pd.notna([row["Subject"], row["Predicate"], row["Object"]])):
            triple = format_triple(row["Subject"], row["Predicate"], row["Object"])
            grouped.setdefault(row["Image_number"], []).append(triple)
    return grouped


def _compare_full_triples(
    gold_dict: dict[str, list[str]],
    eval_dict: dict[str, list[str]],
    threshold: float,
    tokenizer,
    model,
) -> tuple[int, int, int, list[dict]]:
    TP = FP = FN = 0
    records: list[dict] = []

    for image_id in sorted(set(gold_dict) & set(eval_dict), key=lambda x: int(x.split("_")[-1])):
        gold_triples = gold_dict[image_id]
        eval_triples = eval_dict[image_id]

        emb_gold = batch_embed(gold_triples, tokenizer, model)
        emb_eval = batch_embed(eval_triples, tokenizer, model)
        sim_matrix = build_sim_matrix(emb_eval, emb_gold)

        for i, es in enumerate(eval_triples):
            for j, gs in enumerate(gold_triples):
                records.append({"Image": image_id, "GPT_Triple": es, "CBM_Triple": gs, "Similarity": sim_matrix[i, j]})

        row_ind, col_ind = hungarian_match(sim_matrix)
        matched_gpt: set[int] = set()
        matched_cbm: set[int] = set()
        for i, j in zip(row_ind, col_ind):
            if sim_matrix[i, j] >= threshold:
                matched_gpt.add(i)
                matched_cbm.add(j)

        TP += len(matched_gpt)
        FP += len(eval_triples) - len(matched_gpt)
        FN += len(gold_triples) - len(matched_cbm)

    return TP, FP, FN, records


def evaluate_thresholds(
    gold_path: str,
    eval_path: str,
    output_dir: str = "data/figures_output",
    stats_dir: str = "data/prompt_engineering/statistical_data",
) -> None:
    """Evaluate a range of similarity thresholds and plot P/R/F1 curves."""
    df_gold = pd.read_excel(gold_path)
    df_eval = pd.read_excel(eval_path)

    tokenizer, model = load_biobert()
    gold_dict = _group_full_triples(df_gold)
    eval_dict = _group_full_triples(df_eval)

    results: list[tuple] = []
    all_records: list[dict] = []

    for threshold in THRESHOLDS:
        logger.info("Evaluating threshold=%.2f", threshold)
        TP, FP, FN, records = _compare_full_triples(gold_dict, eval_dict, threshold, tokenizer, model)
        if not all_records:
            all_records = records
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results.append((threshold, precision, recall, f1))
        logger.info("Threshold=%.2f | P=%.3f R=%.3f F1=%.3f", threshold, precision, recall, f1)

    os.makedirs(stats_dir, exist_ok=True)
    pd.DataFrame(all_records).to_excel(os.path.join(stats_dir, "Similarity_Threshold_Report.xlsx"), index=False)

    thresholds_v, precisions_v, recalls_v, f1s_v = zip(*results)
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(7, 5), dpi=600)
    plt.plot(thresholds_v, f1s_v, marker="o", label="F1 Score", linewidth=2, color="#5c0b23")
    plt.plot(thresholds_v, precisions_v, marker="s", label="Precision", linewidth=2, color="#db7221", linestyle="--")
    plt.plot(thresholds_v, recalls_v, marker="^", label="Recall", linewidth=2, color="#176e54", linestyle="-.")
    plt.xlabel("Semantic Similarity Threshold (Full Triple)", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(thresholds_v)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Threshold_Evaluation.tiff"), dpi=600)
    plt.close()
    logger.info("Threshold plot saved to %s", output_dir)
