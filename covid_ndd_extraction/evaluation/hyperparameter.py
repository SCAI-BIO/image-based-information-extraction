"""Hyperparameter assessment for GPT triple extraction (temperature / top_p)."""

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
    normalize,
)
from covid_ndd_extraction.utils.matching import hungarian_match

logger = logging.getLogger(__name__)


def _group_full_triples(df: pd.DataFrame) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        if all(pd.notna([row["Subject"], row["Predicate"], row["Object"]])):
            triple = f"{normalize(row['Subject'])} {normalize(row['Predicate'])} {normalize(row['Object'])}"
            grouped.setdefault(row["Image_number"], []).append(triple)
    return grouped


def _compare_triples(
    gold_dict: dict[str, list[str]],
    eval_dict: dict[str, list[str]],
    threshold: float,
    tokenizer,
    model,
) -> tuple[int, int, int]:
    TP = FP = FN = 0
    for image_id in sorted(set(gold_dict) & set(eval_dict), key=lambda x: int(x.split("_")[-1])):
        gold_triples = gold_dict[image_id]
        pred_triples = eval_dict[image_id]

        emb_gold = batch_embed(gold_triples, tokenizer, model)
        emb_pred = batch_embed(pred_triples, tokenizer, model)
        sim_matrix = build_sim_matrix(emb_pred, emb_gold)
        row_ind, col_ind = hungarian_match(sim_matrix)

        matches = sum(1 for i, j in zip(row_ind, col_ind) if sim_matrix[i, j] >= threshold)
        TP += matches
        FP += len(pred_triples) - matches
        FN += len(gold_triples) - matches
    return TP, FP, FN


def _evaluate_setting(
    setting_name: str,
    gold_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    threshold: float,
    tokenizer,
    model,
) -> tuple:
    gold_dict = _group_full_triples(gold_df)
    eval_dict = _group_full_triples(eval_df)
    TP, FP, FN = _compare_triples(gold_dict, eval_dict, threshold, tokenizer, model)
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    logger.info("%s — P=%.3f R=%.3f F1=%.3f", setting_name, precision, recall, f1)
    return setting_name, precision, recall, f1, TP, FP, FN


def run_hyperparameter_assessment(
    gold_path: str,
    hyperparam_paths: dict[str, str],
    threshold: float = 0.85,
    output_stats: str = "data/prompt_engineering/statistical_data/Hyperparameter_Assessment.xlsx",
    output_figures_dir: str = "data/figures_output",
) -> None:
    """Compare GPT triples under different temperature/top_p settings vs gold standard.

    Parameters
    ----------
    gold_path:
        Path to the gold-standard Excel file.
    hyperparam_paths:
        Mapping of setting label → path to GPT triples Excel file.
    threshold:
        Cosine-similarity threshold for a match.
    output_stats:
        Output path for the Excel stats table.
    output_figures_dir:
        Directory where the bar-chart figure is saved.
    """
    tokenizer, model = load_biobert()
    df_gold = pd.read_excel(gold_path)
    results = []

    for setting, path in hyperparam_paths.items():
        df_eval = pd.read_excel(path)
        result = _evaluate_setting(setting, df_gold, df_eval, threshold, tokenizer, model)
        results.append(result)

    df_stats = pd.DataFrame(results, columns=["Setting", "Precision", "Recall", "F1 Score", "TP", "FP", "FN"])
    os.makedirs(os.path.dirname(output_stats), exist_ok=True)
    df_stats.to_excel(output_stats, index=False)

    x = np.arange(len(df_stats))
    width = 0.25
    plt.figure(figsize=(8, 5), dpi=600)
    plt.bar(x + width, df_stats["F1 Score"], width, label="F1 Score", color="#5c0b23")
    plt.bar(x - width, df_stats["Precision"], width, label="Precision", color="#db7221")
    plt.bar(x, df_stats["Recall"], width, label="Recall", color="#176e54")
    plt.xticks(x, df_stats["Setting"], rotation=30, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(output_figures_dir, exist_ok=True)
    plt.savefig(os.path.join(output_figures_dir, "Hyperparameter_Assessment.tiff"), dpi=600)
    plt.savefig(os.path.join(output_figures_dir, "Hyperparameter_Assessment.png"), dpi=600)
    plt.close()
    logger.info("Hyperparameter assessment complete.")
