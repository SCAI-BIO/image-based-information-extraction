"""BERT-based categorisation of biomedical triples using MeSH keywords."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    text = re.sub(r"[;_\-]", " ", str(text))
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _combine_subject_object(row: pd.Series) -> str:
    fields = [row.get("Subject"), row.get("Object")]
    return " ".join([_normalize_text(f) for f in fields if pd.notna(f) and str(f).strip().lower() != "missing"])


def _bert_keyword_classify(
    text: str,
    category_keyword_embeddings: dict,
    model: SentenceTransformer,
    threshold: float = 0.5,
    aggregation: str = "max",
) -> str:
    process_embedding = model.encode(text, convert_to_tensor=True)
    best_category: str | None = None
    best_score = 0.0

    for category, keyword_embeddings in category_keyword_embeddings.items():
        if keyword_embeddings is None or len(keyword_embeddings) == 0:
            continue
        cosine_scores = util.pytorch_cos_sim(process_embedding, keyword_embeddings)[0]
        score = float(torch.max(cosine_scores) if aggregation == "max" else torch.mean(cosine_scores))
        if score > best_score:
            best_score = score
            best_category = category

    return best_category if best_score >= threshold else "Uncategorized"


def run_classification(
    input_path: str,
    mesh_json_path: str,
    output_path: str,
    mode: str = "pp",
) -> None:
    """Classify biomedical triples using BERT + MeSH keyword embeddings.

    Parameters
    ----------
    input_path:
        Path to input CSV or XLSX file.
    mesh_json_path:
        Path to MeSH category JSON produced by ``categorization.mesh``.
    output_path:
        Output file path stem (no extension); ``.csv`` and ``.xlsx`` are written.
    mode:
        ``"pp"`` — classify by Pathophysiological Process column.
        ``"subjobj"`` — classify by Subject + Object columns.
    """
    logger.info("Loading data from %s", input_path)
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_excel(input_path)

    with open(mesh_json_path) as f:
        category_terms: dict[str, list[str]] = json.load(f)

    category_terms = {cat: [_normalize_text(t) for t in terms] for cat, terms in category_terms.items()}

    logger.info("Initialising sentence-transformers model …")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    category_keyword_embeddings = {
        cat: model.encode(terms, convert_to_tensor=True) if terms else None
        for cat, terms in category_terms.items()
    }

    if mode == "pp":
        df["Normalized_Input"] = df["Pathophysiological Process"].apply(_normalize_text)
    elif mode == "subjobj":
        df["Normalized_Input"] = df.apply(_combine_subject_object, axis=1)
    else:
        raise ValueError("mode must be 'pp' or 'subjobj'")

    df["Category"] = df["Normalized_Input"].apply(
        lambda x: _bert_keyword_classify(x, category_keyword_embeddings, model)
    )

    counts = Counter(df["Category"])
    logger.info("Category counts: %s", dict(counts))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path + ".csv", index=False)
    df.to_excel(output_path + ".xlsx", index=False)
    logger.info("Classification complete — saved to %s", output_path)
