"""BioBERT embedding utilities shared across evaluation scripts."""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.1"


def normalize(text: str | float) -> str:
    """Lowercase, replace separators, strip punctuation and excess whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_triple(subject: str, predicate: str, obj: str) -> str:
    """Return a normalised sentence representation of a triple."""
    return f"{normalize(subject)} {normalize(predicate)} {normalize(obj)}"


@lru_cache(maxsize=1)
def load_biobert() -> tuple[AutoTokenizer, AutoModel]:
    """Load and cache the BioBERT tokeniser and model (eval mode)."""
    logger.info("Loading BioBERT model: %s", BIOBERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
    model = AutoModel.from_pretrained(BIOBERT_MODEL)
    model.eval()
    return tokenizer, model


def embed(text: str, tokenizer: AutoTokenizer, model: AutoModel) -> torch.Tensor:
    """Return mean-pooled CLS embedding for a single text string."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


def batch_embed(
    texts: list[str], tokenizer: AutoTokenizer, model: AutoModel
) -> list[torch.Tensor]:
    """Return a list of embeddings for a list of texts."""
    return [embed(t, tokenizer, model) for t in texts]


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def build_sim_matrix(
    emb_rows: list[torch.Tensor],
    emb_cols: list[torch.Tensor],
) -> np.ndarray:
    """Build a (len(emb_rows) × len(emb_cols)) cosine similarity matrix."""
    mat = np.zeros((len(emb_rows), len(emb_cols)))
    for i, row_emb in enumerate(emb_rows):
        for j, col_emb in enumerate(emb_cols):
            mat[i, j] = cosine_sim(row_emb, col_emb)
    return mat
