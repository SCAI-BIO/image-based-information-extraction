"""Upload semantic triples to Neo4j with BEL-style relation normalisation."""

# This module re-exports the core upload logic from the original neo4j_load.py,
# reorganised as importable functions with settings-based credentials.

from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from covid_ndd_extraction.config import settings

logger = logging.getLogger(__name__)

NA_SET = {"", "nan", "none", "unknown", "null"}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def is_na(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str) and v.strip().lower() in NA_SET:
        return True
    return False


def coerce_value(v):
    if is_na(v):
        return None
    if isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, (list, tuple, set, dict)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return str(v)


def sanitize_label(label: str) -> str:
    s = str(label or "").strip()
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    if not s:
        s = "LABEL"
    if re.match(r"^\d", s):
        s = f"L_{s}"
    return s


def sanitize_prop_key(key: str) -> str:
    k = str(key or "").strip().lower()
    k = re.sub(r"[^A-Za-z0-9_]", "_", k)
    if not re.match(r"^[A-Za-z]", k):
        k = "p_" + k
    return k


# ---------------------------------------------------------------------------
# CURIE / IRI normalisation
# ---------------------------------------------------------------------------

CURIE_PREFIX_UPPER = re.compile(r"^[a-z0-9]+:", re.I)


def norm_curie(x: Optional[str]) -> Optional[str]:
    if is_na(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\s+", "", s)
    if CURIE_PREFIX_UPPER.match(s):
        pfx, rest = s.split(":", 1)
        s = f"{pfx.upper()}:{rest}"
    s = re.sub(r":0+(?=\d+$)", ":", s)
    return s


def norm_iri(x: Optional[str]) -> Optional[str]:
    if is_na(x):
        return None
    s = str(x).strip().rstrip("/").replace("http://", "https://")
    return s


def canon_name(x: str) -> str:
    s = (x or "").strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(?<=\w)[\-\_\s]+(?=\w)", "", s)
    s = s.replace("covid19", "covid-19").replace("sarscov2", "sars-cov-2")
    return s


def build_entity_key(
    name: str,
    curie: Optional[str],
    iri: Optional[str],
    raw_id: Optional[str],
) -> str:
    c = norm_curie(curie)
    if c:
        return f"curie:{c}"
    i = norm_iri(iri)
    if i:
        return f"iri:{i}"
    if not is_na(raw_id):
        return f"id:{str(raw_id).strip()}"
    return f"name:{canon_name(name)}"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def sniff_read_csv(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8", errors="ignore") as f:
        sample = f.read(2048)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            sep = dialect.delimiter
        except csv.Error:
            sep = ","
    return pd.read_csv(path, sep=sep, encoding="utf-8", engine="python")


def read_any(path: Path, sheet: Optional[str | int] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt", ".pipe", ".psv"}:
        return sniff_read_csv(str(path))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet, engine="openpyxl") if sheet is not None else pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------


def _pick_header(lower_map: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def detect_spo_columns(
    df: pd.DataFrame,
    subject_override: Optional[str] = None,
    predicate_override: Optional[str] = None,
    object_override: Optional[str] = None,
) -> Tuple[str, str, str]:
    lower_map: Dict[str, str] = {str(c).lower().strip(): c for c in df.columns}

    def require(name: str) -> str:
        low = name.lower().strip()
        if low not in lower_map:
            raise ValueError(f"Override column '{name}' not found in: {list(df.columns)}")
        return lower_map[low]

    subject = require(subject_override) if subject_override else _pick_header(lower_map, ["subject", "subj", "s", "head", "source", "from"])
    predicate = require(predicate_override) if predicate_override else _pick_header(lower_map, ["predicate", "pred", "p", "relation", "rel", "edge", "link"])
    object_ = require(object_override) if object_override else _pick_header(lower_map, ["object", "obj", "o", "target", "tail", "to"])

    if not (subject and predicate and object_):
        raise ValueError(
            "Could not detect subject/predicate/object columns.\n"
            f"Found headers: {list(df.columns)}\n"
            "Use subject_col/predicate_col/object_col to specify explicitly."
        )
    return subject, predicate, object_


# ---------------------------------------------------------------------------
# Relation normalisation (BEL-compact)
# ---------------------------------------------------------------------------

BEL_RELATIONS = {
    "directlyIncreases", "directlyDecreases", "increases", "decreases", "regulates",
    "association", "positiveCorrelation", "negativeCorrelation", "noEffect",
    "partOf", "hasMember", "hasComponent", "localizes", "translocates", "biomarkerFor",
}

REL_TO_BEL = {
    "DIRECT_INCREASE": "directlyIncreases", "DIRECT_DECREASE": "directlyDecreases",
    "INCREASE": "increases", "DECREASE": "decreases", "REGULATES": "regulates",
    "ASSOCIATION": "association", "CORRELATION_POS": "positiveCorrelation",
    "CORRELATION_NEG": "negativeCorrelation", "NO_EFFECT": "noEffect",
    "PART_OF": "partOf", "HAS_MEMBER": "hasMember", "HAS_COMPONENT": "hasComponent",
    "LOCALIZES": "localizes", "TRANSLOCATES": "translocates", "BIOMARKER_FOR": "biomarkerFor",
}

EXACT_MAP = {
    "activates": "INCREASE", "upregulates": "INCREASE", "induces": "INCREASE",
    "stimulates": "INCREASE", "enhances": "INCREASE", "promotes": "INCREASE",
    "increases": "INCREASE", "augments": "INCREASE", "boosts": "INCREASE",
    "elevates": "INCREASE", "potentiates": "INCREASE",
    "phosphorylates": "DIRECT_INCREASE", "phosphorylation": "DIRECT_INCREASE",
    "inhibits": "DECREASE", "suppresses": "DECREASE", "represses": "DECREASE",
    "blocks": "DECREASE", "attenuates": "DECREASE", "reduces": "DECREASE",
    "downregulates": "DECREASE", "abrogates": "DECREASE", "impairs": "DECREASE",
    "regulates": "REGULATES", "modulates": "REGULATES",
    "binds": "ASSOCIATION", "interacts": "ASSOCIATION", "associates": "ASSOCIATION",
    "no effect": "NO_EFFECT", "no change": "NO_EFFECT",
}

_NEGATION = re.compile(r"\b(no(t)?|doesn'?t|without|lack of|fails? to)\b", re.I)
_NO_CHANGE = re.compile(r"\b(no\s+(significant\s+)?(increase|decrease|change)|unchanged)\b", re.I)


def normalize_relation(pred: str) -> Tuple[str, str]:
    original = (pred or "").strip()
    plow = original.lower()
    for phrase, rel in EXACT_MAP.items():
        if phrase in plow:
            return rel, original
    return "ASSOCIATION", original


# ---------------------------------------------------------------------------
# Neo4j uploader
# ---------------------------------------------------------------------------


class Neo4jUploader:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        base_label: str = "Entity",
    ):
        from neo4j import GraphDatabase  # optional dep

        uri = uri or settings.neo4j_uri
        user = user or settings.neo4j_user
        password = password or settings.neo4j_password

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.base_label = sanitize_label(base_label)

    def close(self) -> None:
        self.driver.close()

    def ensure_constraints(self) -> None:
        with self.driver.session() as session:
            session.execute_write(
                lambda tx: tx.run(
                    f"CREATE CONSTRAINT entity_key_unique IF NOT EXISTS "
                    f"FOR (n:{self.base_label}) REQUIRE n.key IS UNIQUE"
                )
            )

    def upload(
        self,
        df: pd.DataFrame,
        s_col: str,
        p_col: str,
        o_col: str,
    ) -> None:
        with self.driver.session() as session:
            total = len(df)
            for i, row in df.iterrows():
                s_raw = str(row[s_col]).strip()
                p_raw = str(row[p_col]).strip()
                o_raw = str(row[o_col]).strip()
                if not all([s_raw, p_raw, o_raw]):
                    continue
                if any(v.lower() in NA_SET for v in [s_raw, p_raw, o_raw]):
                    continue

                compact_rel, original_pred = normalize_relation(p_raw)
                rel_type = sanitize_label(compact_rel)

                s_key = f"name:{canon_name(s_raw)}"
                o_key = f"name:{canon_name(o_raw)}"

                for key, name in [(s_key, s_raw), (o_key, o_raw)]:
                    session.execute_write(
                        lambda tx, k=key, n=name: tx.run(
                            f"MERGE (x:{self.base_label} {{key:$key}}) ON CREATE SET x.name=$name, x.created_at=timestamp()",
                            key=k, name=n,
                        )
                    )

                rel_props = {
                    "subject": s_raw, "predicate": p_raw, "object": o_raw,
                    "relation": compact_rel,
                    "bel_relation": REL_TO_BEL.get(compact_rel, "association"),
                    "original_predicate": original_pred,
                }
                session.execute_write(
                    lambda tx, sk=s_key, ok=o_key, rt=rel_type, rp=rel_props: tx.run(
                        f"MATCH (s:{self.base_label} {{key:$sk}}), (o:{self.base_label} {{key:$ok}}) "
                        f"MERGE (s)-[r:{rt}]->(o) ON CREATE SET r.created_at=timestamp() SET r += $rp",
                        sk=sk, ok=ok, rp=rp,
                    )
                )

                if (i + 1) % 500 == 0 or (i + 1) == total:
                    logger.info("Processed %d/%d rows", i + 1, total)


def upload_triples(
    file_path: str,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    node_label: str = "Entity",
    subject_col: Optional[str] = None,
    predicate_col: Optional[str] = None,
    object_col: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Load a CSV/Excel file and upload its triples to Neo4j."""
    path = Path(file_path)
    df = read_any(path)
    s_col, p_col, o_col = detect_spo_columns(df, subject_col, predicate_col, object_col)

    df = df.dropna(subset=[s_col, p_col, o_col])
    for c in [s_col, p_col, o_col]:
        df = df[df[c].astype(str).str.strip() != ""]
        df = df[~df[c].astype(str).str.strip().str.lower().isin(NA_SET)]

    logger.info("Columns: subject='%s', predicate='%s', object='%s'", s_col, p_col, o_col)
    logger.info("Rows to upload: %d", len(df))

    if dry_run:
        logger.info("Dry run — no writes performed.")
        return

    uploader = Neo4jUploader(uri=uri, user=user, password=password, base_label=node_label)
    try:
        uploader.ensure_constraints()
        uploader.upload(df, s_col, p_col, o_col)
        logger.info("Upload complete.")
    finally:
        uploader.close()
