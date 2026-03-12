"""Biological insights figures and tables from Neo4j Cypher CSV exports."""

# This module re-exports the main entry point from the original bio_insights.py.
# All figure logic is preserved; credentials are drawn from settings when needed.

from __future__ import annotations

import logging
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    import networkx as nx
    from matplotlib.lines import Line2D
    _HAS_GRAPH_DEPS = True
except Exception:
    GraphDatabase = None  # type: ignore[assignment]
    nx = None  # type: ignore[assignment]
    _HAS_GRAPH_DEPS = False

PALETTE = {
    "CBM": "#3274a1",
    "GPT": "#e1812c",
    "GPT-fulltext": "#3a923a",
}

DRUG = "MINOCYCLINE"
MINI_TARGETS = [
    "viral replication", "virus_replication", "microglial cell activation",
    "sars-cov2 crossing blood brain barrier", "impaired_neural-microglial_communication",
    "hypoxia-induced_neuroinflammation", "neuroinflammation", "neuronal_and_glial_injury",
]
BENEFICIAL_DOWN = ["DECREASE", "INHIBIT", "BLOCK", "REDUCE", "ALLEVIATES", "COUNTERACTS", "PREVENTS", "ATTENUATES"]
ALIASES = {
    "MINOCYCLINE": "Minocycline",
    "NEUROINFLAMMATION": "Neuroinflammation",
    "VIRAL REPLICATION": "Viral \nreplication",
}


def _standardize_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = df.copy()
    for cand in ("entity", "cytokine", "name", "x.name", "x", "mediator", "neighbor"):
        if cand in df.columns:
            df.rename(columns={cand: "name"}, inplace=True)
            break
    else:
        raise ValueError(f"[{kind}] Cannot find name column. Got: {df.columns.tolist()}")
    for col in ("deg_total", "deg_cbm", "deg_gpt", "deg_gpt_fulltext"):
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["name_short"] = df["name"].astype(str).str.replace("_", " ")
    return df


def _make_hbar(df: pd.DataFrame, title: str, outfile: Path, top_n: int, dpi: int) -> pd.DataFrame:
    d = df.sort_values("deg_total", ascending=False).head(top_n).iloc[::-1]
    plt.figure(figsize=(9, max(4, 0.45 * len(d))), dpi=dpi)
    plt.barh(d["name_short"], d["deg_gpt"], label="GPT (images)", color=PALETTE["GPT"], edgecolor="black", linewidth=0.3)
    plt.barh(d["name_short"], d["deg_cbm"], label="CBM", color=PALETTE["CBM"], left=d["deg_gpt"], edgecolor="black", linewidth=0.3)
    plt.barh(d["name_short"], d["deg_gpt_fulltext"], label="GPT-fulltext", color=PALETTE["GPT-fulltext"], left=d["deg_gpt"] + d["deg_cbm"], edgecolor="black", linewidth=0.3)
    plt.xlim(0, d["deg_total"].max() * 1.1)
    plt.xlabel("Node degree")
    plt.title(title)
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi)
    plt.close()
    return d.iloc[::-1]


def figure_shared_and_cytokine_hubs(input_dir: Path, output_dir: Path, top_n: int, dpi: int) -> None:
    q1a = _standardize_columns(pd.read_csv(input_dir / "covid_ndd_shared_hubs.csv"), "shared_hubs")
    q1b = _standardize_columns(pd.read_csv(input_dir / "covid_ndd_cytokine_hubs.csv"), "cytokine_hubs")
    top_a = _make_hbar(q1a, textwrap.fill("Shared Hubs Within 2 Hops of Both COVID and NDD", 60), output_dir / "Fig_shared_hubs.tiff", top_n, dpi)
    top_b = _make_hbar(q1b, "Cytokine/Chemokine Hubs Near Both COVID and NDD", output_dir / "Fig_cytokine_hubs.tiff", top_n, dpi)
    top_a.to_csv(output_dir / "shared_hubs_topN.csv", index=False)
    top_b.to_csv(output_dir / "cytokine_hubs_topN.csv", index=False)


def figure_bbb_mediators_and_glia_neighbors(input_dir: Path, output_dir: Path, dpi: int) -> None:
    a2 = pd.read_csv(input_dir / "covid_bbb_path_mediators.csv")
    if "mediator" not in a2.columns:
        for cand in ("name", "entity", "node"):
            if cand in a2.columns:
                a2.rename(columns={cand: "mediator"}, inplace=True)
                break
    for col in ("edges_on_paths", "cbm_edges_on_paths", "gpt_edges_on_paths"):
        if col not in a2.columns:
            a2[col] = 0
    a2_sorted = a2.sort_values("edges_on_paths", ascending=False).head(10)
    y = np.arange(len(a2_sorted))
    plt.figure(figsize=(8, 6), dpi=dpi)
    plt.barh(y, a2_sorted["cbm_edges_on_paths"], label="CBM edges", color=PALETTE["CBM"])
    plt.barh(y, a2_sorted["gpt_edges_on_paths"], left=a2_sorted["cbm_edges_on_paths"], label="GPT edges", color=PALETTE["GPT"], alpha=0.85)
    plt.yticks(y, a2_sorted["mediator"])
    plt.xlabel("Edges on COVID↔BBB paths")
    plt.title("Top Mediators on COVID↔BBB Paths")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "BBB_mediators_top10.tiff", dpi=dpi)
    plt.savefig(output_dir / "BBB_mediators_top10.png", dpi=dpi)
    plt.close()

    b1 = pd.read_csv(input_dir / "glia_neighbors.csv")
    covid_terms = {"covid-19", "sars-cov-2", "severe acute respiratory syndrome coronavirus 2"}
    name_col = "neighbor" if "neighbor" in b1.columns else b1.columns[0]
    b1 = b1[~b1[name_col].astype(str).str.lower().isin(covid_terms)].copy()
    if "glia_anchor" in b1.columns:
        b1 = b1[b1["glia_anchor"].astype(str).str.upper() == "ASTROCYTE ACTIVATION"].copy()
    for src, dest in [("cbm_edges", "edges_cbm"), ("gpt_edges", "edges_gpt")]:
        if dest not in b1.columns:
            b1[dest] = b1[src] if src in b1.columns else 0
    if "deg_local" not in b1.columns:
        b1["deg_local"] = b1.get("edges_cbm", 0).fillna(0) + b1.get("edges_gpt", 0).fillna(0)
    b1_sorted = b1.sort_values("deg_local", ascending=False).head(15)
    y = np.arange(len(b1_sorted))
    plt.figure(figsize=(9, 7), dpi=dpi)
    plt.barh(y, b1_sorted["edges_cbm"], label="CBM edges", color=PALETTE["CBM"])
    plt.barh(y, b1_sorted["edges_gpt"], left=b1_sorted["edges_cbm"], label="GPT edges", color=PALETTE["GPT"], alpha=0.85)
    plt.yticks(y, b1_sorted[name_col])
    plt.xlabel("Number of edges")
    plt.title("Top Neighbors of ASTROCYTE ACTIVATION")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "Astrocyte_neighbors_top15_noCOVID.tiff", dpi=dpi)
    plt.savefig(output_dir / "Astrocyte_neighbors_top15_noCOVID.png", dpi=dpi)
    plt.close()


def run_bio_insights(
    input_dir: str = "data/bio_insights/neo4j_results",
    output_dir: str = "data/bio_insights/outputs",
    top_n: int = 20,
    dpi: int = 300,
    run_neo4j: bool = False,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    neo4j_db: str = "neo4j",
) -> None:
    """Generate publication figures and tables for the biological insights subsection."""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_shared_and_cytokine_hubs(in_dir, out_dir, top_n=top_n, dpi=dpi)
    figure_bbb_mediators_and_glia_neighbors(in_dir, out_dir, dpi=dpi)

    if run_neo4j:
        if not _HAS_GRAPH_DEPS:
            logger.warning("neo4j/networkx not installed — skipping Neo4j subgraphs.")
        elif not neo4j_password:
            logger.error("neo4j_password required when run_neo4j=True.")
        else:
            from covid_ndd_extraction.config import settings as _s
            _neo4j_minocycline_subgraphs(
                out_dir,
                uri=neo4j_uri or _s.neo4j_uri,
                user=neo4j_user or _s.neo4j_user,
                password=neo4j_password,
                db=neo4j_db,
                dpi=dpi,
            )

    produced = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    logger.info("Saved outputs to %s: %s", out_dir.resolve(), produced)


def _neo4j_minocycline_subgraphs(output_dir: Path, uri: str, user: str, password: str, db: str, dpi: int) -> None:
    """Build Minocycline CBM / GPT-fulltext / combined subgraphs."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        cypher = """
        MATCH (c:Chemical {name:$drug})-[r]->(m)
        WHERE any(t IN $targets WHERE replace(toLower(m.name),'_',' ') = replace(toLower(t),'_',' '))
          AND type(r) IN $down
          AND r.source IN $sources
        RETURN c.name AS drug, m.name AS target, type(r) AS rel, r.source AS source
        """

        def fetch(sources):
            rows = []
            with driver.session(database=db) as s:
                for rec in s.run(cypher, drug=DRUG, targets=MINI_TARGETS, down=BENEFICIAL_DOWN, sources=sources):
                    rows.append((rec["drug"], rec["target"], {"source": rec["source"], "rel": rec["rel"]}))
            return rows

        edges_cbm = fetch(["CBM"])
        edges_gptft = fetch(["GPT-fulltext"])

        for label, edges in [("cbm", edges_cbm), ("gptfulltext", edges_gptft), ("combined", edges_cbm + edges_gptft)]:
            G = nx.DiGraph()
            targets = []
            for u, v, attr in edges:
                G.add_node(u); G.add_node(v); G.add_edge(u, v, **attr); targets.append(v)
            n = max(1, len(set(targets)))
            angles = np.linspace(0, 2 * math.pi, n, endpoint=False) + math.radians(90)
            pos = {DRUG: (0.0, 0.0)}
            for t, ang in zip(sorted(set(targets)), angles):
                pos[t] = (4.5 * math.cos(ang), 4.5 * math.sin(ang))
            plt.figure(figsize=(8, 9), dpi=dpi)
            nx.draw_networkx_nodes(G, pos, node_size=5000, node_color="#A0C4FF", alpha=0.65, linewidths=0)
            nx.draw_networkx_labels(G, pos, labels={n: n.replace("_", " ").title() for n in G.nodes()}, font_size=10)
            nx.draw_networkx_edges(G, pos, width=2.2, arrows=True, arrowsize=14, alpha=0.9)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_dir / f"minocycline_{label}.tiff")
            plt.savefig(output_dir / f"minocycline_{label}.png", dpi=dpi)
            plt.close()
    finally:
        driver.close()
