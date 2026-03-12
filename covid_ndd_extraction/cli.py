"""Unified Typer CLI for the covid-ndd-extraction pipeline."""

from __future__ import annotations

import logging
import os
from typing import Optional

import typer

app = typer.Typer(help="COVID-19 / NDD semantic triple extraction pipeline.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


@app.command("enrich")
def enrich(
    query: str = typer.Argument(..., help="Search query for Google Images"),
    output_dir: str = typer.Option("data/enrichment_data", help="Output directory"),
    main: int = typer.Option(100, help="Number of main images"),
    similar: int = typer.Option(100, help="Number of similar images per main image"),
    output_raw: str = typer.Option("Enrichment_Search_URLs", help="Raw output filename stem"),
    output_clean: str = typer.Option("Enrichment_Cleaned", help="Cleaned output filename stem"),
):
    """Scrape Google Images and deduplicate via perceptual hashing."""
    from covid_ndd_extraction.enrichment.image_search import run_image_search_pipeline
    run_image_search_pipeline(
        query=query,
        output_dir=output_dir,
        num_images_main=main,
        num_images_similar=similar,
        output_raw=output_raw,
        output_clean=output_clean,
    )


@app.command("filter-urls")
def filter_urls(
    input: str = typer.Argument(..., help="Excel file with 'image_url' column"),
    output_dir: str = typer.Option("data/URL_relevance_analysis", help="Output directory"),
    api_key: Optional[str] = typer.Option(None, envvar="OPENAI_API_KEY", help="OpenAI API key"),
):
    """Classify image URLs as relevant/irrelevant using GPT-4o."""
    from covid_ndd_extraction.enrichment.relevance import relevance_check_main
    relevance_check_main(input, output_dir, api_key)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@app.command("extract-images")
def extract_images(
    input: str = typer.Argument(..., help="Excel file with image URLs"),
    output: str = typer.Argument(..., help="Output file stem (no extension)"),
    api_key: Optional[str] = typer.Option(None, envvar="OPENAI_API_KEY", help="OpenAI API key"),
):
    """Extract semantic triples from biomedical images via GPT-4o."""
    from covid_ndd_extraction.extraction.images import triples_extraction_from_urls
    triples_extraction_from_urls(input, output, api_key)


@app.command("extract-text")
def extract_text(
    input: str = typer.Argument(..., help="JSON file with full-text articles"),
    output_dir: str = typer.Option("data/gold_standard_comparison", help="Output directory"),
    api_key: Optional[str] = typer.Option(None, envvar="OPENAI_API_KEY", help="OpenAI API key"),
):
    """Extract semantic triples from full-text biomedical articles via GPT-4o."""
    from covid_ndd_extraction.extraction.fulltext import triples_extraction_from_articles
    triples_extraction_from_articles(input, output_dir, api_key)


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


@app.command("select-subset")
def select_subset(
    cbm_metadata: str = typer.Option("data/CBM_data/Data_CBM_with_GitHub_URLs.xlsx", help="CBM metadata Excel"),
    cbm_triples: str = typer.Option("data/CBM_data/Triples_CBM_Gold_Standard.xlsx", help="CBM triples Excel"),
    output_dir: str = typer.Option("data/prompt_engineering/cbm_files", help="Output directory"),
    n: int = typer.Option(50, help="Number of URLs to sample"),
    seed: Optional[int] = typer.Option(None, help="Random seed"),
):
    """Randomly select a URL subset and matching gold-standard triples."""
    from covid_ndd_extraction.enrichment.subset_selector import select_url_subset
    select_url_subset(cbm_metadata, cbm_triples, output_dir, n, seed)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@app.command("threshold")
def threshold(
    gold: str = typer.Argument(..., help="Gold-standard Excel file"),
    eval: str = typer.Argument(..., help="GPT triples Excel file"),
    output_dir: str = typer.Option("data/figures_output", help="Figure output directory"),
    stats_dir: str = typer.Option("data/prompt_engineering/statistical_data", help="Stats directory"),
):
    """Evaluate cosine similarity thresholds for triple matching."""
    from covid_ndd_extraction.evaluation.threshold import evaluate_thresholds
    evaluate_thresholds(gold, eval, output_dir, stats_dir)


@app.command("assess-prompts")
def assess_prompts(
    gold: str = typer.Argument(..., help="Gold-standard Excel file"),
    prompt_files: list[str] = typer.Argument(..., help="Prompt Excel files (Prompt1, Prompt2, ...)"),
    threshold: float = typer.Option(0.85, help="Similarity threshold"),
    output_stats: str = typer.Option("data/prompt_engineering/statistical_data/Prompt_Comparison.xlsx"),
    output_figures_dir: str = typer.Option("data/figures_output"),
):
    """Compare GPT triples from multiple prompts vs a gold standard."""
    from covid_ndd_extraction.evaluation.prompt import run_prompt_assessment
    prompt_paths = {f"Prompt {i+1}": p for i, p in enumerate(prompt_files)}
    run_prompt_assessment(gold, prompt_paths, threshold, output_stats, output_figures_dir)


@app.command("assess-params")
def assess_params(
    gold: str = typer.Argument(..., help="Gold-standard Excel file"),
    param_files: list[str] = typer.Argument(..., help="Parameter Excel files in order"),
    labels: str = typer.Option("", help="Comma-separated labels (same order as files)"),
    threshold: float = typer.Option(0.85, help="Similarity threshold"),
    output_stats: str = typer.Option("data/prompt_engineering/statistical_data/Hyperparameter_Assessment.xlsx"),
    output_figures_dir: str = typer.Option("data/figures_output"),
):
    """Compare GPT triples under different temperature/top_p settings vs gold standard."""
    from covid_ndd_extraction.evaluation.hyperparameter import run_hyperparameter_assessment
    label_list = [l.strip() for l in labels.split(",") if l.strip()] if labels else [f"Setting {i+1}" for i in range(len(param_files))]
    paths = dict(zip(label_list, param_files))
    run_hyperparameter_assessment(gold, paths, threshold, output_stats, output_figures_dir)


@app.command("compare")
def compare(
    gold: str = typer.Argument(..., help="Gold-standard Excel file"),
    eval: str = typer.Argument(..., help="GPT triples Excel file"),
    threshold: float = typer.Option(0.85, help="Similarity threshold"),
    output_dir: str = typer.Option("data/gold_standard_comparison", help="Output directory"),
    figures_dir: str = typer.Option("data/figures_output", help="Figures directory"),
):
    """Compare GPT triples to gold standard using BioBERT + Hungarian matching."""
    import pandas as pd
    from covid_ndd_extraction.evaluation.gold_standard import evaluate_images
    df_gold = pd.read_excel(gold)
    df_eval = pd.read_excel(eval)
    metrics = evaluate_images(df_gold, df_eval, threshold=threshold, output_dir=output_dir, figures_dir=figures_dir)
    typer.echo(f"Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}")


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------


@app.command("extract-mesh")
def extract_mesh(
    mesh_xml: str = typer.Argument(..., help="Path to MeSH descriptor XML (desc2025.xml)"),
    output: str = typer.Argument(..., help="Output JSON path"),
):
    """Extract MeSH terms grouped by biomedical category."""
    from covid_ndd_extraction.categorization.mesh import extract_category_terms, save_to_json
    terms = extract_category_terms(mesh_xml)
    save_to_json(terms, output)


@app.command("categorize")
def categorize(
    input: str = typer.Argument(..., help="Input CSV or XLSX file"),
    mesh: str = typer.Option("data/MeSh_data/mesh_category_terms.json", help="MeSH keyword JSON"),
    output: str = typer.Argument(..., help="Output file stem (no extension)"),
    mode: str = typer.Option("pp", help="Classification mode: 'pp' or 'subjobj'"),
):
    """Classify biomedical triples using BERT + MeSH keywords."""
    from covid_ndd_extraction.categorization.triples import run_classification
    run_classification(input, mesh, output, mode)


@app.command("categorize-fallback")
def categorize_fallback(
    input: str = typer.Argument(..., help="Input XLSX with 'Uncategorized' entries"),
    output: str = typer.Argument(..., help="Output file stem (no extension)"),
    api_key: Optional[str] = typer.Option(None, envvar="OPENAI_API_KEY", help="OpenAI API key"),
):
    """Use GPT-4o to categorize entries labelled 'Uncategorized' by BERT."""
    from covid_ndd_extraction.categorization.fallback import categorize_uncategorized
    categorize_uncategorized(input, output, api_key)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@app.command("upload-graph")
def upload_graph(
    file: str = typer.Argument(..., help="CSV/TSV/Excel file with triples"),
    uri: Optional[str] = typer.Option(None, envvar="NEO4J_URI", help="Neo4j URI"),
    user: Optional[str] = typer.Option(None, envvar="NEO4J_USER", help="Neo4j user"),
    password: Optional[str] = typer.Option(None, envvar="NEO4J_PASSWORD", help="Neo4j password"),
    node_label: str = typer.Option("Entity", help="Base node label"),
    subject_col: Optional[str] = typer.Option(None),
    predicate_col: Optional[str] = typer.Option(None),
    object_col: Optional[str] = typer.Option(None),
    dry_run: bool = typer.Option(False, help="Preview only; no DB writes"),
):
    """Upload triples to Neo4j."""
    from covid_ndd_extraction.graph.loader import upload_triples
    upload_triples(file, uri, user, password, node_label, subject_col, predicate_col, object_col, dry_run)


@app.command("review-labels")
def review_labels(
    uri: Optional[str] = typer.Option(None, envvar="NEO4J_URI", help="Neo4j URI"),
    user: Optional[str] = typer.Option(None, envvar="NEO4J_USER", help="Neo4j user"),
    password: Optional[str] = typer.Option(None, envvar="NEO4J_PASSWORD", help="Neo4j password"),
    api_key: Optional[str] = typer.Option(None, envvar="OPENAI_API_KEY", help="OpenAI API key"),
):
    """Review and correct Neo4j node labels using GPT-4o."""
    from covid_ndd_extraction.graph.review_labels import run_label_review
    run_label_review(uri, user, password, api_key)


@app.command("bio-insights")
def bio_insights(
    input_dir: str = typer.Option("data/bio_insights/neo4j_results", help="Input CSV directory"),
    output_dir: str = typer.Option("data/bio_insights/outputs", help="Output directory"),
    top_n: int = typer.Option(20, help="Top-N entities in hub plots"),
    dpi: int = typer.Option(300, help="Figure DPI"),
    run_neo4j: bool = typer.Option(False, help="Render Minocycline subgraphs via Neo4j"),
    neo4j_uri: Optional[str] = typer.Option(None, envvar="NEO4J_URI"),
    neo4j_user: Optional[str] = typer.Option(None, envvar="NEO4J_USER"),
    neo4j_password: Optional[str] = typer.Option(None, envvar="NEO4J_PASSWORD"),
    neo4j_db: str = typer.Option("neo4j"),
):
    """Generate biological insights figures and tables."""
    from covid_ndd_extraction.graph.bio_insights import run_bio_insights
    run_bio_insights(input_dir, output_dir, top_n, dpi, run_neo4j, neo4j_uri, neo4j_user, neo4j_password, neo4j_db)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


@app.command("baseline")
def baseline(
    image_dir: str = typer.Option("data/CBM_data/images_CBM_subset", help="Image directory"),
    output: str = typer.Option("data/triples_output/rule_based_triples.csv", help="Output CSV path"),
):
    """Run rule-based image parser baseline using OCR."""
    from covid_ndd_extraction.baseline.image_baseline import run_image_baseline
    run_image_baseline(image_dir, output)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    app()
