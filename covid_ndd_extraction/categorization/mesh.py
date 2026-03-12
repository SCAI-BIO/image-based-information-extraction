"""MeSH category term extraction from the descriptor XML file."""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

logger = logging.getLogger(__name__)

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Viral Entry and Neuroinvasion": [
        "neuroinvasion", "receptor", "ACE2", "blood-brain barrier", "BBB", "virus entry", "olfactory",
        "retrograde transport", "endocytosis", "direct invasion", "cranial nerve", "neural pathway",
        "transcribrial", "neurotropic", "trans-synaptic", "neuronal route", "olfactory nerve",
        "hematogenous", "choroid plexus", "neuronal transmission", "entry into CNS",
    ],
    "Immune and Inflammatory Response": [
        "immune", "cytokine", "inflammation", "interferon", "TNF", "IL-6", "IL6", "cytokine storm",
        "immune response", "inflammatory mediators", "macrophage", "microglia", "neutrophil",
        "lymphocyte", "innate immunity", "immune dysregulation", "chemokine", "T cell", "NLRP3",
        "antibody", "immune activation", "immune imbalance", "immune-mediated", "complement",
    ],
    "Neurodegenerative Mechanisms": [
        "neurodegeneration", "protein aggregation", "apoptosis", "cell death", "synaptic loss",
        "neurotoxicity", "oxidative stress", "mitochondrial dysfunction", "tau", "amyloid",
        "α-synuclein", "prion", "demyelination", "neuron loss", "misfolded proteins",
        "chronic neuronal damage", "neurodegenerative", "neuroinflammation",
    ],
    "Vascular Effects": [
        "stroke", "thrombosis", "vascular", "ischemia", "coagulation", "blood clot", "microthrombi",
        "endothelial", "vasculitis", "hemorrhage", "blood vessel", "vascular damage", "capillary",
        "clotting", "hypoperfusion", "angiopathy", "vasculopathy",
    ],
    "Psychological and Neurological Symptoms": [
        "cognitive", "memory", "fatigue", "depression", "anxiety", "brain fog", "psychiatric",
        "mood", "confusion", "neuropsychiatric", "emotional", "behavioral", "neurocognitive",
        "insomnia", "psychosocial", "attention", "motivation", "executive function", "suicidality",
    ],
    "Systemic Cross-Organ Effects": [
        "lungs", "liver", "kidney", "systemic", "multi-organ", "gastrointestinal", "heart",
        "cardiovascular", "endocrine", "renal", "pancreas", "organ failure", "liver damage",
        "pulmonary", "myocardial", "respiratory", "hypoxia", "oxygen deprivation", "fibrosis",
    ],
}


def extract_category_terms(
    mesh_xml_path: str,
    category_keywords: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Parse MeSH XML and extract terms grouped by category via keyword matching."""
    if category_keywords is None:
        category_keywords = CATEGORY_KEYWORDS

    tree = ET.parse(mesh_xml_path)
    root = tree.getroot()
    category_terms: dict[str, set[str]] = defaultdict(set)

    for descriptor in root.findall("DescriptorRecord"):
        descriptor_name_el = descriptor.find("DescriptorName/String")
        if descriptor_name_el is None:
            continue
        descriptor_name = descriptor_name_el.text or ""
        term_elements = descriptor.findall("ConceptList/Concept/TermList/Term/String")
        synonyms = [el.text for el in term_elements if el is not None and el.text]
        all_text = f"{descriptor_name} " + " ".join(synonyms)

        for category, keywords in category_keywords.items():
            if any(kw.lower() in all_text.lower() for kw in keywords):
                category_terms[category].update([descriptor_name] + synonyms)

    return {cat: sorted(terms) for cat, terms in category_terms.items()}


def save_to_json(category_terms: dict[str, list[str]], output_path: str) -> None:
    """Save category-term dictionary to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(category_terms, f, indent=2)
    logger.info("Category terms saved to %s", output_path)
