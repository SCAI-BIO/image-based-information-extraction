"""Rule-based diagram parser baseline using OCR and contour detection."""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

AREA_THRESH = 50
OCR_CONF_THRESH = 60
TESSERACT_CONFIG = "--psm 6"


def _nearest_label(point: tuple, texts: pd.DataFrame) -> str:
    if texts.empty:
        return "N/A"
    dx = texts["cx"] - point[0]
    dy = texts["cy"] - point[1]
    idx = (dx * dx + dy * dy).idxmin()
    return texts.loc[idx, "text"]


def parse_image(img_path: str) -> list[tuple]:
    """Parse one image and return a list of (image_name, subject, relation, object) tuples."""
    import cv2
    import pytesseract

    img = cv2.imread(img_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(bin_img, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arrows = [c for c in contours if cv2.contourArea(c) > AREA_THRESH]

    ocr_df = pytesseract.image_to_data(bin_img, output_type=pytesseract.Output.DATAFRAME, config=TESSERACT_CONFIG)
    if ocr_df is None or ocr_df.empty or "conf" not in ocr_df:
        texts = pd.DataFrame(columns=["text", "left", "top", "width", "height", "cx", "cy"])
    else:
        ocr_df["conf"] = pd.to_numeric(ocr_df["conf"], errors="coerce")
        texts = ocr_df[(ocr_df["conf"] >= OCR_CONF_THRESH) & ocr_df["text"].notna()][
            ["text", "left", "top", "width", "height"]
        ].copy()
        texts["text"] = texts["text"].str.strip()
        texts = texts[texts["text"].str.len() > 0]
        for col in ["left", "top", "width", "height"]:
            texts[col] = pd.to_numeric(texts[col], errors="coerce")
        texts = texts.dropna(subset=["left", "top", "width", "height"]).reset_index(drop=True)
        texts["cx"] = texts["left"] + texts["width"] / 2.0
        texts["cy"] = texts["top"] + texts["height"] / 2.0

    triples = []
    if not arrows or texts.empty:
        return triples

    for cnt in arrows:
        x, y, w, h = cv2.boundingRect(cnt)
        tail = (x, y + h // 2)
        head = (x + w, y + h // 2)
        s = _nearest_label(tail, texts)
        o = _nearest_label(head, texts)
        if s == "N/A" or o == "N/A":
            continue

        def _has_minus(t: str) -> bool:
            return any(ch in t for ch in ["−", "–", "-"])

        predicate = (
            "increases" if ("+" in s or "+" in o)
            else "decreases" if (_has_minus(s) or _has_minus(o))
            else "interacts_with"
        )
        triples.append((os.path.basename(img_path), s, predicate, o))

    return triples


def run_image_baseline(
    image_dir: str = "data/CBM_data/images_CBM_subset",
    output: str = "data/triples_output/rule_based_triples.csv",
) -> None:
    """Parse all images in *image_dir* and save extracted triples to *output*.

    Parameters
    ----------
    image_dir:
        Directory containing images.
    output:
        Path for the output CSV file.
    """
    images = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
    ]
    logger.info("Found %d images in %s", len(images), image_dir)

    all_triples: list[tuple] = []
    for fname in images:
        fpath = os.path.join(image_dir, fname)
        try:
            triples = parse_image(fpath)
            all_triples.extend(triples)
            logger.info("[OK] %s: %d triples", fname, len(triples))
        except Exception as exc:
            logger.error("[Error] %s: %s", fname, exc)

    if all_triples:
        df = pd.DataFrame(all_triples, columns=["image", "subject", "relation", "object"])
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        df.to_csv(output, index=False)
        logger.info("Saved %d triples to %s", len(df), output)
    else:
        logger.warning("No triples extracted.")
