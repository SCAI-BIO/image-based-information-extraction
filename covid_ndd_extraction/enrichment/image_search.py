"""Google Images search and perceptual-hash deduplication via Selenium."""

from __future__ import annotations

import logging
import os
import random
import time
from io import BytesIO
from http.client import RemoteDisconnected

import pandas as pd
import requests
from PIL import Image
import imagehash
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Selenium helpers
# ---------------------------------------------------------------------------


def _find_thumbnails(driver, num_images: int):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import StaleElementReferenceException

    for _ in range(int(num_images / 4)):
        try:
            body = driver.find_element(By.TAG_NAME, "body")
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(random.uniform(0.2, 0.5))
        except StaleElementReferenceException:
            continue

    selectors = [".rg_i.Q4LuWd", ".isv-r.PNCib.MSM1fd.BUooTd img", ".rg_i", ".H8Rx8c img"]
    for selector in selectors:
        try:
            thumbnails = driver.find_elements(By.CSS_SELECTOR, selector)
            if thumbnails:
                return thumbnails
        except Exception:
            pass
    return []


def _find_image_url(url_list: dict, thumbnail, driver) -> None:
    from selenium.webdriver.common.by import By
    from selenium.common.exceptions import NoSuchElementException

    thumb_url = thumbnail.get_attribute("src")
    url_list["thumbnail_url"].append(thumb_url or "Not_found")
    try:
        img_btn = driver.find_element(By.CSS_SELECTOR, ".sFlh5c.FyHeAf.iPVvYb")
        url_list["image_url"].append(img_btn.get_attribute("src") or "Not_found")
    except NoSuchElementException:
        url_list["image_url"].append("Not_found")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def google_image_search(
    query: str,
    output_dir: str,
    num_images_main: int = 100,
    num_images_similar: int = 100,
    output_raw_filename: str = "Enrichment_Search_URLs",
) -> pd.DataFrame:
    """Scrape Google Images and save raw URL list to Excel."""
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.common.exceptions import (
        StaleElementReferenceException,
        ElementClickInterceptedException,
        NoSuchElementException,
    )

    options = webdriver.ChromeOptions()
    options.add_argument("user-agent=Mozilla/5.0")
    driver = webdriver.Chrome(options=options)

    search_url = f"https://www.google.com/search?tbm=isch&q={query}"
    driver.get(search_url)
    time.sleep(random.uniform(2, 5))

    thumbnails = _find_thumbnails(driver, num_images_main)
    url_list: dict = {"thumbnail_url": [], "image_url": []}

    for thumb_num in range(min(num_images_main, len(thumbnails))):
        thumbnail = thumbnails[thumb_num]
        try:
            thumbnail.click()
            time.sleep(random.uniform(2, 5))
            _find_image_url(url_list, thumbnail, driver)

            try:
                see_more = driver.find_element(By.CSS_SELECTOR, ".DFaQu.T38yZ.cS4Vcb-pGL6qe-lfQAOe")
                see_more.click()
                similar_thumbnails = _find_thumbnails(driver, num_images_similar)
                for sim_thumb in similar_thumbnails[:num_images_similar]:
                    try:
                        sim_thumb.click()
                        time.sleep(random.uniform(2, 5))
                        _find_image_url(url_list, sim_thumb, driver)
                    except (StaleElementReferenceException, ElementClickInterceptedException, NoSuchElementException):
                        url_list["thumbnail_url"].append("Processing_problems")
                        url_list["image_url"].append("Processing_problems")
            except NoSuchElementException:
                pass

            try:
                back_btn = driver.find_element(By.CSS_SELECTOR, ".xZaYFf.VAyT2")
                from selenium.webdriver.common.keys import Keys as _Keys
                driver.find_element(By.TAG_NAME, "body").send_keys(_Keys.HOME)
                time.sleep(random.uniform(2, 5))
                back_btn.click()
                time.sleep(random.uniform(2, 5))
            except NoSuchElementException:
                pass

            thumbnails = _find_thumbnails(driver, num_images_main)

        except (StaleElementReferenceException, ElementClickInterceptedException, NoSuchElementException):
            url_list["thumbnail_url"].append("Processing_problems")
            url_list["image_url"].append("Processing_problems")

    driver.quit()

    min_len = min(len(url_list["thumbnail_url"]), len(url_list["image_url"]))
    for k in url_list:
        url_list[k] = url_list[k][:min_len]

    df = pd.DataFrame(url_list)
    os.makedirs(output_dir, exist_ok=True)
    df.to_excel(os.path.join(output_dir, f"{output_raw_filename}.xlsx"))
    logger.info("Saved raw URLs to %s/%s.xlsx", output_dir, output_raw_filename)
    return df


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _get_image_hash(url: str, retries: int = 3, timeout: int = 10):
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return imagehash.phash(img)
        except (requests.ConnectionError, requests.Timeout, RemoteDisconnected) as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(2)
            else:
                return None
        except Exception as exc:
            logger.warning("Error processing %s: %s", url, exc)
            return None
    return None


def _get_resolution(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img.size
    except Exception as exc:
        logger.warning("Resolution error for %s: %s", url, exc)
        return None


def dataset_purification(
    raw_df: pd.DataFrame,
    output_dir: str,
    output_clean_filename: str = "Enrichment_Cleaned",
) -> None:
    """Remove invalid/duplicate entries and save cleaned dataset."""
    df = raw_df.copy()
    df = df[df["image_url"] != "Processing_problems"]
    df = df.drop_duplicates(subset="image_url")
    df = df[df["image_url"] != "Not_found"].reset_index(drop=True)

    image_hashes = {}
    for _, row in df.iterrows():
        h = _get_image_hash(row["image_url"])
        if h is not None:
            image_hashes[row["image_url"]] = h

    urls = list(image_hashes.keys())
    hashes = list(image_hashes.values())
    duplicates = []
    for i, h1 in enumerate(hashes):
        for j, h2 in enumerate(hashes[i + 1:], i + 1):
            if h1 - h2 < 8:
                duplicates.append((urls[i], urls[j]))

    logger.info("Found %d near-duplicate pairs", len(duplicates))
    for url1, url2 in duplicates:
        if url1 in df["image_url"].values and url2 in df["image_url"].values:
            res1 = _get_resolution(url1)
            res2 = _get_resolution(url2)
            if res1 and res2:
                keep = url1 if (res1[0] * res1[1] >= res2[0] * res2[1]) else url2
                remove = url2 if keep == url1 else url1
                df = df[df["image_url"] != remove]
            else:
                df = df[df["image_url"] != url2]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{output_clean_filename}.xlsx")
    df.to_excel(out_path)
    logger.info("Saved cleaned data to %s", out_path)


def run_image_search_pipeline(
    query: str,
    output_dir: str,
    num_images_main: int = 100,
    num_images_similar: int = 100,
    output_raw: str = "Enrichment_Search_URLs",
    output_clean: str = "Enrichment_Cleaned",
) -> None:
    """Full pipeline: search → deduplicate → save."""
    raw_df = google_image_search(
        query=query,
        output_dir=output_dir,
        num_images_main=num_images_main,
        num_images_similar=num_images_similar,
        output_raw_filename=output_raw,
    )
    dataset_purification(raw_df, output_dir=output_dir, output_clean_filename=output_clean)
