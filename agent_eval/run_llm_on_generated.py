"""Run LLM-based END-TO-END table extraction on generated dataset and evaluate.

The LLM sees the FULL document page image + page-level OCR text.
It must independently locate all tables on the page and convert each to HTML.
No GT bboxes are provided to the LLM — this evaluates DETECTION + RECOGNITION.

LLM receives: sample image + sample HTML + target FULL PAGE image + OCR text
LLM returns: one or more <table>...</table> HTML fragments

Scoring: TEDS + GriTS, with greedy best-TEDS matching.
Scoring logic is identical to run_pipeline_on_generated_batch.py.

Run: python -u agent_eval/run_llm_on_generated.py
"""

import html as html_mod
import json
import os
import sys
import time
import re
import random
import base64
import asyncio
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES_PER_SUBSET = {
    "S1": 0,
    "S2": 0,
    "S3": 0,
    "S4": 500,
}

ATTR_FILTER = None              # Same syntax as run_pipeline_on_generated_batch.py

SEED = 42
SCORE_WORKERS = 16              # Threads for parallel TEDS + GriTS scoring

# ── LLM Configuration ──
MODEL_NAME = "gpt-5.2"           # Change to desired model: "gpt-4o", "gpt-4o-mini", "gpt-5.2", etc.
MAX_TOKENS = 8000               # Max completion tokens (need room for multi-table pages)
TEMPERATURE = 0.0               # Deterministic output
CONCURRENT_REQUESTS = 15        # Async concurrent LLM API calls
LLM_TIMEOUT = 120              # Per-image LLM call timeout in seconds (skip if exceeded)
API_BASE_URL = None             # Set to override OpenAI base URL (e.g. for Azure)
API_KEY = "YOUR_API_KEY_HERE"                    # Fill your OpenAI API key here

# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

GENERATED_DIR = os.path.join(PROJECT_ROOT, "data", "WildDocTables")
IMAGES_DIR = os.path.join(GENERATED_DIR, "images")
OCR_DIR = os.path.join(GENERATED_DIR, "ocr")
GT_JSON_V1_DIR = os.path.join(GENERATED_DIR, "gt_json_v1")
GT_JSON_V2_DIR = os.path.join(GENERATED_DIR, "gt_json_v2")
MANIFEST_PATH = os.path.join(GENERATED_DIR, "manifest.jsonl")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_llm_s4")

# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE HTML — reference example for one-shot prompting
# ═══════════════════════════════════════════════════════════════════════════════

# We use the first generated image as sample (will be loaded dynamically).
# If you prefer a fixed sample, set SAMPLE_IMAGE_ID to a specific image_id.
SAMPLE_IMAGE_ID = None  # None = auto-pick first image in dataset


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(OUTPUT_DIR, f"eval_llm_{timestamp}.log")

    logger = logging.getLogger("llm_eval")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger, log_path


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT (follows #11 original, with full-page multi-table addition)
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert table extraction system specialized in precise boundary detection.

**IMPORTANT — Full-Page Input**: The image you receive is a COMPLETE DOCUMENT PAGE, not a pre-cropped single table. The page may contain text paragraphs, figures, and **one or more tables** (typically 1–3). You must find and extract ALL tables on the page, outputting each as a separate HTML block in top-to-bottom order.

## HTML Format Requirements

The target HTML format follows these rules:

1. **Table Structure**:
   - `<table>` as root element
   - Each row is a `<tr>`, each cell is a `<td>` (NEVER use `<th>`)

2. **Content Formatting**:
   - Preserve rich text tags: `<b>`, `<i>`, `<sub>`, `<sup>`
   - Use HTML entities: `&gt;` for >, `&lt;` for <, `&amp;` for &
   - Empty cells: `<td></td>`
   - Text content should be clean and accurate

3. **Attributes**:
   - Use `colspan="N"` for cells spanning multiple columns
   - Use `rowspan="N"` for cells spanning multiple rows
   - Proper nesting: table > tr > td

4. **Formatting**:
   - Compact single-line format: `<table><tr><td>...</td></tr>...</table>`
   - No extra whitespace or newlines between tags

## Critical: Table Boundary Detection

**IMPORTANT**: Carefully identify the EXACT boundaries of each table:
- Only extract content that is CLEARLY INSIDE a table
- Do NOT include any text or elements outside the table borders
- Do NOT hallucinate or invent extra rows or columns
- If you're unsure about the text content of a cell, keep the cell structure and use `<td></td>` as a placeholder — do NOT delete rows or columns
- Count the exact number of rows and columns visible in the image before generating HTML

## Priority Order

1. **FIRST**: Get the table structure correct (exact number of rows and columns)
2. **SECOND**: Accurately recognize cell content
3. **THIRD**: Preserve formatting details (<b>, <i>, etc.)

**Remember**: A table with correct structure but minor text errors is better than a table with extra/missing rows or columns.

## Verification Steps

Before generating the HTML, verify:
1. **Row Count**: Count how many visible rows are in each table
2. **Column Count**: Count how many columns each table has
3. **Cell Matching**: Make sure each visible cell in the image has a corresponding <td> in your HTML
4. **No Extra Content**: Ensure you haven't added any rows or cells that don't exist in the image

## Common Mistakes to AVOID

DO NOT:
- Add extra rows beyond what's visible in the image
- Create cells for empty spaces outside the table
- Continue the table pattern beyond its actual boundary
- Duplicate rows or merge unrelated table regions
- Invent data that isn't clearly visible
- Treat paragraph text as table rows
- Merge two separate tables into one
- Split one table into two

DO:
- Stop exactly where each table ends
- Leave cells empty (`<td></td>`) if they appear empty in the image
- Match the exact structure you see
- Output each table as a separate `<table>...</table>` block

## Consistency Requirements

- **All rows must have the SAME number of `<td>` elements** (accounting for colspan)
- **The number of columns must be CONSISTENT throughout the entire table**
- **If a row appears shorter, use empty `<td></td>` cells to fill it**

## Output Format

- Return one or more bare `<table>...</table>` HTML fragments
- If there are multiple tables, output them in top-to-bottom order
- Do NOT include <html>, <head>, <body>, <thead>, <tbody> wrappers
- Use <td> for ALL cells (never <th>)
- Separate multiple tables with a blank line
"""


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def encode_image(path: str) -> str:
    """Base64 encode image for OpenAI API."""
    suffix = Path(path).suffix.lower()
    mime = "png" if suffix == ".png" else "jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


# ═══════════════════════════════════════════════════════════════════════════════
# HTML EXTRACTION FROM LLM RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

def extract_tables_from_response(text: str) -> List[str]:
    """Extract ALL <table>...</table> fragments from LLM response.

    Returns a list of HTML strings, one per table found.
    Handles code blocks, multiple tables, etc.
    """
    # First, unwrap code blocks if present
    code_blocks = re.findall(r"```(?:html)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        # Search for <table> in code blocks
        tables = []
        for block in code_blocks:
            tables.extend(re.findall(r"(<table[^>]*>.*?</table>)", block, re.DOTALL | re.IGNORECASE))
        if tables:
            return [t.strip() for t in tables]

    # Fallback: find all <table>...</table> in raw text
    tables = re.findall(r"(<table[^>]*>.*?</table>)", text, re.DOTALL | re.IGNORECASE)
    if tables:
        return [t.strip() for t in tables]

    return []


# ═══════════════════════════════════════════════════════════════════════════════
# LLM EXTRACTION (async, one-shot with OCR — end-to-end whole-page)
# ═══════════════════════════════════════════════════════════════════════════════

async def extract_table_llm(
    client,
    sample_img_url: str,
    sample_html: str,
    target_img_url: str,
    ocr_text: Optional[str] = None,
) -> str:
    """End-to-end table extraction via LLM (follows #11 prompt style).

    The LLM sees the FULL document page and must:
    1. Locate all tables on the page
    2. Convert each table to HTML
    3. Return them in top-to-bottom order
    """

    # ── User prompt — follows #11 original structure ──
    user_prompt_parts = [
        "**Sample Table (Reference Example)**:",
        "",
        "This is a reference example showing the standard HTML format you should follow. "
        "The sample is for FORMAT reference only — your output may have a different number of rows, columns, and tables.",
        "",
        "**Sample HTML**:",
        "",
        "```html",
        sample_html,
        "```",
        "",
        "---",
        "",
    ]

    if ocr_text:
        user_prompt_parts.extend([
            "**OCR Reference (for text accuracy)**:",
            "",
            "The following OCR text was extracted from the target document page. "
            "You can use it as a reference to improve text recognition accuracy:",
            "",
            "```",
            ocr_text,
            "```",
            "",
            "**Note**: The OCR text is provided as a reference to help you recognize "
            "text content more accurately. However, you should still rely primarily on "
            "the visual structure from the image for determining table layout, rows, "
            "columns, and cell boundaries.",
            "",
            "---",
            "",
        ])

    user_prompt_parts.extend([
        "**Your Task**:",
        "",
        "Extract ALL tables from the **target image** (the second image) and convert "
        "each to HTML following the same format as the sample above.",
        "",
        "**Before you generate the HTML**:",
        "1. First, determine how many tables are on the page — do NOT merge separate tables into one or split one table into two",
        "2. Count the exact number of rows in each table",
        "3. Count the exact number of columns in each table",
        "4. Identify where each table starts and ends",
        "5. Do NOT add any extra rows or columns beyond what you see",
    ])

    if ocr_text:
        user_prompt_parts.extend([
            "6. Use the OCR reference text above to help with accurate text recognition, "
            "but determine the table structure from the visual image.",
        ])

    user_prompt = "\n".join(user_prompt_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": sample_img_url}},
                {"type": "image_url", "image_url": {"url": target_img_url}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    content = response.choices[0].message.content
    if content is None:
        return ""
    return content.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# HTML NORMALIZATION & SCORING (identical to run_pipeline_on_generated_batch.py)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_html(html_str):
    """Normalize table HTML to canonical form: all <td>, no <thead>/<tbody>."""
    html_str = re.sub(r'<th\b', '<td', html_str)
    html_str = html_str.replace('</th>', '</td>')
    html_str = html_str.replace('<thead>', '<tr>')
    html_str = html_str.replace('</thead>', '</tr>')
    html_str = html_str.replace('<tbody>', '')
    html_str = html_str.replace('</tbody>', '')
    # Also strip <html>, <head>, <body> wrappers that LLM might add
    html_str = re.sub(r'</?html[^>]*>', '', html_str, flags=re.IGNORECASE)
    html_str = re.sub(r'<head>.*?</head>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    html_str = re.sub(r'</?body[^>]*>', '', html_str, flags=re.IGNORECASE)
    # Strip <style> blocks
    html_str = re.sub(r'<style>.*?</style>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    # Strip <meta> tags
    html_str = re.sub(r'<meta[^>]*/>', '', html_str, flags=re.IGNORECASE)
    # Fix bare <br> tags → self-closing <br/> for XML compatibility
    html_str = re.sub(r'<br\s*/?>', '<br/>', html_str, flags=re.IGNORECASE)
    # Replace HTML named entities with Unicode (XML parser only knows 5 entities)
    _XML_ENTITIES = {'amp', 'lt', 'gt', 'quot', 'apos'}
    html_str = re.sub(r'&([a-zA-Z]+);',
                      lambda m: m.group(0) if m.group(1) in _XML_ENTITIES
                      else html_mod.unescape(m.group(0)), html_str)
    return html_str.strip()


def wrap_for_teds(html_str):
    """Wrap <table> HTML with <html><body> for TEDS."""
    return "<html><body>" + html_str + "</body></html>"


def load_teds():
    teds_path = os.path.join(PROJECT_ROOT, "unitable", "unitable", "src", "utils")
    if teds_path not in sys.path:
        sys.path.insert(0, teds_path)
    from teds import TEDS
    return TEDS(n_jobs=1, ignore_nodes="", structure_only=False)


def load_grits():
    from grits import grits_from_html
    return grits_from_html


def compute_scores(pred_html, gt_html, teds_scorer, grits_fn):
    scores = {}
    pred_norm = normalize_html(pred_html)
    gt_norm = normalize_html(gt_html)

    pred_for_teds = wrap_for_teds(pred_norm)
    gt_for_teds = wrap_for_teds(gt_norm)
    try:
        scores["teds"] = teds_scorer.evaluate(pred_for_teds, gt_for_teds)
    except Exception as e:
        scores["teds"] = 0.0
        scores["teds_error"] = str(e)

    try:
        grits_metrics = grits_fn(gt_norm, pred_norm)
        scores["grits_con"] = grits_metrics["grits_con"]
        scores["grits_top"] = grits_metrics["grits_top"]
    except Exception as e:
        scores["grits_con"] = 0.0
        scores["grits_top"] = 0.0
        scores["grits_error"] = str(e)

    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# GREEDY TEDS-BASED MATCHING (no bboxes — LLM does end-to-end extraction)
# ═══════════════════════════════════════════════════════════════════════════════

def greedy_match_tables_by_teds(pred_htmls: List[str], gt_htmls: List[str],
                                teds_matrix: dict) -> List[tuple]:
    """Greedy best-match: for each pred table, find the GT with highest TEDS.

    teds_matrix: dict mapping (pred_idx, gt_idx) -> teds_score (float).

    Algorithm: iterate pred tables in order (0,1,...N-1). For each pred,
    pick the GT with the highest TEDS score from the remaining GT pool.
    Remove that GT from the pool, then move to the next pred.

    Returns list of (pred_idx, gt_idx) tuples (length = min(N, M)).
    """
    N = len(pred_htmls)
    M = len(gt_htmls)
    available_gt = set(range(M))
    matches = []

    for pi in range(N):
        if not available_gt:
            break
        best_gj = max(available_gt, key=lambda gj: teds_matrix.get((pi, gj), 0.0))
        matches.append((pi, best_gj))
        available_gt.remove(best_gj)

    return matches


# ═══════════════════════════════════════════════════════════════════════════════
# OCR LOADING (full-page)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_ocr_text(img_id: str) -> Optional[str]:
    """Load full-page OCR text for an image."""
    ocr_path = os.path.join(OCR_DIR, f"{img_id}_ocr.json")
    if not os.path.exists(ocr_path):
        return None
    try:
        with open(ocr_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
        lines = ocr_data.get("ocr", [])
        if not lines:
            return None
        texts = [entry["text"] for entry in lines]
        return "\n".join(texts)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ATTRIBUTE FILTER (identical)
# ═══════════════════════════════════════════════════════════════════════════════

def image_matches_filter(manifest_rec, attr_filter):
    if attr_filter is None:
        return True
    tables = manifest_rec["tables"]

    if "T1" in attr_filter:
        want = attr_filter["T1"]
        has_t1_yes = any(t["T1"] == "Yes" for t in tables)
        if want == "Yes" and not has_t1_yes:
            return False
        if want == "No" and has_t1_yes:
            return False

    if "T3" in attr_filter:
        want = attr_filter["T3"]
        page_t3 = manifest_rec["T3"]
        if want == "None" and page_t3 != "None":
            return False
        if want == "degraded" and page_t3 == "None":
            return False
        if want in ("Watermark", "Stain", "Stamp") and page_t3 != want:
            return False

    if "T4" in attr_filter:
        want = attr_filter["T4"]
        has_not_full = any(t["T4"] != "Full" for t in tables)
        if want == "Full" and has_not_full:
            return False
        if want == "not_Full" and not has_not_full:
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    logger, log_path = setup_logging()

    logger.info("=" * 70)
    logger.info("LLM-based Table Extraction Evaluation")
    logger.info("=" * 70)
    total_samples = sum(v for v in SAMPLES_PER_SUBSET.values() if v is not None)
    logger.info(f"  SAMPLES/SUBSET   = {SAMPLES_PER_SUBSET}  (total={total_samples})")
    logger.info(f"  ATTR_FILTER      = {ATTR_FILTER}")
    logger.info(f"  MODEL_NAME       = {MODEL_NAME}")
    logger.info(f"  MAX_TOKENS       = {MAX_TOKENS}")
    logger.info(f"  TEMPERATURE      = {TEMPERATURE}")
    logger.info(f"  CONCURRENT_REQ   = {CONCURRENT_REQUESTS}")
    logger.info(f"  LLM_TIMEOUT      = {LLM_TIMEOUT}s")
    logger.info(f"  SCORE_WORKERS    = {SCORE_WORKERS}")
    logger.info(f"  SEED             = {SEED}")
    logger.info(f"  OUTPUT_DIR       = {OUTPUT_DIR}")
    logger.info("")

    # ── Load OpenAI client ──
    api_key = API_KEY
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("API_KEY not set and OPENAI_API_KEY not found in environment")
        sys.exit(1)

    from openai import AsyncOpenAI
    client_kwargs = {"api_key": api_key}
    if API_BASE_URL:
        client_kwargs["base_url"] = API_BASE_URL
    client = AsyncOpenAI(**client_kwargs)

    # ── Load scorers ──
    logger.info("Loading scorers (TEDS + GriTS)...")
    teds_scorer = load_teds()
    grits_fn = load_grits()
    logger.info("Scorers ready.")

    # ── Build subset index from manifest ──
    logger.info(f"Reading manifest: {MANIFEST_PATH}")
    subset_to_ids = {}
    id_to_attrs = {}
    total_manifest = 0
    filtered_out = 0

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total_manifest += 1
            if not image_matches_filter(rec, ATTR_FILTER):
                filtered_out += 1
                continue
            s = rec["subset"]
            img_id = rec["image_id"]
            subset_to_ids.setdefault(s, []).append(img_id)

            tables = rec["tables"]
            has_t1_yes = any(t["T1"] == "Yes" for t in tables)
            has_t3_deg = rec["T3"] != "None"
            has_t4_not_full = any(t["T4"] != "Full" for t in tables)
            id_to_attrs[img_id] = {
                "T1": "Yes" if has_t1_yes else "No",
                "T3": "degraded" if has_t3_deg else "None",
                "T4": "not_Full" if has_t4_not_full else "Full",
            }

    for s in subset_to_ids:
        subset_to_ids[s].sort()
    passed = total_manifest - filtered_out
    logger.info(f"Manifest: {total_manifest} total, {passed} match filter, {filtered_out} filtered out")

    # ── Per-subset sampling ──
    rng = random.Random(SEED)
    sampled_ids = []
    for subset_name in sorted(SAMPLES_PER_SUBSET.keys()):
        n_want = SAMPLES_PER_SUBSET[subset_name]
        available = subset_to_ids.get(subset_name, [])
        if not available:
            logger.warning(f"  Subset {subset_name}: no images, skipping")
            continue
        if n_want is None or n_want >= len(available):
            chosen = available
        else:
            chosen = rng.sample(available, n_want)
            chosen.sort()
        for img_id in chosen:
            img_path = os.path.join(IMAGES_DIR, f"{img_id}.png")
            if os.path.exists(img_path):
                sampled_ids.append(img_id)
        logger.info(f"  {subset_name}: {len(chosen)} sampled from {len(available)} available")

    sampled_ids.sort()
    logger.info(f"Total sampled: {len(sampled_ids)} images")

    if not sampled_ids:
        logger.error("No images to process!")
        return

    # ── Load sample image + GT HTML for one-shot reference ──
    sample_id = SAMPLE_IMAGE_ID or sampled_ids[0]
    sample_img_path = os.path.join(IMAGES_DIR, f"{sample_id}.png")
    sample_gt_v2_path = os.path.join(GT_JSON_V2_DIR, f"{sample_id}_v2.json")

    with open(sample_gt_v2_path, "r", encoding="utf-8") as f:
        sample_gt_v2 = json.load(f)
    sample_tables = [e for e in sample_gt_v2.get("page", []) if e["type"] == "table"]
    sample_html = sample_tables[0]["html"] if sample_tables else "<table><tr><td></td></tr></table>"

    sample_img_url = encode_image(sample_img_path)
    logger.info(f"Sample image: {sample_id} (HTML len={len(sample_html)})")

    # ── Phase 1: Load GT data for all images ──
    logger.info(f"\nPhase 1: Loading GT data for {len(sampled_ids)} images...")
    t_preload = time.time()

    image_data = []  # list of dicts with all needed info
    preload_errors = 0

    for img_id in sampled_ids:
        try:
            gt_v1_path = os.path.join(GT_JSON_V1_DIR, f"{img_id}.json")
            gt_v2_path = os.path.join(GT_JSON_V2_DIR, f"{img_id}_v2.json")

            with open(gt_v1_path, "r", encoding="utf-8") as f:
                gt_v1 = json.load(f)
            with open(gt_v2_path, "r", encoding="utf-8") as f:
                gt_v2 = json.load(f)

            subset = gt_v1.get("subset", "unknown")
            gt_tables_v1 = [e for e in gt_v1.get("page", []) if e["type"] == "table"]
            gt_tables_v2 = [e for e in gt_v2.get("page", []) if e["type"] == "table"]

            gt_info = []
            for ti, t_v1 in enumerate(gt_tables_v1):
                t_v2 = gt_tables_v2[ti] if ti < len(gt_tables_v2) else {}
                gt_info.append({
                    "bbox": t_v1.get("bbox", [0, 0, 0, 0]),
                    "html": t_v2.get("html", ""),
                    "attributes": t_v1.get("attributes", {}),
                })

            image_data.append({
                "image_id": img_id,
                "subset": subset,
                "gt_info": gt_info,
                "num_gt_tables": len(gt_tables_v1),
            })
        except Exception as e:
            image_data.append({"image_id": img_id, "error": str(e)})
            preload_errors += 1

    logger.info(f"Preloading done in {time.time() - t_preload:.1f}s ({preload_errors} errors)")

    # ── Phase 2: LLM extraction (async) ──
    # Send FULL PAGE image + full OCR to LLM. One API call per image.
    # LLM finds all tables and returns multiple <table>...</table> blocks.
    logger.info(f"\nPhase 2: LLM extraction ({CONCURRENT_REQUESTS} concurrent)...")
    t_extract = time.time()

    extraction_tasks = []  # (data_idx, img_id, img_path, ocr_text)
    for di, data in enumerate(image_data):
        if "error" in data:
            continue
        img_id = data["image_id"]
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.png")
        ocr_text = _load_ocr_text(img_id)
        extraction_tasks.append((di, img_id, img_path, ocr_text))

    # Results: data_idx -> list of pred HTML strings (one per table found)
    llm_results = {}  # di -> [html_str, ...]
    llm_errors = 0
    llm_timeouts = 0
    timeout_ids = []  # image_ids that timed out
    timeout_di_set = set()  # data indices that timed out
    total_images = len(extraction_tasks)
    logger.info(f"  Total images to process: {total_images}")
    logger.info(f"  LLM timeout: {LLM_TIMEOUT}s per image")

    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    extract_done = 0
    t_extract_start = time.time()

    async def _extract_one(di, img_id, img_path, ocr_text):
        nonlocal extract_done, llm_errors, llm_timeouts
        async with sem:
            t_img_start = time.time()
            try:
                target_img_url = encode_image(img_path)
                response_text = await asyncio.wait_for(
                    extract_table_llm(
                        client, sample_img_url, sample_html,
                        target_img_url, ocr_text,
                    ),
                    timeout=LLM_TIMEOUT,
                )
                pred_tables = extract_tables_from_response(response_text)
                llm_results[di] = pred_tables
            except asyncio.TimeoutError:
                llm_results[di] = []
                llm_timeouts += 1
                timeout_ids.append(img_id)
                timeout_di_set.add(di)
                t_elapsed = time.time() - t_img_start
                logger.warning(f"  TIMEOUT for {img_id} after {t_elapsed:.1f}s (limit={LLM_TIMEOUT}s)")
            except Exception as e:
                llm_results[di] = []
                llm_errors += 1
                logger.warning(f"  LLM error for {img_id}: {e}")

            extract_done += 1
            if extract_done % 50 == 0 or extract_done == total_images:
                elapsed = time.time() - t_extract_start
                rate = extract_done / elapsed if elapsed > 0 else 0
                remaining = total_images - extract_done
                eta = remaining / rate if rate > 0 else 0
                logger.info(f"  Extracted {extract_done}/{total_images} images "
                            f"| {elapsed:.1f}s elapsed | {rate:.1f} img/s | ETA {eta:.1f}s "
                            f"| errors={llm_errors} timeouts={llm_timeouts}")

    tasks = [_extract_one(di, img_id, img_path, ocr_text)
             for di, img_id, img_path, ocr_text in extraction_tasks]
    await asyncio.gather(*tasks)

    t_extract_total = time.time() - t_extract
    total_pred_tables = sum(len(v) for v in llm_results.values())
    logger.info(f"LLM extraction done in {t_extract_total:.1f}s "
                f"({total_images} images, {total_pred_tables} tables found, "
                f"{llm_errors} errors, {llm_timeouts} timeouts)")
    if timeout_ids:
        logger.info(f"  Timed-out images: {timeout_ids[:20]}"
                     + (f" ... and {len(timeout_ids)-20} more" if len(timeout_ids) > 20 else ""))

    # ── Phase 3: TEDS-based Greedy Matching + Scoring ──
    # Step 1: Compute TEDS for all N×M pred-GT pairs (used for matching)
    # Step 2: Greedy match each pred to best-scoring GT by TEDS
    # Step 3: Compute GriTS for matched pairs only

    logger.info(f"\nPhase 3: Matching + Scoring...")
    t_score_start = time.time()

    all_results = []
    image_teds = []
    image_grits_con = []
    image_grits_top = []
    error_count = 0
    no_det_count = 0
    timeout_count = 0

    # Collect per-image info for matching
    # image_match_info: list of (di, img_id, subset, M, N, pred_htmls, gt_info, gt_htmls)
    image_match_info = []

    for di, data in enumerate(image_data):
        img_id = data["image_id"]
        subset = data.get("subset", "unknown")

        if "error" in data:
            all_results.append({
                "image_id": img_id, "status": "error",
                "error": data["error"], "subset": subset,
            })
            error_count += 1
            continue

        M = data["num_gt_tables"]
        gt_info = data["gt_info"]
        gt_htmls = [g["html"] for g in gt_info]

        pred_htmls = llm_results.get(di, [])
        N = len(pred_htmls)

        if N == 0:
            is_timeout = di in timeout_di_set
            status = "timeout" if is_timeout else "no_detection"
            result = {
                "image_id": img_id, "subset": subset,
                "status": status,
                "num_tables_detected": 0, "num_tables_gt": M,
                "penalized_teds": 0.0,
                "penalized_grits_con": 0.0,
                "penalized_grits_top": 0.0,
            }
            all_results.append(result)
            if is_timeout:
                timeout_count += 1
            else:
                no_det_count += 1
            image_teds.append(0.0)
            image_grits_con.append(0.0)
            image_grits_top.append(0.0)
            continue

        result_idx = len(all_results)
        all_results.append(None)  # placeholder
        image_match_info.append({
            "img_id": img_id, "subset": subset,
            "M": M, "N": N,
            "pred_htmls": pred_htmls, "gt_info": gt_info, "gt_htmls": gt_htmls,
            "result_idx": result_idx,
        })

    # ── Step 1: Compute TEDS for ALL N×M pairs (for matching) ──
    # Build tasks: (info_idx, pi, gj, pred_html, gt_html)
    teds_tasks = []
    for info_idx, info in enumerate(image_match_info):
        for pi in range(info["N"]):
            for gj in range(info["M"]):
                teds_tasks.append((info_idx, pi, gj,
                                   info["pred_htmls"][pi], info["gt_htmls"][gj]))

    logger.info(f"  Step 1: Computing TEDS for {len(teds_tasks)} pred-GT pairs "
                f"(N×M matching) with {SCORE_WORKERS} threads...")

    _thread_local = threading.local()

    def _get_thread_teds():
        if not hasattr(_thread_local, 'teds'):
            _thread_local.teds = load_teds()
        return _thread_local.teds

    # teds_all[(info_idx, pi, gj)] = teds_score
    teds_all = {}

    def _teds_one(task):
        info_idx, pi, gj, pred_html, gt_html = task
        if not pred_html or not gt_html:
            return (info_idx, pi, gj, 0.0)
        thread_teds = _get_thread_teds()
        pred_norm = normalize_html(pred_html)
        gt_norm = normalize_html(gt_html)
        try:
            score = thread_teds.evaluate(wrap_for_teds(pred_norm), wrap_for_teds(gt_norm))
        except Exception:
            score = 0.0
        return (info_idx, pi, gj, score)

    t_teds_start = time.time()
    total_teds_tasks = len(teds_tasks)

    with ThreadPoolExecutor(max_workers=SCORE_WORKERS) as pool:
        futures = {pool.submit(_teds_one, task): task for task in teds_tasks}
        done_count = 0
        for future in as_completed(futures):
            info_idx, pi, gj, teds_score = future.result()
            teds_all[(info_idx, pi, gj)] = teds_score
            done_count += 1
            if done_count % 20 == 0 or done_count == total_teds_tasks:
                elapsed = time.time() - t_teds_start
                rate = done_count / elapsed if elapsed > 0 else 0
                remaining = total_teds_tasks - done_count
                eta = remaining / rate if rate > 0 else 0
                logger.info(f"  TEDS matching: {done_count}/{total_teds_tasks} pairs "
                            f"| {elapsed:.1f}s elapsed | {rate:.1f} pairs/s | ETA {eta:.1f}s")

    t_teds_total = time.time() - t_teds_start
    if total_teds_tasks > 0:
        logger.info(f"  Step 1 done: {total_teds_tasks} TEDS pairs in {t_teds_total:.1f}s")

    # ── Step 2: Greedy matching by TEDS ──
    logger.info(f"  Step 2: Greedy matching by TEDS...")
    per_image_meta = []

    for info_idx, info in enumerate(image_match_info):
        # Build per-image TEDS matrix
        teds_matrix = {}
        for pi in range(info["N"]):
            for gj in range(info["M"]):
                teds_matrix[(pi, gj)] = teds_all.get((info_idx, pi, gj), 0.0)

        matches = greedy_match_tables_by_teds(
            info["pred_htmls"], info["gt_htmls"], teds_matrix)

        info["matches"] = matches
        info["teds_matrix"] = teds_matrix
        per_image_meta.append(info)

    # ── Step 3: Compute GriTS for matched pairs only ──
    grits_tasks = []  # (info_idx, pi, gj, pred_html, gt_html)
    for info_idx, info in enumerate(per_image_meta):
        for pi, gj in info["matches"]:
            grits_tasks.append((info_idx, pi, gj,
                                info["pred_htmls"][pi], info["gt_htmls"][gj]))

    logger.info(f"  Step 3: Computing GriTS for {len(grits_tasks)} matched pairs "
                f"with {SCORE_WORKERS} threads...")

    # grits_all[(info_idx, pi, gj)] = {"grits_con": ..., "grits_top": ...}
    grits_all = {}

    _grits_empty_count = [0]
    _grits_ok_count = [0]

    def _grits_one(task):
        info_idx, pi, gj, pred_html, gt_html = task
        if not pred_html or not gt_html:
            _grits_empty_count[0] += 1
            if _grits_empty_count[0] <= 5:
                logger.warning(f"GriTS skip empty [{info_idx}] pred={pi} gt={gj}: "
                               f"pred_html={'(empty)' if not pred_html else '(ok)'} "
                               f"gt_html={'(empty)' if not gt_html else '(ok)'}")
            return (info_idx, pi, gj, {"grits_con": 0.0, "grits_top": 0.0})
        pred_norm = normalize_html(pred_html)
        gt_norm = normalize_html(gt_html)
        try:
            grits_metrics = grits_fn(gt_norm, pred_norm)
            _grits_ok_count[0] += 1
            return (info_idx, pi, gj, {
                "grits_con": grits_metrics["grits_con"],
                "grits_top": grits_metrics["grits_top"],
            })
        except Exception as e:
            if not hasattr(_grits_one, '_err_count'):
                _grits_one._err_count = 0
            _grits_one._err_count += 1
            if _grits_one._err_count <= 10:
                logger.warning(f"GriTS error [{info_idx}] pred={pi} gt={gj}: "
                               f"{type(e).__name__}: {e}")
            return (info_idx, pi, gj, {"grits_con": 0.0, "grits_top": 0.0})

    t_grits_start = time.time()
    total_grits_tasks = len(grits_tasks)

    with ThreadPoolExecutor(max_workers=SCORE_WORKERS) as pool:
        futures = {pool.submit(_grits_one, task): task for task in grits_tasks}
        done_count = 0
        for future in as_completed(futures):
            info_idx, pi, gj, grits_scores = future.result()
            grits_all[(info_idx, pi, gj)] = grits_scores
            done_count += 1
            if done_count % 20 == 0 or done_count == total_grits_tasks:
                elapsed = time.time() - t_grits_start
                rate = done_count / elapsed if elapsed > 0 else 0
                remaining = total_grits_tasks - done_count
                eta = remaining / rate if rate > 0 else 0
                logger.info(f"  GriTS scoring: {done_count}/{total_grits_tasks} pairs "
                            f"| {elapsed:.1f}s elapsed | {rate:.1f} pairs/s | ETA {eta:.1f}s")

    t_grits_total = time.time() - t_grits_start
    _err_total = getattr(_grits_one, '_err_count', 0)
    logger.info(f"  GriTS summary: {_grits_ok_count[0]} ok, {_grits_empty_count[0]} empty, "
                f"{_err_total} errors out of {total_grits_tasks} total")
    if total_grits_tasks > 0:
        logger.info(f"  Step 3 done: {total_grits_tasks} GriTS pairs in {t_grits_total:.1f}s")

    # Merge TEDS + GriTS into score_results for downstream assembly
    score_results = {}
    for info_idx, info in enumerate(per_image_meta):
        result_idx = info["result_idx"]
        for pi, gj in info["matches"]:
            teds_val = teds_all.get((info_idx, pi, gj), 0.0)
            grits = grits_all.get((info_idx, pi, gj), {"grits_con": 0.0, "grits_top": 0.0})
            score_results[(result_idx, pi, gj)] = {
                "teds": teds_val,
                "grits_con": grits["grits_con"],
                "grits_top": grits["grits_top"],
            }

    t_scoring_total = time.time() - t_score_start
    logger.info(f"  Phase 3 total: {t_scoring_total:.1f}s")

    # ── Assemble final results ──
    for meta in per_image_meta:
        img_id = meta["img_id"]
        subset = meta["subset"]
        M, N = meta["M"], meta["N"]
        matches = meta["matches"]
        pred_htmls = meta["pred_htmls"]
        gt_info = meta["gt_info"]
        result_idx = meta["result_idx"]

        table_matches = []
        pair_teds, pair_grits_con, pair_grits_top = [], [], []

        for pi, gj in matches:
            scores = score_results.get((result_idx, pi, gj),
                                       {"teds": 0.0, "grits_con": 0.0, "grits_top": 0.0})
            table_matches.append({
                "pred_idx": pi, "gt_idx": gj,
                "pred_html": pred_htmls[pi],
                "gt_html": gt_info[gj]["html"],
                "scores": scores,
            })
            if scores.get("teds") is not None:
                pair_teds.append(scores["teds"])
            if scores.get("grits_con") is not None:
                pair_grits_con.append(scores["grits_con"])
            if scores.get("grits_top") is not None:
                pair_grits_top.append(scores["grits_top"])

        if M == 0 and N == 0:
            penalty = 1.0
        elif M == 0 or N == 0:
            penalty = 0.0
        else:
            penalty = min(M, N) / max(M, N)

        avg_teds = sum(pair_teds) / len(pair_teds) if pair_teds else 0.0
        avg_gc = sum(pair_grits_con) / len(pair_grits_con) if pair_grits_con else 0.0
        avg_gt = sum(pair_grits_top) / len(pair_grits_top) if pair_grits_top else 0.0

        result = {
            "image_id": img_id, "subset": subset, "status": "ok",
            "num_tables_detected": N, "num_tables_gt": M,
            "penalty": round(penalty, 4),
            "table_matches": table_matches,
            "raw_avg_teds": round(avg_teds, 6),
            "raw_avg_grits_con": round(avg_gc, 6),
            "raw_avg_grits_top": round(avg_gt, 6),
            "penalized_teds": round(avg_teds * penalty, 6),
            "penalized_grits_con": round(avg_gc * penalty, 6),
            "penalized_grits_top": round(avg_gt * penalty, 6),
        }
        all_results[result_idx] = result
        image_teds.append(result["penalized_teds"])
        image_grits_con.append(result["penalized_grits_con"])
        image_grits_top.append(result["penalized_grits_top"])

    # Attach attribute flags
    for r in all_results:
        if r is None:
            continue
        img_id = r.get("image_id", "")
        if img_id in id_to_attrs:
            r["attrs"] = id_to_attrs[img_id]
        # Find matching image_data entry for gt_table_attrs
        for data in image_data:
            if data.get("image_id") == img_id and "error" not in data:
                r["gt_table_attrs"] = [g.get("attributes", {}) for g in data["gt_info"]]
                break

    t_score_total = time.time() - t_score_start
    t_total = time.time() - t_preload

    # ══════════════════════════════════════════════════════════════════════════
    # STATISTICS — identical to run_pipeline_on_generated_batch.py
    # ══════════════════════════════════════════════════════════════════════════

    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total images processed:  {len(image_data)}")
    logger.info(f"  Successful:              {len(image_data) - error_count - timeout_count}")
    logger.info(f"  No table detected:       {no_det_count}")
    logger.info(f"  Timeouts:                {timeout_count}")
    logger.info(f"  Errors:                  {error_count}")
    logger.info("")

    def log_stats(name, scores):
        if not scores:
            return
        s = sorted(scores)
        n = len(s)
        avg = sum(s) / n
        median = s[n // 2]
        p25 = s[n // 4] if n >= 4 else s[0]
        p75 = s[3 * n // 4] if n >= 4 else s[-1]
        logger.info(f"  {name}:")
        logger.info(f"    mean={avg:.4f}  median={median:.4f}  "
                     f"p25={p25:.4f}  p75={p75:.4f}  "
                     f"min={s[0]:.4f}  max={s[-1]:.4f}  n={n}")

    log_stats("TEDS (penalized)", image_teds)
    log_stats("GriTS_Con (penalized)", image_grits_con)
    log_stats("GriTS_Top (penalized)", image_grits_top)

    logger.info("")
    logger.info("Timing:")
    logger.info(f"  LLM extraction:  {t_extract_total:.1f}s")
    logger.info(f"  Scoring:         {t_score_total:.1f}s")
    logger.info(f"  Wall clock:      {t_total:.1f}s")

    # Detection accuracy (exact count match)
    gt_counts = [r.get("num_tables_gt", 0) for r in all_results if r and r.get("status") not in ("error",)]
    det_counts = [r.get("num_tables_detected", 0) for r in all_results if r and r.get("status") not in ("error",)]
    exact_match = 0
    if gt_counts:
        exact_match = sum(1 for g, d in zip(gt_counts, det_counts) if g == d)
        logger.info("")
        logger.info("Detection accuracy:")
        logger.info(f"  Exact count match: {exact_match}/{len(gt_counts)} "
                     f"({100*exact_match/len(gt_counts):.1f}%)")
        logger.info(f"  Total GT tables:   {sum(gt_counts)}")
        logger.info(f"  Total detected:    {sum(det_counts)}")

    # ── Per-subset statistics ──
    subset_scores = defaultdict(lambda: {"teds": [], "grits_con": [], "grits_top": [],
                                          "count": 0, "no_det": 0, "timeouts": 0, "errors": 0})
    for r in all_results:
        if r is None:
            continue
        s = r.get("subset", "unknown")
        subset_scores[s]["count"] += 1
        st = r.get("status", "error")
        if st == "error":
            subset_scores[s]["errors"] += 1
        elif st == "timeout":
            subset_scores[s]["timeouts"] += 1
            subset_scores[s]["teds"].append(0.0)
            subset_scores[s]["grits_con"].append(0.0)
            subset_scores[s]["grits_top"].append(0.0)
        elif st == "no_detection":
            subset_scores[s]["no_det"] += 1
            subset_scores[s]["teds"].append(0.0)
            subset_scores[s]["grits_con"].append(0.0)
            subset_scores[s]["grits_top"].append(0.0)
        else:
            subset_scores[s]["teds"].append(r["penalized_teds"])
            subset_scores[s]["grits_con"].append(r["penalized_grits_con"])
            subset_scores[s]["grits_top"].append(r["penalized_grits_top"])

    if len(subset_scores) > 1 or "unknown" not in subset_scores:
        logger.info("")
        logger.info("Per-subset breakdown:")
        logger.info(f"  {'Subset':<8} {'Count':>6} {'NoDet':>6} {'TMOut':>6} {'Err':>5} "
                     f"{'TEDS':>8} {'GriTS_C':>8} {'GriTS_T':>8}")
        logger.info(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
        for s_name in sorted(subset_scores.keys()):
            ss = subset_scores[s_name]
            t_avg = sum(ss["teds"]) / len(ss["teds"]) if ss["teds"] else 0
            gc_avg = sum(ss["grits_con"]) / len(ss["grits_con"]) if ss["grits_con"] else 0
            gt_avg = sum(ss["grits_top"]) / len(ss["grits_top"]) if ss["grits_top"] else 0
            logger.info(f"  {s_name:<8} {ss['count']:>6} {ss['no_det']:>6} {ss['timeouts']:>6} {ss['errors']:>5} "
                         f"{t_avg:>8.4f} {gc_avg:>8.4f} {gt_avg:>8.4f}")

    # ── Per-attribute difficulty statistics ──
    attr_dimensions = [
        ("T1", ["Yes", "No"]),
        ("T3", ["degraded", "None"]),
        ("T4", ["not_Full", "Full"]),
    ]
    attr_scores = {}
    for r in all_results:
        if r is None:
            continue
        attrs = r.get("attrs")
        if not attrs:
            continue
        st = r.get("status", "error")
        for attr_name, _ in attr_dimensions:
            val = attrs[attr_name]
            key = (attr_name, val)
            if key not in attr_scores:
                attr_scores[key] = {"teds": [], "grits_con": [], "grits_top": [],
                                     "count": 0, "no_det": 0, "timeouts": 0, "errors": 0}
            bucket = attr_scores[key]
            bucket["count"] += 1
            if st == "error":
                bucket["errors"] += 1
            elif st == "timeout":
                bucket["timeouts"] += 1
                bucket["teds"].append(0.0)
                bucket["grits_con"].append(0.0)
                bucket["grits_top"].append(0.0)
            elif st == "no_detection":
                bucket["no_det"] += 1
                bucket["teds"].append(0.0)
                bucket["grits_con"].append(0.0)
                bucket["grits_top"].append(0.0)
            else:
                bucket["teds"].append(r["penalized_teds"])
                bucket["grits_con"].append(r["penalized_grits_con"])
                bucket["grits_top"].append(r["penalized_grits_top"])

    if attr_scores:
        logger.info("")
        logger.info("Per-attribute difficulty breakdown (image-level):")
        logger.info(f"  {'Attribute':<14} {'Count':>6} {'NoDet':>6} {'TMOut':>6} {'Err':>5} "
                     f"{'TEDS':>8} {'GriTS_C':>8} {'GriTS_T':>8}")
        logger.info(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
        for attr_name, values in attr_dimensions:
            for val in values:
                key = (attr_name, val)
                if key not in attr_scores:
                    continue
                b = attr_scores[key]
                t_avg = sum(b["teds"]) / len(b["teds"]) if b["teds"] else 0
                gc_avg = sum(b["grits_con"]) / len(b["grits_con"]) if b["grits_con"] else 0
                gt_avg = sum(b["grits_top"]) / len(b["grits_top"]) if b["grits_top"] else 0
                label = f"{attr_name}={val}"
                logger.info(f"  {label:<14} {b['count']:>6} {b['no_det']:>6} {b['timeouts']:>6} {b['errors']:>5} "
                             f"{t_avg:>8.4f} {gc_avg:>8.4f} {gt_avg:>8.4f}")

    # ── Per-table attribute statistics ──
    table_attr_dimensions = [
        ("T1", ["Yes", "No"]),
        ("T2", [1, 2, 3, 4, 5]),
        ("T3", ["Watermark", "Stain", "Stamp", "None"]),
        ("T4", ["Full"]),
    ]
    table_attr_scores = {}

    def _tbl_attr_bucket(key):
        if key not in table_attr_scores:
            table_attr_scores[key] = {"teds": [], "grits_con": [], "grits_top": [], "count": 0}
        return table_attr_scores[key]

    def _classify_table(ta, attr_name):
        if attr_name == "T4":
            return ta.get("T4", "Full") if ta.get("T4") == "Full" else "not_Full"
        elif attr_name == "T2":
            return ta.get("T2", 1)
        else:
            return ta.get(attr_name, "")

    for r in all_results:
        if r is None:
            continue
        gt_ta_list = r.get("gt_table_attrs")
        if not gt_ta_list:
            continue
        st = r.get("status", "error")
        if st == "error":
            continue

        if st in ("no_detection", "timeout"):
            for ta in gt_ta_list:
                for attr_name, _ in table_attr_dimensions:
                    val = _classify_table(ta, attr_name)
                    bucket = _tbl_attr_bucket((attr_name, val))
                    bucket["count"] += 1
                    bucket["teds"].append(0.0)
                    bucket["grits_con"].append(0.0)
                    bucket["grits_top"].append(0.0)
        else:
            matched_gt_indices = set()
            for m in r.get("table_matches", []):
                gi_idx = m["gt_idx"]
                matched_gt_indices.add(gi_idx)
                scores = m.get("scores", {})
                teds_val = scores.get("teds", 0.0) or 0.0
                gc_val = scores.get("grits_con", 0.0) or 0.0
                gt_val = scores.get("grits_top", 0.0) or 0.0
                if gi_idx < len(gt_ta_list):
                    ta = gt_ta_list[gi_idx]
                    for attr_name, _ in table_attr_dimensions:
                        val = _classify_table(ta, attr_name)
                        bucket = _tbl_attr_bucket((attr_name, val))
                        bucket["count"] += 1
                        bucket["teds"].append(teds_val)
                        bucket["grits_con"].append(gc_val)
                        bucket["grits_top"].append(gt_val)

            for gi_idx, ta in enumerate(gt_ta_list):
                if gi_idx in matched_gt_indices:
                    continue
                for attr_name, _ in table_attr_dimensions:
                    val = _classify_table(ta, attr_name)
                    bucket = _tbl_attr_bucket((attr_name, val))
                    bucket["count"] += 1
                    bucket["teds"].append(0.0)
                    bucket["grits_con"].append(0.0)
                    bucket["grits_top"].append(0.0)

    print_order = [
        ("T1", "Yes"), ("T1", "No"),
        ("T2", 1), ("T2", 2), ("T2", 3), ("T2", 4), ("T2", 5),
        ("T3", "Watermark"), ("T3", "Stain"), ("T3", "Stamp"), ("T3", "None"),
        ("T4", "not_Full"), ("T4", "Full"),
    ]

    if table_attr_scores:
        logger.info("")
        logger.info("Per-table attribute breakdown (table-level scores):")
        logger.info(f"  {'Attribute':<14} {'Tables':>7} "
                     f"{'TEDS':>8} {'GriTS_C':>8} {'GriTS_T':>8}")
        logger.info(f"  {'-'*14} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
        for key in print_order:
            if key not in table_attr_scores:
                continue
            b = table_attr_scores[key]
            t_avg = sum(b["teds"]) / len(b["teds"]) if b["teds"] else 0
            gc_avg = sum(b["grits_con"]) / len(b["grits_con"]) if b["grits_con"] else 0
            gt_avg = sum(b["grits_top"]) / len(b["grits_top"]) if b["grits_top"] else 0
            label = f"{key[0]}={key[1]}"
            logger.info(f"  {label:<14} {b['count']:>7} "
                         f"{t_avg:>8.4f} {gc_avg:>8.4f} {gt_avg:>8.4f}")

    # ── Per-table attribute: matched-only recognition + detection recall ──
    tbl_matched_scores = {}
    tbl_recall_counts = {}

    def _tbl_matched_bucket(key):
        if key not in tbl_matched_scores:
            tbl_matched_scores[key] = {"teds": [], "grits_con": [], "grits_top": []}
        return tbl_matched_scores[key]

    def _tbl_recall_bucket(key):
        if key not in tbl_recall_counts:
            tbl_recall_counts[key] = {"gt_total": 0, "detected": 0}
        return tbl_recall_counts[key]

    for r in all_results:
        if r is None:
            continue
        gt_ta_list = r.get("gt_table_attrs")
        if not gt_ta_list:
            continue
        st = r.get("status", "error")
        if st == "error":
            continue

        matched_gt_indices = set()
        if st not in ("no_detection", "timeout"):
            for m in r.get("table_matches", []):
                gi_idx = m["gt_idx"]
                matched_gt_indices.add(gi_idx)
                scores = m.get("scores", {})
                teds_val = scores.get("teds", 0.0) or 0.0
                gc_val = scores.get("grits_con", 0.0) or 0.0
                gt_val = scores.get("grits_top", 0.0) or 0.0
                if gi_idx < len(gt_ta_list):
                    ta = gt_ta_list[gi_idx]
                    for attr_name, _ in table_attr_dimensions:
                        val = _classify_table(ta, attr_name)
                        mb = _tbl_matched_bucket((attr_name, val))
                        mb["teds"].append(teds_val)
                        mb["grits_con"].append(gc_val)
                        mb["grits_top"].append(gt_val)

        for gi_idx, ta in enumerate(gt_ta_list):
            for attr_name, _ in table_attr_dimensions:
                val = _classify_table(ta, attr_name)
                rb = _tbl_recall_bucket((attr_name, val))
                rb["gt_total"] += 1
                if gi_idx in matched_gt_indices:
                    rb["detected"] += 1

    if tbl_matched_scores or tbl_recall_counts:
        logger.info("")
        logger.info("Per-table attribute: matched-only recognition + detection recall:")
        logger.info(f"  {'Attribute':<14} {'GT_Tot':>7} {'Detect':>7} {'Recall':>7} "
                     f"{'TEDS':>8} {'GriTS_C':>8} {'GriTS_T':>8}")
        logger.info(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
        for key in print_order:
            rb = tbl_recall_counts.get(key, {"gt_total": 0, "detected": 0})
            mb = tbl_matched_scores.get(key, {"teds": [], "grits_con": [], "grits_top": []})
            gt_total = rb["gt_total"]
            detected = rb["detected"]
            recall = detected / gt_total if gt_total > 0 else 0
            t_avg = sum(mb["teds"]) / len(mb["teds"]) if mb["teds"] else 0
            gc_avg = sum(mb["grits_con"]) / len(mb["grits_con"]) if mb["grits_con"] else 0
            gt_avg = sum(mb["grits_top"]) / len(mb["grits_top"]) if mb["grits_top"] else 0
            label = f"{key[0]}={key[1]}"
            logger.info(f"  {label:<14} {gt_total:>7} {detected:>7} {recall:>7.4f} "
                         f"{t_avg:>8.4f} {gc_avg:>8.4f} {gt_avg:>8.4f}")

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_label = "_".join(f"{k}{v}" for k, v in sorted(SAMPLES_PER_SUBSET.items()) if v is not None)
    if ATTR_FILTER:
        filter_label = "_".join(f"{k}{v}" for k, v in sorted(ATTR_FILTER.items()))
        n_label += f"_F_{filter_label}"
    output_path = os.path.join(OUTPUT_DIR,
                               f"eval_llm_{MODEL_NAME}_{n_label}_seed{SEED}_{timestamp}.json")

    results_slim = []
    for r in all_results:
        if r is None:
            continue
        r_copy = dict(r)
        if "table_matches" in r_copy:
            slim_matches = []
            for m in r_copy["table_matches"]:
                m_copy = dict(m)
                m_copy["pred_html_len"] = len(m.get("pred_html", ""))
                m_copy["gt_html_len"] = len(m.get("gt_html", ""))
                slim_matches.append(m_copy)
            r_copy["table_matches"] = slim_matches
        results_slim.append(r_copy)

    # Build summaries
    subset_summary = {}
    for s_name in sorted(subset_scores.keys()):
        ss = subset_scores[s_name]
        subset_summary[s_name] = {
            "count": ss["count"], "no_det": ss["no_det"],
            "timeouts": ss["timeouts"], "errors": ss["errors"],
            "avg_teds": round(sum(ss["teds"]) / len(ss["teds"]), 6) if ss["teds"] else None,
            "avg_grits_con": round(sum(ss["grits_con"]) / len(ss["grits_con"]), 6) if ss["grits_con"] else None,
            "avg_grits_top": round(sum(ss["grits_top"]) / len(ss["grits_top"]), 6) if ss["grits_top"] else None,
        }

    attr_summary = {}
    for attr_name, values in attr_dimensions:
        for val in values:
            key = (attr_name, val)
            if key not in attr_scores:
                continue
            b = attr_scores[key]
            attr_summary[f"{attr_name}={val}"] = {
                "count": b["count"], "no_det": b["no_det"],
                "timeouts": b["timeouts"], "errors": b["errors"],
                "avg_teds": round(sum(b["teds"]) / len(b["teds"]), 6) if b["teds"] else None,
                "avg_grits_con": round(sum(b["grits_con"]) / len(b["grits_con"]), 6) if b["grits_con"] else None,
                "avg_grits_top": round(sum(b["grits_top"]) / len(b["grits_top"]), 6) if b["grits_top"] else None,
            }

    table_attr_summary = {}
    for key in print_order:
        if key not in table_attr_scores:
            continue
        b = table_attr_scores[key]
        label = f"{key[0]}={key[1]}"
        table_attr_summary[label] = {
            "table_count": b["count"],
            "avg_teds": round(sum(b["teds"]) / len(b["teds"]), 6) if b["teds"] else None,
            "avg_grits_con": round(sum(b["grits_con"]) / len(b["grits_con"]), 6) if b["grits_con"] else None,
            "avg_grits_top": round(sum(b["grits_top"]) / len(b["grits_top"]), 6) if b["grits_top"] else None,
        }

    table_matched_summary = {}
    for key in print_order:
        rb = tbl_recall_counts.get(key, {"gt_total": 0, "detected": 0})
        mb = tbl_matched_scores.get(key, {"teds": [], "grits_con": [], "grits_top": []})
        gt_total = rb["gt_total"]
        detected = rb["detected"]
        label = f"{key[0]}={key[1]}"
        table_matched_summary[label] = {
            "gt_total": gt_total, "detected": detected,
            "recall": round(detected / gt_total, 6) if gt_total > 0 else None,
            "matched_count": len(mb["teds"]),
            "avg_teds": round(sum(mb["teds"]) / len(mb["teds"]), 6) if mb["teds"] else None,
            "avg_grits_con": round(sum(mb["grits_con"]) / len(mb["grits_con"]), 6) if mb["grits_con"] else None,
            "avg_grits_top": round(sum(mb["grits_top"]) / len(mb["grits_top"]), 6) if mb["grits_top"] else None,
        }

    output_data = {
        "config": {
            "samples_per_subset": SAMPLES_PER_SUBSET,
            "attr_filter": ATTR_FILTER,
            "model_name": MODEL_NAME,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "concurrent_requests": CONCURRENT_REQUESTS,
            "llm_timeout": LLM_TIMEOUT,
            "score_workers": SCORE_WORKERS,
            "seed": SEED,
            "matching": "greedy best-TEDS matching",
            "timestamp": timestamp,
        },
        "summary": {
            "total_images": len(image_data),
            "successful": len(image_data) - error_count - timeout_count,
            "no_table_detected": no_det_count,
            "timeouts": timeout_count,
            "errors": error_count,
            "avg_teds_penalized": round(sum(image_teds) / len(image_teds), 6) if image_teds else None,
            "avg_grits_con_penalized": round(sum(image_grits_con) / len(image_grits_con), 6) if image_grits_con else None,
            "avg_grits_top_penalized": round(sum(image_grits_top) / len(image_grits_top), 6) if image_grits_top else None,
            "detection_exact_match_rate": round(exact_match / len(gt_counts), 4) if gt_counts else None,
            "wall_clock_s": round(t_total, 1),
            "llm_extract_s": round(t_extract_total, 1),
            "score_s": round(t_score_total, 1),
        },
        "per_subset": subset_summary,
        "per_attribute_image": attr_summary,
        "per_attribute_table": table_attr_summary,
        "per_attribute_table_matched": table_matched_summary,
        "results": results_slim,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Log saved to {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
