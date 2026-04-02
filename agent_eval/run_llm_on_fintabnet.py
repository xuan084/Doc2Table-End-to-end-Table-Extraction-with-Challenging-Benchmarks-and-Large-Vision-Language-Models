"""Run LLM-based table extraction on FinTabNet test 2K and evaluate.

Two evaluation flows:
  Flow 1 (with_ocr):    image + OCR text → LLM → HTML → score
  Flow 2 (without_ocr): image only       → LLM → HTML → score

Scoring: TEDS + GriTS, greedy TEDS matching, penalty ONLY for under-detection
(if N_pred < M_gt, multiply by N/M; if N_pred >= M_gt, no penalty).

Run: python -u agent_eval/run_llm_on_fintabnet.py
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

SEED = 42
SCORE_WORKERS = 16
FLOW_MODE = "with_ocr"            # "with_ocr", "without_ocr", or "both"

# ── LLM Configuration ──
MODEL_NAME = "gpt-5.2"
MAX_TOKENS = 8000
TEMPERATURE = 0.0
CONCURRENT_REQUESTS = 15
LLM_TIMEOUT = 120
API_BASE_URL = None
API_KEY = "YOUR_API_KEY_HERE"

# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "fintabnet_test2k")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OCR_DIR = os.path.join(DATA_DIR, "ocr")
GT_HTML_DIR = os.path.join(DATA_DIR, "gt_html")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")

# Sample image for one-shot reference (first image in manifest)
SAMPLE_PAGE_NAME = None  # None = auto-pick first

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
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
- Separate multiple tables with a blank line"""


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"eval_llm_{timestamp}.log")

    logger = logging.getLogger("fintabnet_eval")
    logger.setLevel(logging.DEBUG)
    # Clear existing handlers
    logger.handlers.clear()

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
# IMAGE ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def encode_image(path):
    ext = os.path.splitext(path)[1].lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext.lstrip("."), "jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


# ═══════════════════════════════════════════════════════════════════════════════
# HTML EXTRACTION FROM LLM RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

def extract_tables_from_response(text: str) -> List[str]:
    code_blocks = re.findall(r"```(?:html)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        tables = []
        for block in code_blocks:
            tables.extend(re.findall(r"(<table[^>]*>.*?</table>)", block, re.DOTALL | re.IGNORECASE))
        if tables:
            return [t.strip() for t in tables]
    tables = re.findall(r"(<table[^>]*>.*?</table>)", text, re.DOTALL | re.IGNORECASE)
    if tables:
        return [t.strip() for t in tables]
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# LLM EXTRACTION (async)
# ═══════════════════════════════════════════════════════════════════════════════

async def extract_table_llm(client, sample_img_url, sample_html,
                             target_img_url, ocr_text=None):
    """Extract tables from a page image via LLM."""

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
        user_prompt_parts.append(
            "6. Use the OCR reference text above to help with accurate text recognition, "
            "but determine the table structure from the visual image."
        )

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
    return content.strip() if content else ""


# ═══════════════════════════════════════════════════════════════════════════════
# HTML NORMALIZATION & SCORING (identical to run_llm_on_generated.py)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_html(html_str):
    html_str = re.sub(r'<th\b', '<td', html_str)
    html_str = html_str.replace('</th>', '</td>')
    html_str = html_str.replace('<thead>', '<tr>')
    html_str = html_str.replace('</thead>', '</tr>')
    html_str = html_str.replace('<tbody>', '')
    html_str = html_str.replace('</tbody>', '')
    html_str = re.sub(r'</?html[^>]*>', '', html_str, flags=re.IGNORECASE)
    html_str = re.sub(r'<head>.*?</head>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    html_str = re.sub(r'</?body[^>]*>', '', html_str, flags=re.IGNORECASE)
    html_str = re.sub(r'<style>.*?</style>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    html_str = re.sub(r'<meta[^>]*/>', '', html_str, flags=re.IGNORECASE)
    html_str = re.sub(r'<br\s*/?>', '<br/>', html_str, flags=re.IGNORECASE)
    _XML_ENTITIES = {'amp', 'lt', 'gt', 'quot', 'apos'}
    html_str = re.sub(r'&([a-zA-Z]+);',
                      lambda m: m.group(0) if m.group(1) in _XML_ENTITIES
                      else html_mod.unescape(m.group(0)), html_str)
    return html_str.strip()


def wrap_for_teds(html_str):
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


def greedy_match_tables_by_teds(pred_htmls, gt_htmls, teds_matrix):
    """Greedy best-match: for each pred table, find GT with highest TEDS.
    Matches until either pred or GT is exhausted."""
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
# OCR LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_ocr_text(page_name):
    ocr_path = os.path.join(OCR_DIR, f"{page_name}_ocr.json")
    if not os.path.exists(ocr_path):
        return None
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    lines = ocr_data.get("ocr", [])
    if not lines:
        return None
    return "\n".join(entry["text"] for entry in lines)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ONE FLOW (with_ocr or without_ocr)
# ═══════════════════════════════════════════════════════════════════════════════

async def run_flow(flow_name, use_ocr, logger, manifest, sample_img_url, sample_html,
                   output_dir):
    """Run one complete evaluation flow: extraction + scoring."""

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"FLOW: {flow_name} (OCR={'YES' if use_ocr else 'NO'})")
    logger.info("=" * 70)

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
        logger.error("API_KEY not set")
        return

    from openai import AsyncOpenAI
    client_kwargs = {"api_key": api_key}
    if API_BASE_URL:
        client_kwargs["base_url"] = API_BASE_URL
    client = AsyncOpenAI(**client_kwargs)

    # ── Load scorers ──
    teds_scorer = load_teds()
    grits_fn = load_grits()

    # ── Phase 1: LLM extraction ──
    logger.info(f"\nPhase 1: LLM extraction ({CONCURRENT_REQUESTS} concurrent)...")

    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    llm_results = {}  # idx -> [html_str, ...]
    llm_errors = 0
    llm_timeouts = 0
    extract_done = 0
    t_extract_start = time.time()
    total_images = len(manifest)

    async def _extract_one(idx, page_name):
        nonlocal extract_done, llm_errors, llm_timeouts
        async with sem:
            try:
                target_img_url = encode_image(os.path.join(IMAGES_DIR, f"{page_name}.png"))
                ocr_text = load_ocr_text(page_name) if use_ocr else None

                response_text = await asyncio.wait_for(
                    extract_table_llm(client, sample_img_url, sample_html,
                                      target_img_url, ocr_text),
                    timeout=LLM_TIMEOUT,
                )
                pred_tables = extract_tables_from_response(response_text)
                llm_results[idx] = pred_tables
            except asyncio.TimeoutError:
                llm_results[idx] = []
                llm_timeouts += 1
            except Exception as e:
                llm_results[idx] = []
                llm_errors += 1
                logger.warning(f"  LLM error for {page_name}: {e}")

            extract_done += 1
            if extract_done % 50 == 0 or extract_done == total_images:
                elapsed = time.time() - t_extract_start
                rate = extract_done / elapsed if elapsed > 0 else 0
                logger.info(f"  Extracted {extract_done}/{total_images} "
                            f"| {elapsed:.1f}s | {rate:.1f} img/s "
                            f"| errors={llm_errors} timeouts={llm_timeouts}")

    tasks = [_extract_one(idx, m["page_name"]) for idx, m in enumerate(manifest)]
    await asyncio.gather(*tasks)

    t_extract = time.time() - t_extract_start
    logger.info(f"Extraction done in {t_extract:.1f}s")

    # ── Save predictions for future re-scoring ──
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_save_path = os.path.join(output_dir, f"predictions_{flow_name}_{timestamp}.jsonl")
    with open(pred_save_path, "w", encoding="utf-8") as pf:
        for idx in range(len(manifest)):
            pred_tables = llm_results.get(idx, [])
            pred_text = "\n".join(pred_tables) if pred_tables else ""
            pf.write(json.dumps({"idx": idx, "pred_text": pred_text}, ensure_ascii=False) + "\n")
    logger.info(f"Predictions saved to {pred_save_path} ({len(llm_results)} entries)")
    logger.info("Inference done. Skipping old scoring — use eval_hunyuan.py for scoring.")
    return None

    # ── Phase 2: Scoring (multi-threaded) — SKIPPED (use eval_hunyuan.py instead) ──
    logger.info(f"\nPhase 2: Scoring with {SCORE_WORKERS} threads...")
    t_score_start = time.time()

    # Build all TEDS tasks
    image_info = []
    teds_tasks = []
    for idx, m in enumerate(manifest):
        gt = json.load(open(os.path.join(GT_HTML_DIR, f"{m['page_name']}_gt.json"), encoding="utf-8"))
        gt_htmls = [t["html"] for t in gt["tables"]]
        pred_htmls = llm_results.get(idx, [])
        M, N = len(gt_htmls), len(pred_htmls)
        image_info.append({"page_name": m["page_name"], "M": M, "N": N,
                           "pred_htmls": pred_htmls, "gt_htmls": gt_htmls})
        if M > 0 and N > 0:
            for pi in range(N):
                for gj in range(M):
                    teds_tasks.append((idx, pi, gj, pred_htmls[pi], gt_htmls[gj]))

    logger.info(f"  TEDS tasks: {len(teds_tasks)} pairs")

    # Thread-local TEDS scorers
    _thread_local = threading.local()

    def _get_thread_teds():
        if not hasattr(_thread_local, 'teds'):
            _thread_local.teds = load_teds()
        return _thread_local.teds

    teds_all = {}

    def _teds_one(task):
        idx, pi, gj, pred_html, gt_html = task
        if not pred_html or not gt_html:
            return (idx, pi, gj, 0.0)
        thread_teds = _get_thread_teds()
        pn = normalize_html(pred_html)
        gn = normalize_html(gt_html)
        try:
            score = thread_teds.evaluate(wrap_for_teds(pn), wrap_for_teds(gn))
        except Exception:
            score = 0.0
        return (idx, pi, gj, score)

    with ThreadPoolExecutor(max_workers=SCORE_WORKERS) as pool:
        futures = {pool.submit(_teds_one, t): t for t in teds_tasks}
        done_count = 0
        for future in as_completed(futures):
            idx, pi, gj, score = future.result()
            teds_all[(idx, pi, gj)] = score
            done_count += 1
            if done_count % 100 == 0 or done_count == len(teds_tasks):
                elapsed = time.time() - t_score_start
                rate = done_count / elapsed if elapsed > 0 else 0
                logger.info(f"  TEDS: {done_count}/{len(teds_tasks)} ({rate:.1f} pairs/s)")

    # Greedy matching + GriTS
    grits_tasks = []
    per_image_matches = {}

    for idx, info in enumerate(image_info):
        M, N = info["M"], info["N"]
        if M == 0 or N == 0:
            per_image_matches[idx] = []
            continue
        teds_matrix = {(pi, gj): teds_all.get((idx, pi, gj), 0.0)
                       for pi in range(N) for gj in range(M)}
        matches = greedy_match_tables_by_teds(info["pred_htmls"], info["gt_htmls"], teds_matrix)
        per_image_matches[idx] = matches
        for pi, gj in matches:
            grits_tasks.append((idx, pi, gj, info["pred_htmls"][pi], info["gt_htmls"][gj]))

    logger.info(f"  GriTS tasks: {len(grits_tasks)} matched pairs")

    grits_all = {}

    def _grits_one(task):
        idx, pi, gj, pred_html, gt_html = task
        pn = normalize_html(pred_html)
        gn = normalize_html(gt_html)
        try:
            g = grits_fn(gn, pn)
            return (idx, pi, gj, {"grits_con": g["grits_con"], "grits_top": g["grits_top"]})
        except Exception:
            return (idx, pi, gj, {"grits_con": 0.0, "grits_top": 0.0})

    with ThreadPoolExecutor(max_workers=SCORE_WORKERS) as pool:
        futures = {pool.submit(_grits_one, t): t for t in grits_tasks}
        done_count = 0
        for future in as_completed(futures):
            idx, pi, gj, scores = future.result()
            grits_all[(idx, pi, gj)] = scores
            done_count += 1
            if done_count % 100 == 0 or done_count == len(grits_tasks):
                logger.info(f"  GriTS: {done_count}/{len(grits_tasks)}")

    t_score = time.time() - t_score_start
    logger.info(f"Scoring done in {t_score:.1f}s")

    # ── Assemble per-image scores (penalty ONLY for under-detection) ──
    all_results = []
    image_teds = []
    image_gc = []
    image_gt = []

    # Also collect per-table scores
    table_teds_all = []
    table_gc_all = []
    table_gt_all = []

    for idx, info in enumerate(image_info):
        M, N = info["M"], info["N"]
        matches = per_image_matches.get(idx, [])

        if not matches:
            # No matches: score = 0 if there were GT tables, skip if no GT
            if M > 0:
                image_teds.append(0.0)
                image_gc.append(0.0)
                image_gt.append(0.0)
            result = {
                "page_name": info["page_name"], "status": "no_detection" if N == 0 else "no_match",
                "num_tables_gt": M, "num_tables_detected": N,
                "penalty": 0.0,
                "avg_teds": 0.0, "avg_grits_con": 0.0, "avg_grits_top": 0.0,
            }
        else:
            pair_teds = []
            pair_gc = []
            pair_gt_scores = []
            for pi, gj in matches:
                t = teds_all.get((idx, pi, gj), 0.0)
                g = grits_all.get((idx, pi, gj), {"grits_con": 0.0, "grits_top": 0.0})
                pair_teds.append(t)
                pair_gc.append(g["grits_con"])
                pair_gt_scores.append(g["grits_top"])
                # Per-table
                table_teds_all.append(t)
                table_gc_all.append(g["grits_con"])
                table_gt_all.append(g["grits_top"])

            avg_t = sum(pair_teds) / len(pair_teds)
            avg_gc = sum(pair_gc) / len(pair_gc)
            avg_gt = sum(pair_gt_scores) / len(pair_gt_scores)

            # Penalty ONLY for under-detection (N < M)
            # Over-detection (N > M) is not penalized
            penalty = N / M if N < M else 1.0
            penalized_t = avg_t * penalty
            penalized_gc = avg_gc * penalty
            penalized_gt = avg_gt * penalty

            image_teds.append(penalized_t)
            image_gc.append(penalized_gc)
            image_gt.append(penalized_gt)

            result = {
                "page_name": info["page_name"], "status": "ok",
                "num_tables_gt": M, "num_tables_detected": N,
                "num_matched": len(matches),
                "penalty": round(penalty, 4),
                "raw_avg_teds": round(avg_t, 6),
                "raw_avg_grits_con": round(avg_gc, 6),
                "raw_avg_grits_top": round(avg_gt, 6),
                "avg_teds": round(penalized_t, 6),
                "avg_grits_con": round(penalized_gc, 6),
                "avg_grits_top": round(penalized_gt, 6),
            }

        all_results.append(result)

    # ── Statistics ──
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def log_stats(name, scores):
        if not scores:
            return
        s = sorted(scores)
        n = len(s)
        logger.info(f"  {name}:")
        logger.info(f"    mean={mean(s):.4f}  median={s[n//2]:.4f}  "
                     f"p25={s[n//4]:.4f}  p75={s[3*n//4]:.4f}  "
                     f"min={s[0]:.4f}  max={s[-1]:.4f}  n={n}")

    n_no_det = sum(1 for r in all_results if r.get("status") == "no_detection")
    total_gt = sum(r["num_tables_gt"] for r in all_results)
    total_pred = sum(r["num_tables_detected"] for r in all_results)
    exact_match = sum(1 for r in all_results if r["num_tables_gt"] == r["num_tables_detected"])

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"RESULTS — {flow_name}")
    logger.info("=" * 70)
    logger.info(f"  Images:      {len(manifest)}")
    logger.info(f"  No detection: {n_no_det}")
    logger.info(f"  Timeouts:    {llm_timeouts}")
    logger.info(f"  Errors:      {llm_errors}")
    logger.info(f"  GT tables:   {total_gt}")
    logger.info(f"  Pred tables: {total_pred}")
    logger.info(f"  Det exact:   {exact_match}/{len(manifest)} ({100*exact_match/len(manifest):.1f}%)")
    logger.info("")

    logger.info("  --- Per-image (penalized for under-detection only) ---")
    log_stats("TEDS", image_teds)
    log_stats("GriTS_Con", image_gc)
    log_stats("GriTS_Top", image_gt)

    logger.info("")
    logger.info(f"  --- Per-table (matched only, N={len(table_teds_all)}) ---")
    log_stats("TEDS", table_teds_all)
    log_stats("GriTS_Con", table_gc_all)
    log_stats("GriTS_Top", table_gt_all)

    # By GT table count
    by_n = defaultdict(lambda: {"teds": [], "gc": [], "gt": []})
    for r in all_results:
        n_gt = r["num_tables_gt"]
        by_n[n_gt]["teds"].append(r.get("avg_teds", 0.0))
        by_n[n_gt]["gc"].append(r.get("avg_grits_con", 0.0))
        by_n[n_gt]["gt"].append(r.get("avg_grits_top", 0.0))

    logger.info("")
    logger.info("  --- By GT table count ---")
    for n_gt in sorted(by_n.keys()):
        b = by_n[n_gt]
        logger.info(f"  {n_gt}-table (n={len(b['teds'])}): "
                     f"TEDS={mean(b['teds']):.4f}  GC={mean(b['gc']):.4f}  GT={mean(b['gt']):.4f}")

    logger.info("")
    logger.info(f"  Timing: extraction={t_extract:.0f}s  scoring={t_score:.0f}s  "
                f"total={t_extract+t_score:.0f}s")

    # ── Save JSON ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "config": {
            "flow": flow_name,
            "use_ocr": use_ocr,
            "model_name": MODEL_NAME,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "scoring": "greedy TEDS matching, penalty for under-detection only",
            "timestamp": timestamp,
        },
        "summary": {
            "total_images": len(manifest),
            "no_detection": n_no_det,
            "timeouts": llm_timeouts,
            "errors": llm_errors,
            "avg_teds": round(mean(image_teds), 6),
            "avg_grits_con": round(mean(image_gc), 6),
            "avg_grits_top": round(mean(image_gt), 6),
            "per_table_avg_teds": round(mean(table_teds_all), 6),
            "per_table_avg_grits_con": round(mean(table_gc_all), 6),
            "per_table_avg_grits_top": round(mean(table_gt_all), 6),
            "detection_exact_match": f"{exact_match}/{len(manifest)}",
            "extraction_s": round(t_extract, 1),
            "scoring_s": round(t_score, 1),
        },
        "results": all_results,
    }

    json_path = os.path.join(output_dir,
                              f"eval_{MODEL_NAME}_{flow_name}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {json_path}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    output_dir = os.path.join(SCRIPT_DIR, "output_fintabnet_llm_prediction")
    logger, log_path = setup_logging(output_dir)

    logger.info("=" * 70)
    logger.info("LLM Table Extraction on FinTabNet Test 2K")
    logger.info("=" * 70)
    logger.info(f"  MODEL       = {MODEL_NAME}")
    logger.info(f"  CONCURRENT  = {CONCURRENT_REQUESTS}")
    logger.info(f"  TIMEOUT     = {LLM_TIMEOUT}s")
    logger.info(f"  SCORE_WORKERS = {SCORE_WORKERS}")

    # Load manifest
    manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    logger.info(f"  Images      = {len(manifest)}")
    logger.info(f"  Total tables = {sum(m['n_tables'] for m in manifest)}")

    # Load sample image for one-shot
    sample_name = SAMPLE_PAGE_NAME or manifest[0]["page_name"]
    sample_img_url = encode_image(os.path.join(IMAGES_DIR, f"{sample_name}.png"))
    sample_gt = json.load(open(os.path.join(GT_HTML_DIR, f"{sample_name}_gt.json"), encoding="utf-8"))
    sample_html = sample_gt["tables"][0]["html"] if sample_gt["tables"] else "<table><tr><td></td></tr></table>"
    logger.info(f"  Sample      = {sample_name} (HTML len={len(sample_html)})")

    logger.info(f"  FLOW_MODE   = {FLOW_MODE}")

    summary_ocr = None
    summary_no_ocr = None

    # ── Flow 1: With OCR ──
    if FLOW_MODE in ("with_ocr", "both"):
        summary_ocr = await run_flow(
            "with_ocr", use_ocr=True, logger=logger, manifest=manifest,
            sample_img_url=sample_img_url, sample_html=sample_html,
            output_dir=output_dir)

    # ── Flow 2: Without OCR ──
    if FLOW_MODE in ("without_ocr", "both"):
        summary_no_ocr = await run_flow(
            "without_ocr", use_ocr=False, logger=logger, manifest=manifest,
            sample_img_url=sample_img_url, sample_html=sample_html,
            output_dir=output_dir)

    # ── Comparison ──
    if summary_ocr and summary_no_ocr:
        logger.info("")
        logger.info("=" * 70)
        logger.info("COMPARISON: With OCR vs Without OCR")
        logger.info("=" * 70)
        for metric in ["avg_teds", "avg_grits_con", "avg_grits_top"]:
            v1 = summary_ocr["summary"][metric]
            v2 = summary_no_ocr["summary"][metric]
            diff = v1 - v2
            logger.info(f"  {metric}: with_ocr={v1:.4f}  without_ocr={v2:.4f}  diff={diff:+.4f}")

    logger.info(f"\nLog saved to {log_path}")
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
