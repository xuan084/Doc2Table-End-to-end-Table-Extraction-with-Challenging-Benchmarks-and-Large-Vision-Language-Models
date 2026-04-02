"""Run TableExtractionPipeline on generated dataset and evaluate with TEDS + GriTS_Con.

Pipeline flow:
1. Load image from pipeline_output/images/
2. Extract word-level tokens from GT JSON v1 (paragraphs + table cells)
3. Run detection → get table bboxes and crops
4. Run recognition on each crop → get predicted HTML per table
5. Greedy IoU matching: pred tables ↔ GT tables
6. Post-process predicted HTML: <th> → <td>, <thead> → <tr> for TEDS compatibility
7. Score matched pairs with TEDS and GriTS_Con
8. Apply count penalty: avg_score * min(M,N) / max(M,N)
9. Report results with timing stats

All parameters are set as constants below — no CLI arguments needed.
Run: python -u pipeline_eval/run_pipeline_on_generated.py
"""

import json
import os
import sys
import time
import glob
import random
import re
import logging
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from PIL import Image, ImageDraw, ImageFont

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these constants directly
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES_PER_SUBSET = {     # Number of images to sample per subset
    "S1": 1000,              # (set a value to None or omit a key to use all images in that subset)
    "S2": 0,
    "S3": 0,
    "S4": 0,
}

# Attribute filter — only sample images matching ALL specified conditions.
# Set to None to disable (use all images in each subset).
# Each key is optional; omitted keys are not filtered.
#
# Available keys and values:
#   "T1":  "Yes" | "No"    — image has at least one table with T1=Yes / all tables T1=No
#   "T3":  "degraded" | "None"  — page-level: any degradation vs clean
#          or specific type: "Watermark" | "Stain" | "Stamp"
#   "T4":  "Full" | "not_Full"  — all tables have full borders / at least one table without
#
# Examples:
#   ATTR_FILTER = None                              # no filter (default)
#   ATTR_FILTER = {"T1": "Yes"}                     # only images with split-inducing tables
#   ATTR_FILTER = {"T3": "degraded"}                # only degraded images
#   ATTR_FILTER = {"T3": "Watermark"}               # only watermark degradation
#   ATTR_FILTER = {"T4": "Full"}                    # only full-border images
#   ATTR_FILTER = {"T1": "Yes", "T3": "degraded"}   # split + degraded (hardest)
#   ATTR_FILTER = {"T1": "No", "T3": "None", "T4": "Full"}  # easiest possible
ATTR_FILTER = None

DEVICE = "cuda"            # "cuda" or "cpu"
SEED = 42                  # Random seed for reproducible sampling
NUM_WORKERS = 12            # Number of threads for I/O-bound preprocessing
IOU_THRESHOLD = 0.0        # Minimum IoU for a valid table match (0.0 = any overlap)
DET_SCORE_THRESHOLD = None # Detection score threshold (None = use pipeline defaults)
SAVE_DET_VIS = True        # Save detection visualization images (pred + GT bboxes)

# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # pipeline_eval/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                       # table-transformer/
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "detr"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

GENERATED_DIR = os.path.join(PROJECT_ROOT, "data", "WildDocTables")
IMAGES_DIR = os.path.join(GENERATED_DIR, "images")
GT_JSON_V1_DIR = os.path.join(GENERATED_DIR, "gt_json_v1")
GT_JSON_V2_DIR = os.path.join(GENERATED_DIR, "gt_json_v2")
MANIFEST_PATH = os.path.join(GENERATED_DIR, "manifest.jsonl")

DETECTION_CONFIG = os.path.join(PROJECT_ROOT, "src", "detection_config.json")
DETECTION_MODEL = os.path.join(PROJECT_ROOT, "pubtables1m_detection_detr_r18.pth")
STRUCTURE_CONFIG = os.path.join(PROJECT_ROOT, "src", "structure_config.json")
STRUCTURE_MODEL = os.path.join(PROJECT_ROOT, "pubtables1m_structure_detr_r18.pth")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_s1")
DET_VIS_DIR = os.path.join(OUTPUT_DIR, "det_visualizations")

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS — pipeline and scorers (must come after sys.path setup)
# ═══════════════════════════════════════════════════════════════════════════════

from inference import TableExtractionPipeline, objects_to_crops


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    """Create logger that writes to both console and log file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(OUTPUT_DIR, f"eval_{timestamp}.log")

    logger = logging.getLogger("pipeline_eval")
    logger.setLevel(logging.DEBUG)

    # File handler — full detail
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # Console handler — info and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger, log_path


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def draw_detection_boxes(img, det_objects, gt_bboxes, img_id, subset):
    """Draw pred (red) and GT (green) bboxes on image and save to DET_VIS_DIR.

    Pred boxes: red solid, with score label.
    GT boxes: green dashed (simulated with short segments).
    """
    vis = img.copy()
    draw = ImageDraw.Draw(vis)

    # GT boxes — green
    for i, bbox in enumerate(gt_bboxes):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw.text((x1, y1 - 12), f"GT{i}", fill="green")

    # Pred boxes — red, with score
    for i, obj in enumerate(det_objects):
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = f"P{i} {obj['score']:.2f}"
        draw.text((x1, y2 + 2), label, fill="red")

    os.makedirs(DET_VIS_DIR, exist_ok=True)
    save_path = os.path.join(DET_VIS_DIR, f"{img_id}_{subset}.png")
    vis.save(save_path)
    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_word_tokens(gt_json):
    """Extract word-level tokens from GT JSON v1 for pipeline input.

    Tokens come from:
    - paragraph → lines → words (free text around tables)
    - table → cells → words (text inside table cells)

    Each token has: text, bbox, span_num, line_num, block_num
    These fields are used by the pipeline's objects_to_crops() for IoB filtering
    and slot_into_containers() for cell text assignment.
    """
    tokens = []
    span_num = 0
    block_num = 0

    for element in gt_json.get("page", []):
        if element["type"] == "paragraph":
            for line_num, line in enumerate(element.get("lines", [])):
                for word in line.get("words", []):
                    tokens.append({
                        "text": word["text"],
                        "bbox": word["bbox"],
                        "span_num": span_num,
                        "line_num": line_num,
                        "block_num": block_num,
                    })
                    span_num += 1
            block_num += 1

        elif element["type"] == "table":
            for cell in element.get("cells", []):
                for word in cell.get("words", []):
                    tokens.append({
                        "text": word["text"],
                        "bbox": word["bbox"],
                        "span_num": span_num,
                        "line_num": 0,
                        "block_num": block_num,
                    })
                    span_num += 1
            block_num += 1

    return tokens


# ═══════════════════════════════════════════════════════════════════════════════
# HTML POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_html(html_str):
    """Normalize table HTML to a canonical form for both scorers.

    Converts to a flat structure with only <table>, <tr>, <td> tags.
    This ensures pred HTML (from pipeline) and GT HTML (from v2) are
    treated identically by both TEDS and GriTS.

    Pipeline output has: <th> inside <thead> (no <tr> wrapper), no <tbody>
    GT v1 output has:    <td> inside <thead><tr>, plus <tbody><tr>
    GT v2 output has:    <td> inside <tr>, no <thead>/<tbody>
    Normalized form:     <td>, <tr>,     no <thead>/<tbody>

    Key: pipeline's cells_to_html() uses <thead> as the row container for
    header cells (i.e. <thead><th>A</th></thead>), NOT <thead><tr><th>...</th></tr></thead>.
    So <thead> → <tr> converts the pipeline's header row wrapper correctly.
    For GT v1 which has <thead><tr>..., this produces <tr><tr>... (double-nested),
    but we only use GT v2 HTML for scoring, which has no <thead>.
    """
    # <th ...> → <td ...>  (preserves attributes like colspan)
    html_str = re.sub(r'<th\b', '<td', html_str)
    html_str = html_str.replace('</th>', '</td>')
    # <thead> → <tr>  (pipeline uses <thead> as row wrapper for headers)
    html_str = html_str.replace('<thead>', '<tr>')
    html_str = html_str.replace('</thead>', '</tr>')
    # <tbody> → remove (GT v1 uses it, but inner <tr> tags remain)
    html_str = html_str.replace('<tbody>', '')
    html_str = html_str.replace('</tbody>', '')
    return html_str


def wrap_for_teds(html_str):
    """Wrap <table> HTML with <html><body> for TEDS xpath("body/table") compatibility."""
    return "<html><body>" + html_str + "</body></html>"


# ═══════════════════════════════════════════════════════════════════════════════
# IoU MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def greedy_match_tables(pred_bboxes, gt_bboxes, iou_threshold=0.0):
    """Greedy IoU matching: for each pred table, find best unmatched GT table.

    Returns list of (pred_idx, gt_idx, iou) tuples.
    Only matches with IoU > iou_threshold are kept.
    """
    matched = []
    used_gt = set()

    for pi in range(len(pred_bboxes)):
        best_gi = -1
        best_iou = 0.0
        for gi in range(len(gt_bboxes)):
            if gi in used_gt:
                continue
            iou = compute_iou(pred_bboxes[pi], gt_bboxes[gi])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi >= 0 and best_iou > iou_threshold:
            matched.append((pi, best_gi, best_iou))
            used_gt.add(best_gi)

    return matched


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def load_teds():
    """Load TEDS scorer from unitable."""
    teds_path = os.path.join(PROJECT_ROOT, "unitable", "unitable", "src", "utils")
    if teds_path not in sys.path:
        sys.path.insert(0, teds_path)
    from teds import TEDS
    return TEDS(n_jobs=1, ignore_nodes="", structure_only=False)


def load_grits():
    """Load GriTS scoring function from src/grits.py."""
    from grits import grits_from_html
    return grits_from_html


def compute_scores(pred_html, gt_html, teds_scorer, grits_fn):
    """Compute TEDS and GriTS_Con scores for a single table pair.

    Both pred and GT are first normalized to canonical form (all <td>, no
    <thead>/<tbody>) so both scorers see identical tag structure.

    TEDS: normalized + wrapped with <html><body>
    GriTS: normalized only (bare <table>)
    """
    scores = {}

    pred_norm = normalize_html(pred_html)
    gt_norm = normalize_html(gt_html)

    # TEDS
    pred_for_teds = wrap_for_teds(pred_norm)
    gt_for_teds = wrap_for_teds(gt_norm)
    try:
        scores["teds"] = teds_scorer.evaluate(pred_for_teds, gt_for_teds)
    except Exception as e:
        scores["teds"] = None
        scores["teds_error"] = str(e)

    # GriTS
    try:
        grits_metrics = grits_fn(gt_norm, pred_norm)
        scores["grits_con"] = grits_metrics["grits_con"]
        scores["grits_top"] = grits_metrics["grits_top"]
    except Exception as e:
        scores["grits_con"] = None
        scores["grits_top"] = None
        scores["grits_error"] = str(e)

    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# I/O PREPROCESSING (parallelizable)
# ═══════════════════════════════════════════════════════════════════════════════

def preload_image_data(img_path):
    """Load image + GT JSONs from disk. Thread-safe, no GPU usage.

    Returns dict with all data needed for pipeline inference, or error string.
    """
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    gt_v1_path = os.path.join(GT_JSON_V1_DIR, f"{img_id}.json")
    gt_v2_path = os.path.join(GT_JSON_V2_DIR, f"{img_id}_v2.json")

    try:
        img = Image.open(img_path).convert("RGB")

        if not os.path.exists(gt_v1_path):
            return {"error": f"GT v1 not found: {gt_v1_path}", "image_id": img_id}
        if not os.path.exists(gt_v2_path):
            return {"error": f"GT v2 not found: {gt_v2_path}", "image_id": img_id}

        with open(gt_v1_path, "r", encoding="utf-8") as f:
            gt_v1 = json.load(f)
        with open(gt_v2_path, "r", encoding="utf-8") as f:
            gt_v2 = json.load(f)

        tokens = extract_word_tokens(gt_v1)

        # Subset info is stored in GT JSON v1 top-level
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

        return {
            "image_id": img_id,
            "subset": subset,
            "image": img,
            "tokens": tokens,
            "gt_info": gt_info,
            "num_gt_tables": len(gt_tables_v1),
        }
    except Exception as e:
        return {"error": str(e), "image_id": img_id}


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE IMAGE PROCESSING (GPU inference — must be serialized)
# ═══════════════════════════════════════════════════════════════════════════════

def process_single_image(data, pipe, teds_scorer, grits_fn, logger):
    """Run detection + recognition + scoring on a single preloaded image.

    Returns result dict with per-image scores, timing, and match details.
    """
    img_id = data["image_id"]

    if "error" in data:
        logger.debug(f"  SKIP {img_id}: {data['error']}")
        return {"image_id": img_id, "status": "error", "error": data["error"],
                "subset": data.get("subset", "unknown")}

    img = data["image"]
    tokens = data["tokens"]
    gt_info = data["gt_info"]
    M = data["num_gt_tables"]
    subset = data.get("subset", "unknown")

    t_start = time.time()

    # ── Detection ──
    try:
        detect_out = pipe.detect(img, tokens=tokens, out_objects=True, out_crops=True)
    except Exception as e:
        logger.debug(f"  ERROR detection {img_id}: {e}")
        return {"image_id": img_id, "subset": subset, "status": "error",
                "error": f"detection: {e}"}

    # det_objects must mirror the same filtering as objects_to_crops() so that
    # det_objects[ci] corresponds to crops[ci].  objects_to_crops uses
    # pipe.det_class_thresholds internally, so we replicate that exact filter.
    all_objects = detect_out.get("objects", [])
    det_objects = [
        o for o in all_objects
        if o["score"] >= pipe.det_class_thresholds.get(o["label"], 10)
    ]
    crops = detect_out.get("crops", [])

    # Apply optional user-specified score threshold as secondary filter
    if DET_SCORE_THRESHOLD is not None:
        keep = [(o, c) for o, c in zip(det_objects, crops)
                if o["score"] >= DET_SCORE_THRESHOLD]
        if keep:
            det_objects, crops = zip(*keep)
            det_objects, crops = list(det_objects), list(crops)
        else:
            det_objects, crops = [], []

    t_det = time.time() - t_start

    # Save detection visualization
    if SAVE_DET_VIS:
        gt_bboxes_for_vis = [g["bbox"] for g in gt_info]
        draw_detection_boxes(img, det_objects, gt_bboxes_for_vis, img_id, subset)

    if not det_objects:
        return {
            "image_id": img_id,
            "subset": subset,
            "status": "no_detection",
            "num_tables_detected": 0,
            "num_tables_gt": M,
            "time_det_s": round(t_det, 4),
            "time_total_s": round(time.time() - t_start, 4),
            "penalized_teds": 0.0,
            "penalized_grits_con": 0.0,
            "penalized_grits_top": 0.0,
        }

    # ── Recognition ──
    t_rec_start = time.time()
    pred_tables = []
    for ci, crop in enumerate(crops):
        try:
            rec_out = pipe.recognize(
                crop["image"], tokens=crop["tokens"],
                out_html=True, out_cells=True,
            )
            html_list = rec_out.get("html", [])
            pred_html = html_list[0] if html_list else ""
        except Exception as e:
            logger.debug(f"  ERROR recognition {img_id} table {ci}: {e}")
            pred_html = ""

        det_bbox = det_objects[ci]["bbox"] if ci < len(det_objects) else [0, 0, 0, 0]
        pred_tables.append({"bbox": det_bbox, "html": pred_html})

    t_rec = time.time() - t_rec_start
    N = len(pred_tables)

    # ── Greedy IoU matching ──
    pred_bboxes = [p["bbox"] for p in pred_tables]
    gt_bboxes = [g["bbox"] for g in gt_info]
    matches = greedy_match_tables(pred_bboxes, gt_bboxes, IOU_THRESHOLD)

    # ── Score matched pairs ──
    t_score_start = time.time()
    table_matches = []
    pair_teds = []
    pair_grits_con = []
    pair_grits_top = []

    for pi, gi, iou in matches:
        pred_html = pred_tables[pi]["html"]
        gt_html = gt_info[gi]["html"]

        if not pred_html or not gt_html:
            logger.debug(f"  Empty HTML match {img_id} pred={pi} gt={gi}: scoring as 0")
            scores = {"teds": 0.0, "grits_con": 0.0, "grits_top": 0.0}
        else:
            scores = compute_scores(pred_html, gt_html, teds_scorer, grits_fn)

        table_matches.append({
            "pred_idx": pi,
            "gt_idx": gi,
            "iou": round(iou, 4),
            "pred_bbox": pred_tables[pi]["bbox"],
            "gt_bbox": gt_info[gi]["bbox"],
            "pred_html": pred_html,
            "gt_html": gt_html,
            "scores": scores,
        })

        if scores.get("teds") is not None:
            pair_teds.append(scores["teds"])
        if scores.get("grits_con") is not None:
            pair_grits_con.append(scores["grits_con"])
        if scores.get("grits_top") is not None:
            pair_grits_top.append(scores["grits_top"])

    t_score = time.time() - t_score_start

    # ── Count penalty ──
    if M == 0 and N == 0:
        penalty = 1.0
    elif M == 0 or N == 0:
        penalty = 0.0
    else:
        penalty = min(M, N) / max(M, N)

    avg_teds_raw = sum(pair_teds) / len(pair_teds) if pair_teds else 0.0
    avg_grits_con_raw = sum(pair_grits_con) / len(pair_grits_con) if pair_grits_con else 0.0
    avg_grits_top_raw = sum(pair_grits_top) / len(pair_grits_top) if pair_grits_top else 0.0

    t_total = time.time() - t_start

    return {
        "image_id": img_id,
        "subset": subset,
        "status": "ok",
        "num_tables_detected": N,
        "num_tables_gt": M,
        "penalty": round(penalty, 4),
        "table_matches": table_matches,
        "raw_avg_teds": round(avg_teds_raw, 6),
        "raw_avg_grits_con": round(avg_grits_con_raw, 6),
        "raw_avg_grits_top": round(avg_grits_top_raw, 6),
        "penalized_teds": round(avg_teds_raw * penalty, 6),
        "penalized_grits_con": round(avg_grits_con_raw * penalty, 6),
        "penalized_grits_top": round(avg_grits_top_raw * penalty, 6),
        "time_det_s": round(t_det, 4),
        "time_rec_s": round(t_rec, 4),
        "time_score_s": round(t_score, 4),
        "time_total_s": round(t_total, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ATTRIBUTE FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def image_matches_filter(manifest_rec, attr_filter):
    """Check if a manifest record matches the attribute filter.

    Filter keys:
      T1: "Yes" = at least one table has T1=Yes; "No" = all tables T1=No
      T3: "None" = page clean; "degraded" = any degradation; or specific "Watermark"/"Stain"/"Stamp"
      T4: "Full" = all tables T4=Full; "not_Full" = at least one table T4!=Full
    """
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

def main():
    logger, log_path = setup_logging()

    logger.info("=" * 70)
    logger.info("Pipeline Evaluation on Generated Dataset")
    logger.info("=" * 70)
    total_samples = sum(v for v in SAMPLES_PER_SUBSET.values() if v is not None)
    logger.info(f"  SAMPLES/SUBSET = {SAMPLES_PER_SUBSET}  (total={total_samples})")
    logger.info(f"  ATTR_FILTER  = {ATTR_FILTER}")
    logger.info(f"  DEVICE       = {DEVICE}")
    logger.info(f"  SEED         = {SEED}")
    logger.info(f"  NUM_WORKERS  = {NUM_WORKERS}")
    logger.info(f"  IOU_THRESHOLD= {IOU_THRESHOLD}")
    logger.info(f"  DET_SCORE_TH = {DET_SCORE_THRESHOLD}")
    logger.info(f"  SAVE_DET_VIS = {SAVE_DET_VIS}")
    logger.info(f"  OUTPUT_DIR   = {OUTPUT_DIR}")
    logger.info("")

    # ── Verify model files ──
    for p, name in [(DETECTION_MODEL, "Detection model"),
                     (STRUCTURE_MODEL, "Structure model")]:
        if not os.path.exists(p):
            logger.error(f"{name} not found: {p}")
            sys.exit(1)

    # ── Load pipeline ──
    logger.info("Loading pipeline models...")
    t0 = time.time()
    pipe = TableExtractionPipeline(
        det_config_path=DETECTION_CONFIG,
        det_model_path=DETECTION_MODEL,
        str_config_path=STRUCTURE_CONFIG,
        str_model_path=STRUCTURE_MODEL,
        det_device=DEVICE,
        str_device=DEVICE,
    )
    t_model_load = time.time() - t0
    logger.info(f"Models loaded in {t_model_load:.1f}s")

    # ── Load scorers ──
    logger.info("Loading scorers (TEDS + GriTS)...")
    teds_scorer = load_teds()
    grits_fn = load_grits()
    logger.info("Scorers ready.")

    # ── Build subset index from manifest (with attribute filter) ──
    logger.info(f"Reading manifest: {MANIFEST_PATH}")
    subset_to_ids = {}  # subset -> [image_id, ...]
    id_to_attrs = {}    # image_id -> {"T1": "Yes"/"No", "T3": "degraded"/"None", "T4": "Full"/"not_Full"}
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

            # Build per-image attribute flags for difficulty analysis
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
    logger.info(f"Filtered subsets: {{{', '.join(f'{s}: {len(ids)}' for s, ids in sorted(subset_to_ids.items()))}}}")

    # ── Per-subset sampling ──
    rng = random.Random(SEED)
    sampled = []
    for subset_name in sorted(SAMPLES_PER_SUBSET.keys()):
        n_want = SAMPLES_PER_SUBSET[subset_name]
        available = subset_to_ids.get(subset_name, [])
        if not available:
            logger.warning(f"  Subset {subset_name}: no images in manifest, skipping")
            continue
        if n_want is None or n_want >= len(available):
            chosen = available
        else:
            chosen = rng.sample(available, n_want)
            chosen.sort()
        # Convert image_id to file path, keep only those that exist on disk
        for img_id in chosen:
            img_path = os.path.join(IMAGES_DIR, f"{img_id}.png")
            if os.path.exists(img_path):
                sampled.append(img_path)
            else:
                logger.debug(f"  {img_id}.png not on disk, skipped")
        logger.info(f"  {subset_name}: {len(chosen)} sampled from {len(available)} available")

    sampled.sort()
    logger.info(f"Total sampled: {len(sampled)} images (seed={SEED})")
    logger.debug(f"First 5 samples: {[os.path.basename(p) for p in sampled[:5]]}")

    # ── Phase 1: Preload data with thread pool ──
    logger.info(f"\nPhase 1: Preloading image data with {NUM_WORKERS} threads...")
    t_preload = time.time()

    preloaded = [None] * len(sampled)
    preload_errors = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        future_to_idx = {
            pool.submit(preload_image_data, img_path): idx
            for idx, img_path in enumerate(sampled)
        }
        done_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                preloaded[idx] = future.result()
                if "error" in preloaded[idx]:
                    preload_errors += 1
            except Exception as e:
                img_id = os.path.splitext(os.path.basename(sampled[idx]))[0]
                preloaded[idx] = {"error": str(e), "image_id": img_id}
                preload_errors += 1

            done_count += 1
            if done_count % 50 == 0 or done_count == len(sampled):
                logger.info(f"  Preloaded {done_count}/{len(sampled)} "
                            f"({preload_errors} errors)")

    logger.info(f"Preloading done in {time.time() - t_preload:.1f}s "
                f"({preload_errors} errors)")

    # ── Phase 2: GPU inference + scoring (sequential, GPU not thread-safe) ──
    logger.info(f"\nPhase 2: Pipeline inference + scoring...")
    t_infer = time.time()

    all_results = []
    image_teds = []
    image_grits_con = []
    image_grits_top = []
    no_det_count = 0
    error_count = 0
    per_image_times = []

    for i, data in enumerate(preloaded):
        result = process_single_image(data, pipe, teds_scorer, grits_fn, logger)
        # Attach attribute flags from manifest for difficulty analysis
        img_id = result.get("image_id", "")
        if img_id in id_to_attrs:
            result["attrs"] = id_to_attrs[img_id]
        # Attach per-table attributes from GT JSON v1 (same order as gt_idx)
        if "error" not in data and "gt_info" in data:
            result["gt_table_attrs"] = [g.get("attributes", {}) for g in data["gt_info"]]
        all_results.append(result)

        status = result.get("status", "error")
        if status == "error":
            error_count += 1
        elif status == "no_detection":
            no_det_count += 1
            image_teds.append(0.0)
            image_grits_con.append(0.0)
            image_grits_top.append(0.0)
        else:
            image_teds.append(result["penalized_teds"])
            image_grits_con.append(result["penalized_grits_con"])
            image_grits_top.append(result["penalized_grits_top"])

        if "time_total_s" in result:
            per_image_times.append(result["time_total_s"])

        # Progress every 10 images or at start/end
        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == len(preloaded):
            avg_t = sum(image_teds) / len(image_teds) if image_teds else 0
            avg_g = sum(image_grits_con) / len(image_grits_con) if image_grits_con else 0
            avg_time = sum(per_image_times) / len(per_image_times) if per_image_times else 0
            elapsed = time.time() - t_infer
            eta = (elapsed / (i + 1)) * (len(preloaded) - i - 1) if i > 0 else 0

            logger.info(
                f"  [{i+1:>{len(str(len(preloaded)))}}/{len(preloaded)}] "
                f"{result['image_id']}  "
                f"TEDS={avg_t:.4f}  GriTS={avg_g:.4f}  "
                f"no_det={no_det_count} err={error_count}  "
                f"avg={avg_time:.2f}s/img  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
            )

    t_infer_total = time.time() - t_infer
    t_total = time.time() - t0

    # ── Summary ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total images processed:  {len(preloaded)}")
    logger.info(f"  Successful:              {len(preloaded) - error_count}")
    logger.info(f"  No table detected:       {no_det_count}")
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
    logger.info(f"  Model loading:   {t_model_load:.1f}s")
    if per_image_times:
        logger.info(f"  Per-image avg:   {sum(per_image_times)/len(per_image_times):.3f}s")
        logger.info(f"  Per-image min:   {min(per_image_times):.3f}s")
        logger.info(f"  Per-image max:   {max(per_image_times):.3f}s")
    logger.info(f"  Inference total: {t_infer_total:.1f}s")
    logger.info(f"  Wall clock:      {t_total:.1f}s")

    # Detection accuracy
    gt_counts = [r.get("num_tables_gt", 0) for r in all_results if r.get("status") != "error"]
    det_counts = [r.get("num_tables_detected", 0) for r in all_results if r.get("status") != "error"]
    if gt_counts:
        exact_match = sum(1 for g, d in zip(gt_counts, det_counts) if g == d)
        logger.info("")
        logger.info("Detection accuracy:")
        logger.info(f"  Exact count match: {exact_match}/{len(gt_counts)} "
                     f"({100*exact_match/len(gt_counts):.1f}%)")
        logger.info(f"  Total GT tables:   {sum(gt_counts)}")
        logger.info(f"  Total detected:    {sum(det_counts)}")

    # ── Per-subset statistics ──
    from collections import defaultdict
    subset_scores = defaultdict(lambda: {"teds": [], "grits_con": [], "grits_top": [],
                                          "count": 0, "no_det": 0, "errors": 0})
    for r in all_results:
        s = r.get("subset", "unknown")
        subset_scores[s]["count"] += 1
        st = r.get("status", "error")
        if st == "error":
            subset_scores[s]["errors"] += 1
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
        logger.info(f"  {'Subset':<8} {'Count':>6} {'NoDet':>6} {'Err':>5} "
                     f"{'TEDS':>8} {'GriTS_C':>8} {'GriTS_T':>8}")
        logger.info(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
        for s_name in sorted(subset_scores.keys()):
            ss = subset_scores[s_name]
            t_avg = sum(ss["teds"]) / len(ss["teds"]) if ss["teds"] else 0
            gc_avg = sum(ss["grits_con"]) / len(ss["grits_con"]) if ss["grits_con"] else 0
            gt_avg = sum(ss["grits_top"]) / len(ss["grits_top"]) if ss["grits_top"] else 0
            logger.info(f"  {s_name:<8} {ss['count']:>6} {ss['no_det']:>6} {ss['errors']:>5} "
                         f"{t_avg:>8.4f} {gc_avg:>8.4f} {gt_avg:>8.4f}")

    # ── Per-attribute difficulty statistics ──
    # Group results by each difficulty attribute dimension
    attr_dimensions = [
        ("T1", ["Yes", "No"]),
        ("T3", ["degraded", "None"]),
        ("T4", ["not_Full", "Full"]),
    ]
    attr_scores = {}  # (attr_name, attr_val) -> {"teds": [], "grits_con": [], ...}
    for r in all_results:
        attrs = r.get("attrs")
        if not attrs:
            continue
        st = r.get("status", "error")
        for attr_name, _ in attr_dimensions:
            val = attrs[attr_name]
            key = (attr_name, val)
            if key not in attr_scores:
                attr_scores[key] = {"teds": [], "grits_con": [], "grits_top": [],
                                     "count": 0, "no_det": 0, "errors": 0}
            bucket = attr_scores[key]
            bucket["count"] += 1
            if st == "error":
                bucket["errors"] += 1
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
        logger.info(f"  {'Attribute':<14} {'Count':>6} {'NoDet':>6} {'Err':>5} "
                     f"{'TEDS':>8} {'GriTS_C':>8} {'GriTS_T':>8}")
        logger.info(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
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
                logger.info(f"  {label:<14} {b['count']:>6} {b['no_det']:>6} {b['errors']:>5} "
                             f"{t_avg:>8.4f} {gc_avg:>8.4f} {gt_avg:>8.4f}")

    # ── Per-table attribute statistics ──
    # Unlike per-image stats above, this groups individual matched GT tables
    # by their own T1/T2/T3/T4 attributes and reports per-table scores.
    table_attr_dimensions = [
        ("T1", ["Yes", "No"]),
        ("T2", [1, 2, 3, 4, 5]),
        ("T3", ["Watermark", "Stain", "Stamp", "None"]),
        ("T4", ["Full"]),  # all non-Full values collected as "not_Full"
    ]
    table_attr_scores = {}  # (attr_name, attr_val) -> {"teds": [], ...}

    def _tbl_attr_bucket(key):
        if key not in table_attr_scores:
            table_attr_scores[key] = {"teds": [], "grits_con": [], "grits_top": [], "count": 0}
        return table_attr_scores[key]

    def _classify_table(ta, attr_name):
        """Get the bucket value for a table's attribute."""
        if attr_name == "T4":
            return ta.get("T4", "Full") if ta.get("T4") == "Full" else "not_Full"
        elif attr_name == "T2":
            return ta.get("T2", 1)
        else:
            return ta.get(attr_name, "")

    for r in all_results:
        gt_ta_list = r.get("gt_table_attrs")
        if not gt_ta_list:
            continue
        st = r.get("status", "error")
        if st == "error":
            continue

        if st == "no_detection":
            # All GT tables on this image got score=0
            for ta in gt_ta_list:
                for attr_name, _ in table_attr_dimensions:
                    val = _classify_table(ta, attr_name)
                    bucket = _tbl_attr_bucket((attr_name, val))
                    bucket["count"] += 1
                    bucket["teds"].append(0.0)
                    bucket["grits_con"].append(0.0)
                    bucket["grits_top"].append(0.0)
        else:
            # Collect scores for matched GT tables
            matched_gt_indices = set()
            for m in r.get("table_matches", []):
                gi = m["gt_idx"]
                matched_gt_indices.add(gi)
                scores = m.get("scores", {})
                teds_val = scores.get("teds", 0.0) or 0.0
                gc_val = scores.get("grits_con", 0.0) or 0.0
                gt_val = scores.get("grits_top", 0.0) or 0.0

                if gi < len(gt_ta_list):
                    ta = gt_ta_list[gi]
                    for attr_name, _ in table_attr_dimensions:
                        val = _classify_table(ta, attr_name)
                        bucket = _tbl_attr_bucket((attr_name, val))
                        bucket["count"] += 1
                        bucket["teds"].append(teds_val)
                        bucket["grits_con"].append(gc_val)
                        bucket["grits_top"].append(gt_val)

            # Unmatched GT tables get score=0
            for gi, ta in enumerate(gt_ta_list):
                if gi in matched_gt_indices:
                    continue
                for attr_name, _ in table_attr_dimensions:
                    val = _classify_table(ta, attr_name)
                    bucket = _tbl_attr_bucket((attr_name, val))
                    bucket["count"] += 1
                    bucket["teds"].append(0.0)
                    bucket["grits_con"].append(0.0)
                    bucket["grits_top"].append(0.0)

    if table_attr_scores:
        logger.info("")
        logger.info("Per-table attribute breakdown (table-level scores):")
        logger.info(f"  {'Attribute':<14} {'Tables':>7} "
                     f"{'TEDS':>8} {'GriTS_C':>8} {'GriTS_T':>8}")
        logger.info(f"  {'-'*14} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
        # Print in order: T1, T2, T3, T4 with non-Full grouped
        print_order = [
            ("T1", "Yes"), ("T1", "No"),
            ("T2", 1), ("T2", 2), ("T2", 3), ("T2", 4), ("T2", 5),
            ("T3", "Watermark"), ("T3", "Stain"), ("T3", "Stamp"), ("T3", "None"),
            ("T4", "not_Full"), ("T4", "Full"),
        ]
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
    # "score-only-matched": only matched (detected) GT tables count toward recognition averages.
    # Detection recall: detected_count / gt_total for each attribute bucket.
    tbl_matched_scores = {}   # (attr_name, val) -> {"teds": [], "grits_con": [], "grits_top": []}
    tbl_recall_counts = {}    # (attr_name, val) -> {"gt_total": 0, "detected": 0}

    def _tbl_matched_bucket(key):
        if key not in tbl_matched_scores:
            tbl_matched_scores[key] = {"teds": [], "grits_con": [], "grits_top": []}
        return tbl_matched_scores[key]

    def _tbl_recall_bucket(key):
        if key not in tbl_recall_counts:
            tbl_recall_counts[key] = {"gt_total": 0, "detected": 0}
        return tbl_recall_counts[key]

    for r in all_results:
        gt_ta_list = r.get("gt_table_attrs")
        if not gt_ta_list:
            continue
        st = r.get("status", "error")
        if st == "error":
            continue

        # Count ALL GT tables toward recall denominator
        matched_gt_indices = set()
        if st != "no_detection":
            for m in r.get("table_matches", []):
                gi = m["gt_idx"]
                matched_gt_indices.add(gi)
                scores = m.get("scores", {})
                teds_val = scores.get("teds", 0.0) or 0.0
                gc_val = scores.get("grits_con", 0.0) or 0.0
                gt_val = scores.get("grits_top", 0.0) or 0.0
                if gi < len(gt_ta_list):
                    ta = gt_ta_list[gi]
                    for attr_name, _ in table_attr_dimensions:
                        val = _classify_table(ta, attr_name)
                        mb = _tbl_matched_bucket((attr_name, val))
                        mb["teds"].append(teds_val)
                        mb["grits_con"].append(gc_val)
                        mb["grits_top"].append(gt_val)

        # All GT tables count toward gt_total; matched ones also count toward detected
        for gi, ta in enumerate(gt_ta_list):
            for attr_name, _ in table_attr_dimensions:
                val = _classify_table(ta, attr_name)
                rb = _tbl_recall_bucket((attr_name, val))
                rb["gt_total"] += 1
                if gi in matched_gt_indices:
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
    output_path = os.path.join(OUTPUT_DIR, f"eval_{n_label}_seed{SEED}_{timestamp}.json")

    # Strip full HTML from results to reduce file size; keep them in a separate field
    results_slim = []
    for r in all_results:
        r_copy = dict(r)
        if "table_matches" in r_copy:
            slim_matches = []
            for m in r_copy["table_matches"]:
                m_copy = dict(m)
                # Keep HTML in log but abbreviate in output
                m_copy["pred_html_len"] = len(m.get("pred_html", ""))
                m_copy["gt_html_len"] = len(m.get("gt_html", ""))
                # Keep full HTML for debugging — comment out next 2 lines to save space
                # m_copy.pop("pred_html", None)
                # m_copy.pop("gt_html", None)
                slim_matches.append(m_copy)
            r_copy["table_matches"] = slim_matches
        results_slim.append(r_copy)

    # Build per-subset summary for JSON output
    subset_summary = {}
    for s_name in sorted(subset_scores.keys()):
        ss = subset_scores[s_name]
        subset_summary[s_name] = {
            "count": ss["count"],
            "no_det": ss["no_det"],
            "errors": ss["errors"],
            "avg_teds": round(sum(ss["teds"]) / len(ss["teds"]), 6) if ss["teds"] else None,
            "avg_grits_con": round(sum(ss["grits_con"]) / len(ss["grits_con"]), 6) if ss["grits_con"] else None,
            "avg_grits_top": round(sum(ss["grits_top"]) / len(ss["grits_top"]), 6) if ss["grits_top"] else None,
        }

    # Build per-attribute summary for JSON output
    attr_summary = {}
    for attr_name, values in attr_dimensions:
        for val in values:
            key = (attr_name, val)
            if key not in attr_scores:
                continue
            b = attr_scores[key]
            attr_summary[f"{attr_name}={val}"] = {
                "count": b["count"],
                "no_det": b["no_det"],
                "errors": b["errors"],
                "avg_teds": round(sum(b["teds"]) / len(b["teds"]), 6) if b["teds"] else None,
                "avg_grits_con": round(sum(b["grits_con"]) / len(b["grits_con"]), 6) if b["grits_con"] else None,
                "avg_grits_top": round(sum(b["grits_top"]) / len(b["grits_top"]), 6) if b["grits_top"] else None,
            }

    # Build per-table attribute summary for JSON output
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

    # Build per-table matched-only recognition + detection recall for JSON output
    table_matched_summary = {}
    for key in print_order:
        rb = tbl_recall_counts.get(key, {"gt_total": 0, "detected": 0})
        mb = tbl_matched_scores.get(key, {"teds": [], "grits_con": [], "grits_top": []})
        gt_total = rb["gt_total"]
        detected = rb["detected"]
        label = f"{key[0]}={key[1]}"
        table_matched_summary[label] = {
            "gt_total": gt_total,
            "detected": detected,
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
            "device": DEVICE,
            "seed": SEED,
            "num_workers": NUM_WORKERS,
            "iou_threshold": IOU_THRESHOLD,
            "det_score_threshold": DET_SCORE_THRESHOLD,
            "timestamp": timestamp,
        },
        "summary": {
            "total_images": len(preloaded),
            "successful": len(preloaded) - error_count,
            "no_table_detected": no_det_count,
            "errors": error_count,
            "avg_teds_penalized": round(sum(image_teds) / len(image_teds), 6) if image_teds else None,
            "avg_grits_con_penalized": round(sum(image_grits_con) / len(image_grits_con), 6) if image_grits_con else None,
            "avg_grits_top_penalized": round(sum(image_grits_top) / len(image_grits_top), 6) if image_grits_top else None,
            "detection_exact_match_rate": round(exact_match / len(gt_counts), 4) if gt_counts else None,
            "wall_clock_s": round(t_total, 1),
            "inference_s": round(t_infer_total, 1),
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
    main()
