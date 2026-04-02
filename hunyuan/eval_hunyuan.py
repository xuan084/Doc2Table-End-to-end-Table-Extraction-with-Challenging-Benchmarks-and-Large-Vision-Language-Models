"""Evaluate HunyuanOCR on table extraction benchmarks.

Two-phase design:
  Phase 1: Inference — call HunyuanOCR via vLLM OpenAI-compatible API
           Supports sharding for multi-GPU parallel inference
           Saves raw predictions to JSONL (one line per image)

  Phase 2: Scoring — load predictions + GT, compute TEDS/GriTS per pair,
           Hungarian matching, then compute all 4 scoring schemes (A/B/C/D)
           with multiple beta values

Usage (multi-GPU):

    # Start vLLM server (one GPU, or use tensor_parallel for speed)
    vllm serve tencent/HunyuanOCR --no-enable-prefix-caching --mm-processor-cache-gb 0 &

    # Phase 1: Inference (can shard across multiple vLLM instances on different ports)
    python eval_hunyuan.py infer \
        --dataset generated \
        --test_jsonl /path/to/test_swift.jsonl \
        --output_dir /path/to/output \
        --api_base http://localhost:8000/v1 \
        --shard_id 0 --n_shards 1

    # Phase 2: Score
    python eval_hunyuan.py score \
        --dataset generated \
        --test_jsonl /path/to/test_swift.jsonl \
        --output_dir /path/to/output \
        --teds_dir /path/to/eval_deps \
        --grits_dir /path/to/eval_deps \
        --n_shards 1 \
        --score_workers 64

Supported datasets:
    --dataset generated    → WildDocTables test split (2000 images)
    --dataset fintabnet    → FinTabNet test (2000 images)
    --dataset pubtables    → PubTables-1M test (2000 images)
"""

import argparse
import base64
import html as html_mod
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL WORKER FUNCTIONS (must be picklable for ProcessPoolExecutor)
# ═══════════════════════════════════════════════════════════════════════════════

def _teds_worker(task, teds_dir):
    """Compute TEDS for one (pred, gt) pair. Runs in a separate process."""
    import sys
    sys.path.insert(0, teds_dir)
    from teds import TEDS
    img_idx, pi, gj, pred_h, gt_h = task
    try:
        scorer = TEDS(n_jobs=1, ignore_nodes="", structure_only=False)
        pn = "<html><body>" + pred_h + "</body></html>" if pred_h else "<html><body><table></table></body></html>"
        gn = "<html><body>" + gt_h + "</body></html>"
        score = scorer.evaluate(gn, pn)
    except Exception:
        score = 0.0
    return (img_idx, pi, gj, score)


def _grits_worker(task, grits_dir):
    """Compute GriTS for one matched (pred, gt) pair. Runs in a separate process."""
    import sys
    sys.path.insert(0, grits_dir)
    from grits import grits_from_html
    img_idx, pi, gj, pred_h, gt_h = task
    try:
        g = grits_from_html(gt_h, pred_h)
        return (img_idx, pi, gj, {"grits_con": g["grits_con"], "grits_top": g["grits_top"]})
    except Exception:
        return (img_idx, pi, gj, {"grits_con": 0.0, "grits_top": 0.0})


# ═══════════════════════════════════════════════════════════════════════════════
# HTML EXTRACTION & NORMALIZATION (shared with other eval scripts)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_tables_from_response(text: str) -> List[str]:
    """Extract ALL <table>...</table> fragments from model response."""
    # First try inside code blocks
    code_blocks = re.findall(r"```(?:html)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        tables = []
        for block in code_blocks:
            tables.extend(re.findall(r"(<table[^>]*>.*?</table>)", block, re.DOTALL | re.IGNORECASE))
        if tables:
            return [t.strip() for t in tables]
    # Fallback: find in raw text
    tables = re.findall(r"(<table[^>]*>.*?</table>)", text, re.DOTALL | re.IGNORECASE)
    if tables:
        return [t.strip() for t in tables]
    return []


def normalize_html(html_str):
    """Normalize table HTML to canonical form."""
    html_str = re.sub(r'<th\b', '<td', html_str)
    html_str = html_str.replace('</th>', '</td>')
    html_str = html_str.replace('<thead>', '').replace('</thead>', '')
    html_str = html_str.replace('<tbody>', '').replace('</tbody>', '')
    html_str = re.sub(r'<caption>.*?</caption>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    html_str = re.sub(r'</?html[^>]*>', '', html_str, flags=re.IGNORECASE)
    html_str = re.sub(r'<head>.*?</head>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    html_str = re.sub(r'</?body[^>]*>', '', html_str, flags=re.IGNORECASE)
    html_str = re.sub(r'<style>.*?</style>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    html_str = re.sub(r'<meta[^>]*/>', '', html_str, flags=re.IGNORECASE)
    html_str = re.sub(r'<br\s*/?>', '<br/>', html_str, flags=re.IGNORECASE)
    _XML_ENTITIES = {'amp', 'lt', 'gt', 'quot', 'apos'}
    html_str = re.sub(
        r'&([a-zA-Z]+);',
        lambda m: m.group(0) if m.group(1) in _XML_ENTITIES else html_mod.unescape(m.group(0)),
        html_str
    )
    return html_str.strip()


def wrap_for_teds(html_str):
    return "<html><body>" + html_str + "</body></html>"


def count_cells(html_str: str) -> int:
    """Count <td> cells in an HTML table string."""
    return len(re.findall(r'<td[\s>]', html_str, re.IGNORECASE))


# ═══════════════════════════════════════════════════════════════════════════════
# MATCHING: Hungarian (optimal) instead of greedy
# ═══════════════════════════════════════════════════════════════════════════════

def hungarian_match_tables(pred_htmls, gt_htmls, score_matrix):
    """Optimal bipartite matching using Hungarian algorithm.

    Args:
        pred_htmls: list of N pred HTML strings
        gt_htmls: list of M GT HTML strings
        score_matrix: dict {(pi, gj): score} for all pairs

    Returns:
        list of (pred_idx, gt_idx) matched pairs
    """
    from scipy.optimize import linear_sum_assignment

    N = len(pred_htmls)
    M = len(gt_htmls)
    if N == 0 or M == 0:
        return []

    # Build cost matrix (scipy minimizes, so negate scores)
    cost = np.zeros((N, M))
    for pi in range(N):
        for gj in range(M):
            cost[pi, gj] = -score_matrix.get((pi, gj), 0.0)

    row_ind, col_ind = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]
    return matches


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING SCHEMES A/B/C/D
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scheme_A(matches, scores_dict, M, N):
    """Scheme A: avg_matched × min(M,N)/max(M,N)"""
    if M == 0 and N == 0:
        return {"score": 1.0}
    if M == 0 or N == 0:
        return {"score": 0.0}
    K = len(matches)
    if K == 0:
        return {"score": 0.0}
    sum_s = sum(scores_dict[(pi, gj)] for pi, gj in matches)
    avg = sum_s / K
    penalty = min(M, N) / max(M, N)
    return {"score": round(avg * penalty, 6), "avg_matched": round(avg, 6), "penalty": round(penalty, 4)}


def compute_scheme_B(matches, scores_dict, M, N, betas=[1.0, 1.5, 2.0, 3.0]):
    """Scheme B: unweighted P/R/F_beta"""
    if M == 0 and N == 0:
        return {f"f{b}": 1.0 for b in betas} | {"P": 1.0, "R": 1.0}
    if M == 0 or N == 0:
        return {f"f{b}": 0.0 for b in betas} | {"P": 0.0, "R": 0.0}

    sum_s = sum(scores_dict[(pi, gj)] for pi, gj in matches)
    P = sum_s / N
    R = sum_s / M

    result = {"P": round(P, 6), "R": round(R, 6)}
    for b in betas:
        if P + R > 0:
            fb = (1 + b**2) * P * R / (b**2 * P + R)
        else:
            fb = 0.0
        result[f"f{b}"] = round(fb, 6)
    return result


def compute_scheme_C(matches, scores_dict, M, N,
                     gt_cells: List[int], pred_cells: List[int],
                     matched_gt_indices: set, matched_pred_indices: set):
    """Scheme C: weighted min/max. M>=N uses GT weights, N>M uses pred weights."""
    if M == 0 and N == 0:
        return {"score": 1.0}
    if M == 0 or N == 0:
        return {"score": 0.0}

    if M >= N:
        # Use GT weights
        total = sum(gt_cells)
        if total == 0:
            return {"score": 0.0}
        score = 0.0
        for pi, gj in matches:
            w = gt_cells[gj] / total
            score += w * scores_dict[(pi, gj)]
    else:
        # N > M: use pred weights
        total = sum(pred_cells)
        if total == 0:
            return {"score": 0.0}
        score = 0.0
        for pi, gj in matches:
            w = pred_cells[pi] / total
            score += w * scores_dict[(pi, gj)]

    return {"score": round(score, 6)}


def compute_scheme_D(matches, scores_dict, M, N,
                     gt_cells: List[int], pred_cells: List[int],
                     matched_gt_indices: set, matched_pred_indices: set,
                     betas=[1.0, 1.5, 2.0, 3.0]):
    """Scheme D (PWTF): weighted P/R/F_beta with mixed precision weights."""
    if M == 0 and N == 0:
        return {f"f{b}": 1.0 for b in betas} | {"P": 1.0, "R": 1.0}
    if M == 0 or N == 0:
        return {f"f{b}": 0.0 for b in betas} | {"P": 0.0, "R": 0.0}

    # --- Recall: all GT weights ---
    gt_total = sum(gt_cells)
    if gt_total == 0:
        R = 0.0
    else:
        R = sum((gt_cells[gj] / gt_total) * scores_dict[(pi, gj)] for pi, gj in matches)

    # --- Precision: mixed weights ---
    # Matched pred: use corresponding GT cell count
    # Unmatched pred: use pred's own cell count
    match_map = {}  # pi -> gj
    for pi, gj in matches:
        match_map[pi] = gj

    denom = 0.0
    for pi in range(N):
        if pi in match_map:
            denom += gt_cells[match_map[pi]]
        else:
            denom += pred_cells[pi]

    if denom == 0:
        P = 0.0
    else:
        P = 0.0
        for pi, gj in matches:
            weight = gt_cells[gj] / denom
            P += weight * scores_dict[(pi, gj)]

    result = {"P": round(P, 6), "R": round(R, 6)}
    for b in betas:
        if P + R > 0:
            fb = (1 + b**2) * P * R / (b**2 * P + R)
        else:
            fb = 0.0
        result[f"f{b}"] = round(fb, 6)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_path(sample: dict, image_base_dir: str = None) -> str:
    """Extract image path from sample, handling 3 different formats:

    Format A (PubTables/FinTabNet test_swift.jsonl):
        {"messages": [...], "images": ["/abs/path/to/image.jpg"]}
        -> s["images"][0]

    Format B (Generated dataset qwen_sft/test.jsonl):
        {"messages": [sys, {"role":"user","content":[{"type":"image","image":"images/img_XXX.png"}, ...]}, gt]}
        -> s["messages"][1]["content"][0]["image"]  (relative path)

    Args:
        sample: one JSONL record
        image_base_dir: base directory to prepend to relative paths
    Returns:
        absolute image path
    """
    # Format A: top-level "images" field
    if "images" in sample and sample["images"]:
        img_path = sample["images"][0]
    else:
        # Format B: nested in user message content list
        user_content = sample["messages"][1]["content"]
        if isinstance(user_content, list):
            for item in user_content:
                if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                    img_path = item["image"]
                    break
            else:
                return ""
        else:
            return ""

    # Handle relative paths
    if not os.path.isabs(img_path):
        if image_base_dir:
            img_path = os.path.join(image_base_dir, img_path)

    return img_path


def get_gt_text(sample: dict) -> str:
    """Extract GT HTML text from sample. Works for all 3 datasets."""
    gt_msg = sample["messages"][2]
    content = gt_msg["content"]
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # If GT is a list (shouldn't happen but handle it)
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item["text"]
            elif isinstance(item, str):
                return item
    return ""


def run_inference_shard(args, samples):
    """Run inference for one shard."""
    from openai import OpenAI

    total = len(samples)
    shard_size = (total + args.n_shards - 1) // args.n_shards
    start = args.shard_id * shard_size
    end = min(start + shard_size, total)
    shard_samples = samples[start:end]

    print(f"Shard {args.shard_id}: [{start}, {end}) = {len(shard_samples)} samples", flush=True)

    pred_path = os.path.join(args.output_dir, f"predictions_shard{args.shard_id}.jsonl")

    # Resume: load existing predictions
    existing = {}
    if os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                existing[rec["idx"]] = rec["pred_text"]
        print(f"  Resumed: {len(existing)} existing predictions", flush=True)

    client = OpenAI(api_key="EMPTY", base_url=args.api_base, timeout=600)

    # Prompt for table extraction
    prompt = "Parse all tables in the image into HTML format. Output each table as a separate <table>...</table> block. Use <td> for all cells (never <th>). Include colspan and rowspan attributes where needed."

    t_infer_start = time.time()
    new_count = 0

    with open(pred_path, "a", encoding="utf-8") as fout:
        for local_i, sample in enumerate(shard_samples):
            global_idx = start + local_i
            if global_idx in existing:
                continue

            # Get image path (handles both formats)
            img_path = get_image_path(sample, args.image_base_dir)
            if not img_path or not os.path.exists(img_path):
                print(f"  SKIP idx={global_idx}: image not found: {img_path}", flush=True)
                fout.write(json.dumps({"idx": global_idx, "pred_text": ""}, ensure_ascii=False) + "\n")
                new_count += 1
                continue

            try:
                b64 = encode_image_base64(img_path)
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt}
                    ]}
                ]
                response = client.chat.completions.create(
                    model="tencent/HunyuanOCR",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=16384,
                    extra_body={"top_k": 1, "repetition_penalty": 1.0}
                )
                pred_text = response.choices[0].message.content or ""
            except Exception as e:
                print(f"  ERROR idx={global_idx}: {e}", flush=True)
                pred_text = ""

            fout.write(json.dumps({"idx": global_idx, "pred_text": pred_text}, ensure_ascii=False) + "\n")
            fout.flush()
            new_count += 1

            done = local_i + 1
            if done % 20 == 0 or done == len(shard_samples):
                skipped = done - new_count
                elapsed = time.time() - t_infer_start
                rate = new_count / elapsed if elapsed > 0 else 0
                remaining = len(shard_samples) - done
                eta = remaining / (done / elapsed) if elapsed > 0 and done > 0 else 0
                print(f"  [{done}/{len(shard_samples)}] "
                      f"(skipped={skipped}, new={new_count}) "
                      f"{rate:.2f} img/s | "
                      f"Elapsed: {elapsed/60:.1f}min | "
                      f"ETA: {eta/60:.1f}min", flush=True)

    print(f"Shard {args.shard_id} done.", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def load_teds(teds_dir):
    sys.path.insert(0, teds_dir)
    from teds import TEDS
    return TEDS


def load_grits(grits_dir):
    sys.path.insert(0, grits_dir)
    from grits import grits_from_html
    return grits_from_html


def run_scoring(args, samples):
    """Merge shards, compute TEDS/GriTS matrices, Hungarian match, score all schemes."""

    total = len(samples)
    betas = [1.0, 1.5, 2.0, 3.0]

    # ── Merge predictions ──
    print("Merging predictions...", flush=True)
    all_preds = {}
    for sid in range(args.n_shards):
        sp = os.path.join(args.output_dir, f"predictions_shard{sid}.jsonl")
        if os.path.exists(sp):
            count = 0
            with open(sp, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    all_preds[rec["idx"]] = rec["pred_text"]
                    count += 1
            print(f"  Shard {sid}: {count} predictions", flush=True)

    # Also load old merged file if exists
    merged_path = os.path.join(args.output_dir, "predictions_merged.jsonl")
    if os.path.exists(merged_path):
        with open(merged_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["idx"] not in all_preds:
                    all_preds[rec["idx"]] = rec["pred_text"]

    missing = [i for i in range(total) if i not in all_preds]
    print(f"  Total merged: {len(all_preds)}/{total}", flush=True)
    if missing:
        print(f"  WARNING: {len(missing)} missing indices", flush=True)
        # Fill missing with empty
        for i in missing:
            all_preds[i] = ""

    # Save merged
    with open(merged_path, "w", encoding="utf-8") as f:
        for i in range(total):
            f.write(json.dumps({"idx": i, "pred_text": all_preds.get(i, "")}, ensure_ascii=False) + "\n")

    # ── Extract and normalize tables ──
    print("Extracting tables...", flush=True)
    all_pred_htmls = []  # [img_idx] -> [normalized html strings]
    all_gt_htmls = []
    all_pred_cells = []  # [img_idx] -> [cell_count per table]
    all_gt_cells = []

    gt_texts = [get_gt_text(s) for s in samples]

    for i in range(total):
        # Pred
        pred_raw = extract_tables_from_response(all_preds.get(i, ""))
        pred_norm = [normalize_html(h) for h in pred_raw]
        all_pred_htmls.append(pred_norm)
        all_pred_cells.append([count_cells(h) for h in pred_norm])

        # GT
        gt_raw = extract_tables_from_response(gt_texts[i])
        gt_norm = [normalize_html(h) for h in gt_raw]
        all_gt_htmls.append(gt_norm)
        all_gt_cells.append([count_cells(h) for h in gt_norm])

    # ── Build TEDS tasks ──
    print("Loading scorers...", flush=True)
    TEDS_cls = load_teds(args.teds_dir)
    grits_fn = load_grits(args.grits_dir)
    print("Scorers ready.", flush=True)

    image_info = []  # [(M, N), ...]
    teds_tasks = []  # (img_idx, pi, gj, pred_html, gt_html)

    for i in range(total):
        ph = all_pred_htmls[i]
        gh = all_gt_htmls[i]
        M, N = len(gh), len(ph)
        image_info.append((M, N))
        if M > 0 and N > 0:
            for pi in range(N):
                for gj in range(M):
                    teds_tasks.append((i, pi, gj, ph[pi], gh[gj]))

    # ── Parallel TEDS computation (multiprocess to bypass GIL) ──
    # Resume: load cached TEDS scores if available
    teds_cache_path = os.path.join(args.output_dir, "teds_cache.jsonl")
    teds_all = {}  # (img_idx, pi, gj) -> score
    if os.path.exists(teds_cache_path):
        with open(teds_cache_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                teds_all[(rec["i"], rec["pi"], rec["gj"])] = rec["s"]
        print(f"  TEDS cache loaded: {len(teds_all)} pairs", flush=True)

    # Filter out already-computed pairs
    teds_todo = [t for t in teds_tasks if (t[0], t[1], t[2]) not in teds_all]
    print(f"Scoring {len(teds_todo)} TEDS pairs ({len(teds_tasks) - len(teds_todo)} cached) "
          f"with {args.score_workers} processes...", flush=True)
    t0 = time.time()

    if teds_todo:
        cache_f = open(teds_cache_path, "a", encoding="utf-8")
        with ProcessPoolExecutor(max_workers=args.score_workers) as pool:
            futures = {pool.submit(_teds_worker, t, args.teds_dir): t for t in teds_todo}
            done = 0
            for fut in as_completed(futures):
                img_idx, pi, gj, score = fut.result()
                teds_all[(img_idx, pi, gj)] = score
                cache_f.write(json.dumps({"i": img_idx, "pi": pi, "gj": gj, "s": score}) + "\n")
                done += 1
                if done % 200 == 0:
                    cache_f.flush()
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(teds_todo) - done) / rate if rate > 0 else 0
                    print(f"  TEDS: {done}/{len(teds_todo)} ({rate:.1f} pairs/s, ETA {eta/60:.1f}min)", flush=True)
        cache_f.close()

    print(f"  TEDS done in {time.time()-t0:.1f}s (total {len(teds_all)} pairs)", flush=True)

    # ── Hungarian matching per image ──
    print("Hungarian matching...", flush=True)
    per_image_matches = {}

    grits_tasks = []
    for i in range(total):
        M, N = image_info[i]
        if M == 0 or N == 0:
            per_image_matches[i] = []
            continue
        # Build score matrix for this image
        score_matrix = {}
        for pi in range(N):
            for gj in range(M):
                score_matrix[(pi, gj)] = teds_all.get((i, pi, gj), 0.0)
        matches = hungarian_match_tables(all_pred_htmls[i], all_gt_htmls[i], score_matrix)
        per_image_matches[i] = matches
        for pi, gj in matches:
            grits_tasks.append((i, pi, gj, all_pred_htmls[i][pi], all_gt_htmls[i][gj]))

    # ── Parallel GriTS computation (multiprocess to bypass GIL) ──
    # Resume: load cached GriTS scores if available
    grits_cache_path = os.path.join(args.output_dir, "grits_cache.jsonl")
    grits_all = {}
    if os.path.exists(grits_cache_path):
        with open(grits_cache_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                grits_all[(rec["i"], rec["pi"], rec["gj"])] = {
                    "grits_con": rec["gc"], "grits_top": rec["gt"]
                }
        print(f"  GriTS cache loaded: {len(grits_all)} pairs", flush=True)

    grits_todo = [t for t in grits_tasks if (t[0], t[1], t[2]) not in grits_all]
    print(f"Scoring {len(grits_todo)} GriTS pairs ({len(grits_tasks) - len(grits_todo)} cached) "
          f"with {args.score_workers} processes...", flush=True)
    t1 = time.time()

    if grits_todo:
        cache_g = open(grits_cache_path, "a", encoding="utf-8")
        with ProcessPoolExecutor(max_workers=args.score_workers) as pool:
            futures = {pool.submit(_grits_worker, t, args.grits_dir): t for t in grits_todo}
            done = 0
            for fut in as_completed(futures):
                img_idx, pi, gj, scores = fut.result()
                grits_all[(img_idx, pi, gj)] = scores
                cache_g.write(json.dumps({
                    "i": img_idx, "pi": pi, "gj": gj,
                    "gc": scores["grits_con"], "gt": scores["grits_top"]
                }) + "\n")
                done += 1
                if done % 200 == 0:
                    cache_g.flush()
                    print(f"  GriTS: {done}/{len(grits_todo)}", flush=True)
        cache_g.close()

    print(f"  GriTS done in {time.time()-t1:.1f}s (total {len(grits_all)} pairs)", flush=True)

    # ── Compute all 4 schemes for each image ──
    print("Computing scores (schemes A/B/C/D)...", flush=True)

    all_results = []
    for i in range(total):
        M, N = image_info[i]
        matches = per_image_matches.get(i, [])
        matched_gt = {gj for _, gj in matches}
        matched_pred = {pi for pi, _ in matches}

        gt_c = all_gt_cells[i]
        pred_c = all_pred_cells[i]

        # Per-metric scores
        for metric_name, score_source in [("teds", teds_all), ("grits_con", None), ("grits_top", None)]:
            if metric_name == "teds":
                s_dict = {(pi, gj): teds_all.get((i, pi, gj), 0.0) for pi, gj in matches}
            else:
                s_dict = {}
                for pi, gj in matches:
                    g = grits_all.get((i, pi, gj), {"grits_con": 0.0, "grits_top": 0.0})
                    s_dict[(pi, gj)] = g[metric_name]

            # Also need full matrix for scheme A avg
            full_s_dict = dict(s_dict)

            a = compute_scheme_A(matches, full_s_dict, M, N)
            b = compute_scheme_B(matches, full_s_dict, M, N, betas)
            c = compute_scheme_C(matches, full_s_dict, M, N, gt_c, pred_c, matched_gt, matched_pred)
            d = compute_scheme_D(matches, full_s_dict, M, N, gt_c, pred_c, matched_gt, matched_pred, betas)

            if i == 0 or metric_name == "teds":
                # Store in result dict
                pass

        # Build per-image result
        def _scores_for_metric(metric_name):
            if metric_name == "teds":
                s_dict = {(pi, gj): teds_all.get((i, pi, gj), 0.0) for pi, gj in matches}
            else:
                s_dict = {(pi, gj): grits_all.get((i, pi, gj), {}).get(metric_name, 0.0) for pi, gj in matches}
            return s_dict

        result = {
            "idx": i,
            "M": M, "N": N,
            "det_exact": 1 if M == N else 0,
            "table_matches": [
                {
                    "pred_idx": pi, "gt_idx": gj,
                    "scores": {
                        "teds": teds_all.get((i, pi, gj), 0.0),
                        "grits_con": grits_all.get((i, pi, gj), {}).get("grits_con", 0.0),
                        "grits_top": grits_all.get((i, pi, gj), {}).get("grits_top", 0.0),
                    },
                    "pred_cells": pred_c[pi] if pi < len(pred_c) else 0,
                    "gt_cells": gt_c[gj] if gj < len(gt_c) else 0,
                }
                for pi, gj in matches
            ],
            "gt_cells": gt_c,
            "pred_cells": pred_c,
        }

        # Compute all schemes for all metrics
        for metric in ["teds", "grits_con", "grits_top"]:
            s_dict = _scores_for_metric(metric)
            result[f"A_{metric}"] = compute_scheme_A(matches, s_dict, M, N)
            result[f"B_{metric}"] = compute_scheme_B(matches, s_dict, M, N, betas)
            result[f"C_{metric}"] = compute_scheme_C(matches, s_dict, M, N, gt_c, pred_c, matched_gt, matched_pred)
            result[f"D_{metric}"] = compute_scheme_D(matches, s_dict, M, N, gt_c, pred_c, matched_gt, matched_pred, betas)

        all_results.append(result)

    # ── Aggregate ──
    print("Aggregating...", flush=True)

    def _avg(key_fn):
        vals = [key_fn(r) for r in all_results]
        return round(sum(vals) / len(vals), 6) if vals else 0.0

    summary = {
        "dataset": args.dataset,
        "n_images": total,
        "n_no_pred": sum(1 for r in all_results if r["N"] == 0),
        "det_exact_match": _avg(lambda r: r["det_exact"]),
        "matching": "hungarian",
        "betas": betas,
    }

    for metric in ["teds", "grits_con", "grits_top"]:
        summary[f"A_{metric}"] = _avg(lambda r, m=metric: r[f"A_{m}"]["score"])
        for b in betas:
            summary[f"B_{metric}_f{b}"] = _avg(lambda r, m=metric, bb=b: r[f"B_{m}"][f"f{bb}"])
        summary[f"B_{metric}_P"] = _avg(lambda r, m=metric: r[f"B_{m}"]["P"])
        summary[f"B_{metric}_R"] = _avg(lambda r, m=metric: r[f"B_{m}"]["R"])
        summary[f"C_{metric}"] = _avg(lambda r, m=metric: r[f"C_{m}"]["score"])
        for b in betas:
            summary[f"D_{metric}_f{b}"] = _avg(lambda r, m=metric, bb=b: r[f"D_{m}"][f"f{bb}"])
        summary[f"D_{metric}_P"] = _avg(lambda r, m=metric: r[f"D_{m}"]["P"])
        summary[f"D_{metric}_R"] = _avg(lambda r, m=metric: r[f"D_{m}"]["R"])

    # ── Per-S / Per-T attribute slicing (generated dataset only) ──
    if getattr(args, "manifest", None) and os.path.exists(args.manifest):
        print("Loading manifest for attribute slicing...", flush=True)
        manifest = {}
        with open(args.manifest, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                manifest[rec["image_id"]] = rec

        # Map each test sample index -> image_id
        test_img_ids = []
        for s in samples:
            img = get_image_path(s, getattr(args, "image_base_dir", None))
            img_id = os.path.splitext(os.path.basename(img))[0] if img else ""
            test_img_ids.append(img_id)

        # ── Per-Subset (S1-S4) — page-level scores ──
        subset_groups = defaultdict(list)  # subset -> [result indices]
        for i, img_id in enumerate(test_img_ids):
            m = manifest.get(img_id)
            if m:
                subset_groups[m["subset"]].append(i)

        summary["per_subset"] = {}
        for subset in sorted(subset_groups.keys()):
            indices = subset_groups[subset]
            sub_results = [all_results[i] for i in indices]
            sub_avg = lambda fn: round(sum(fn(r) for r in sub_results) / len(sub_results), 6) if sub_results else 0.0
            entry = {"count": len(indices), "det_exact": sub_avg(lambda r: r["det_exact"])}
            for metric in ["teds", "grits_con"]:
                entry[f"A_{metric}"] = sub_avg(lambda r, m=metric: r[f"A_{m}"]["score"])
                entry[f"D_{metric}_f2.0"] = sub_avg(lambda r, m=metric: r[f"D_{m}"]["f2.0"])
                entry[f"D_{metric}_P"] = sub_avg(lambda r, m=metric: r[f"D_{m}"]["P"])
                entry[f"D_{metric}_R"] = sub_avg(lambda r, m=metric: r[f"D_{m}"]["R"])
            summary["per_subset"][subset] = entry

        # ── Per-T attribute — table-level scores (matched tables only) ──
        # Collect per-table scores with their attributes
        table_scores_by_attr = defaultdict(lambda: defaultdict(list))
        # {attr_name: {attr_value: [(teds, grits_con), ...]}}

        for i, img_id in enumerate(test_img_ids):
            m = manifest.get(img_id)
            if not m:
                continue
            matches = per_image_matches.get(i, [])
            gt_tables_manifest = m.get("tables", [])

            for pi, gj in matches:
                teds_s = teds_all.get((i, pi, gj), 0.0)
                gcon_s = grits_all.get((i, pi, gj), {}).get("grits_con", 0.0)

                # Get GT table attributes from manifest
                if gj < len(gt_tables_manifest):
                    gt_attr = gt_tables_manifest[gj]
                    table_scores_by_attr["T1"][gt_attr.get("T1", "?")].append((teds_s, gcon_s))
                    table_scores_by_attr["T2"][gt_attr.get("T2", "?")].append((teds_s, gcon_s))
                    table_scores_by_attr["T4"][gt_attr.get("T4", "?")].append((teds_s, gcon_s))

                # T3 is page-level (from manifest root)
                table_scores_by_attr["T3"][m.get("T3", "?")].append((teds_s, gcon_s))

        summary["per_attribute"] = {}
        for attr_name in ["T1", "T2", "T3", "T4"]:
            attr_dict = {}
            for attr_val, scores_list in sorted(table_scores_by_attr[attr_name].items(), key=lambda x: str(x[0])):
                n = len(scores_list)
                avg_teds = round(sum(s[0] for s in scores_list) / n, 6) if n else 0.0
                avg_gcon = round(sum(s[1] for s in scores_list) / n, 6) if n else 0.0
                attr_dict[str(attr_val)] = {"count": n, "teds": avg_teds, "grits_con": avg_gcon}
            summary["per_attribute"][attr_name] = attr_dict

        print(f"  Per-subset: {list(summary['per_subset'].keys())}", flush=True)
        print(f"  Per-attribute: T1={list(summary['per_attribute']['T1'].keys())}, "
              f"T2={list(summary['per_attribute']['T2'].keys())}, "
              f"T3={list(summary['per_attribute']['T3'].keys())}, "
              f"T4={list(summary['per_attribute']['T4'].keys())}", flush=True)

    # ── Save ──
    detail_path = os.path.join(args.output_dir, "eval_detail.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Print ──
    print("\n" + "=" * 70, flush=True)
    print(f"Results: {args.dataset} ({total} images)", flush=True)
    print(f"Detection exact match: {summary['det_exact_match']:.4f}", flush=True)
    print(f"No prediction: {summary['n_no_pred']}", flush=True)
    print(f"Matching: Hungarian", flush=True)
    print()

    header = f"{'Metric':<12} {'A(min/max)':>10} {'B_F1':>8} {'B_F2':>8} {'C(wtd)':>10} {'D_F1':>8} {'D_F2':>8} {'D_P':>8} {'D_R':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for metric in ["teds", "grits_con", "grits_top"]:
        print(f"{metric:<12} "
              f"{summary[f'A_{metric}']:>10.4f} "
              f"{summary[f'B_{metric}_f1.0']:>8.4f} "
              f"{summary[f'B_{metric}_f2.0']:>8.4f} "
              f"{summary[f'C_{metric}']:>10.4f} "
              f"{summary[f'D_{metric}_f1.0']:>8.4f} "
              f"{summary[f'D_{metric}_f2.0']:>8.4f} "
              f"{summary[f'D_{metric}_P']:>8.4f} "
              f"{summary[f'D_{metric}_R']:>8.4f}",
              flush=True)

    # ── Print per-S / per-T if available ──
    if "per_subset" in summary:
        print("\nPer-Subset (page-level, scheme D F2):", flush=True)
        print(f"  {'Subset':<6} {'Count':>6} {'Det%':>6} {'TEDS':>8} {'GCon':>8} {'D_P':>8} {'D_R':>8}", flush=True)
        print(f"  {'-'*52}", flush=True)
        for subset, entry in sorted(summary["per_subset"].items()):
            print(f"  {subset:<6} {entry['count']:>6} {entry['det_exact']:>6.2%} "
                  f"{entry['D_teds_f2.0']:>8.4f} {entry['D_grits_con_f2.0']:>8.4f} "
                  f"{entry['D_teds_P']:>8.4f} {entry['D_teds_R']:>8.4f}", flush=True)

    if "per_attribute" in summary:
        for attr_name in ["T1", "T2", "T3", "T4"]:
            print(f"\nPer-{attr_name} (table-level, matched only):", flush=True)
            print(f"  {'Value':<30} {'Count':>6} {'TEDS':>8} {'GCon':>8}", flush=True)
            print(f"  {'-'*54}", flush=True)
            for val, entry in summary["per_attribute"][attr_name].items():
                print(f"  {val:<30} {entry['count']:>6} {entry['teds']:>8.4f} {entry['grits_con']:>8.4f}", flush=True)

    print(f"\nSaved: {summary_path}", flush=True)
    print(f"Detail: {detail_path}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate HunyuanOCR on table extraction")
    sub = parser.add_subparsers(dest="cmd")

    # Infer subcommand
    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--dataset", required=True, choices=["generated", "fintabnet", "pubtables"])
    p_infer.add_argument("--test_jsonl", required=True)
    p_infer.add_argument("--output_dir", required=True)
    p_infer.add_argument("--api_base", default="http://localhost:8000/v1")
    p_infer.add_argument("--shard_id", type=int, default=0)
    p_infer.add_argument("--n_shards", type=int, default=1)
    p_infer.add_argument("--image_base_dir", default=None,
                         help="Base directory for relative image paths (needed for generated dataset). "
                              "E.g. /path/to/data for images/img_XXX.png")

    # Score subcommand
    p_score = sub.add_parser("score")
    p_score.add_argument("--dataset", required=True, choices=["generated", "fintabnet", "pubtables"])
    p_score.add_argument("--test_jsonl", required=True)
    p_score.add_argument("--output_dir", required=True)
    p_score.add_argument("--teds_dir", required=True)
    p_score.add_argument("--grits_dir", required=True)
    p_score.add_argument("--n_shards", type=int, default=1)
    p_score.add_argument("--score_workers", type=int, default=16)
    p_score.add_argument("--manifest", default=None,
                         help="Path to manifest.jsonl for per-S/per-T attribute slicing "
                              "(only for generated dataset)")

    args = parser.parse_args()

    # Load test data
    print(f"Loading {args.test_jsonl}...", flush=True)
    samples = []
    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"  {len(samples)} samples loaded", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.cmd == "infer":
        run_inference_shard(args, samples)
    elif args.cmd == "score":
        run_scoring(args, samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
