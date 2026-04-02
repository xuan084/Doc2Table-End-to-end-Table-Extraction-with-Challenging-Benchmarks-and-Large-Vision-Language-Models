"""Evaluate fine-tuned Qwen2.5-VL LoRA model on DSW test set.

Loads the LoRA adapter via SWIFT, runs inference on 2000 test samples,
scores with TEDS + GriTS (greedy matching), and prints summary.

Usage (on DSW):
    # Step 1: Copy scoring modules to DSW
    #   scp src/grits.py  SERVER:/path/to/eval_deps/
    #   scp unitable/unitable/src/utils/teds.py  SERVER:/path/to/eval_deps/
    #   scp src/postprocess.py  SERVER:/path/to/eval_deps/
    #   scp this script to  SERVER:eval_finetuned_on_dsw.py
    #
    # Step 2: pip install apted lxml
    #
    # Step 3: Run:
    python eval_finetuned_on_dsw.py \
        --model /path/to/Qwen2.5-VL-7B-Instruct \
        --adapter /path/to/lora-checkpoint \
        --test_jsonl /path/to/test.jsonl \
        --output_dir /path/to/output \
        --teds_dir /path/to/eval_deps \
        --grits_dir /path/to/eval_deps

Scoring logic identical to run_llm_on_generated.py:
  - extract_tables_from_response() for parsing
  - normalize_html() for canonicalization
  - compute_scores() for TEDS + GriTS
  - greedy_match_tables_by_teds() for multi-table matching
"""

import argparse
import html as html_mod
import json
import os
import re
import sys
import time
import threading
from pathlib import Path
from typing import List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ═══════════════════════════════════════════════════════════════════════════════
# HTML EXTRACTION FROM MODEL RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

def extract_tables_from_response(text: str) -> List[str]:
    """Extract ALL <table>...</table> fragments from model response."""
    # First, unwrap code blocks if present
    code_blocks = re.findall(r"```(?:html)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if code_blocks:
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
# HTML NORMALIZATION & SCORING (identical to run_llm_on_generated.py)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_html(html_str):
    """Normalize table HTML to canonical form: all <td>, no <thead>/<tbody>."""
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
    """Wrap <table> HTML with <html><body> for TEDS."""
    return "<html><body>" + html_str + "</body></html>"


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
# GREEDY TEDS-BASED MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def greedy_match_tables_by_teds(pred_htmls, gt_htmls, teds_matrix):
    """Greedy best-match: for each pred table, find the GT with highest TEDS."""
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
# LOAD SCORERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_teds(teds_dir):
    """Load TEDS scorer. teds_dir should contain teds.py."""
    if teds_dir not in sys.path:
        sys.path.insert(0, teds_dir)
    from teds import TEDS
    return TEDS(n_jobs=1, ignore_nodes="", structure_only=False)


def load_grits(grits_dir):
    """Load GriTS scorer. grits_dir should contain grits.py."""
    if grits_dir not in sys.path:
        sys.path.insert(0, grits_dir)
    from grits import grits_from_html
    return grits_from_html


# ═══════════════════════════════════════════════════════════════════════════════
# SWIFT INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def load_swift_engine(model_dir, adapter_dir=None, infer_batch_size=4):
    """Load model with optional LoRA adapter via SWIFT for inference."""
    os.environ.setdefault("MAX_PIXELS", "1003520")

    from swift import TransformersEngine, InferRequest, RequestConfig

    kwargs = {
        "torch_dtype": "bfloat16",
        "max_batch_size": infer_batch_size,
    }
    if adapter_dir:
        kwargs["adapters"] = [adapter_dir]

    engine = TransformersEngine(model_dir, **kwargs)
    return engine, InferRequest, RequestConfig


def build_infer_request(InferRequest, sample):
    """Build an InferRequest from a SWIFT-format sample."""
    msgs = sample["messages"]
    images = sample.get("images", [])
    infer_msgs = [
        {"role": msgs[0]["role"], "content": msgs[0]["content"]},
        {"role": msgs[1]["role"], "content": msgs[1]["content"]},
    ]
    return InferRequest(messages=infer_msgs, images=images)


def load_existing_predictions(pred_path):
    """Load already-completed predictions from a previous run for resume."""
    existing = {}
    if os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                existing[rec["idx"]] = rec["pred_text"]
    return existing


def run_batch_inference(pt_engine, InferRequest, RequestConfig, samples,
                        batch_size=4, pred_path=None):
    """Run inference on all samples with batching. Returns list of response texts.

    If pred_path is given:
      - Loads existing predictions to skip already-completed samples (resume).
      - Appends new predictions to pred_path after each batch (crash-safe).
    """
    config = RequestConfig(
        max_tokens=8192,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    # Load existing predictions for resume
    existing = load_existing_predictions(pred_path) if pred_path else {}
    if existing:
        print(f"  Resuming: found {len(existing)} existing predictions, "
              f"skipping completed samples.", flush=True)

    all_responses = [""] * len(samples)
    n_skipped = 0

    # Fill in already-completed predictions
    for idx, text in existing.items():
        if idx < len(all_responses):
            all_responses[idx] = text
            n_skipped += 1

    # Open pred_path for appending new results
    pred_file = open(pred_path, "a", encoding="utf-8") if pred_path else None

    t0 = time.time()
    n_inferred = 0

    for start in range(0, len(samples), batch_size):
        batch_indices = list(range(start, min(start + batch_size, len(samples))))

        # Filter out already-completed samples
        todo_indices = [i for i in batch_indices if i not in existing]
        if not todo_indices:
            continue

        todo_samples = [samples[i] for i in todo_indices]
        reqs = []
        for s in todo_samples:
            try:
                reqs.append(build_infer_request(InferRequest, s))
            except Exception as e:
                print(f"  ERROR building request for sample {todo_indices[len(reqs)]}: {e}",
                      flush=True)
                reqs.append(None)

        # Filter out failed requests, track indices
        valid_local = [j for j, r in enumerate(reqs) if r is not None]
        valid_reqs = [reqs[j] for j in valid_local]

        if valid_reqs:
            try:
                resp_list = pt_engine.infer(valid_reqs, config)
            except Exception as e:
                print(f"  ERROR batch inference at {start}: {e}", flush=True)
                resp_list = [None] * len(valid_reqs)
        else:
            resp_list = []

        # Reassemble and save
        for k, local_j in enumerate(valid_local):
            global_idx = todo_indices[local_j]
            if resp_list and k < len(resp_list) and resp_list[k] is not None:
                content = resp_list[k].choices[0].message.content
                all_responses[global_idx] = content if content else ""
            # Write to file immediately (crash-safe)
            if pred_file:
                pred_file.write(json.dumps(
                    {"idx": global_idx, "pred_text": all_responses[global_idx]},
                    ensure_ascii=False) + "\n")
                pred_file.flush()

        n_inferred += len(todo_indices)
        done = n_skipped + n_inferred
        if done % 50 == 0 or done == len(samples):
            elapsed = time.time() - t0
            rate = n_inferred / elapsed if elapsed > 0 else 0
            remaining = len(samples) - n_skipped - n_inferred
            eta = remaining / rate if rate > 0 else 0
            print(f"  Inference: [{done}/{len(samples)}] "
                  f"(skipped={n_skipped}, new={n_inferred})  "
                  f"{rate:.2f} img/s  ETA {eta/60:.1f}min", flush=True)

    if pred_file:
        pred_file.close()

    return all_responses


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE ONE IMAGE (multi-table greedy matching)
# ═══════════════════════════════════════════════════════════════════════════════

def score_one_image(pred_text, gt_text, teds_scorer, grits_fn, penalty_mode="both"):
    """Score one image: extract tables, match, compute per-image penalized scores.

    penalty_mode: 'both' = min(M,N)/max(M,N), 'underdet_only' = N/M if N<M else 1.0

    Penalization logic identical to run_llm_on_generated.py:
      - Greedy match pred tables to GT tables by TEDS
      - avg_score = mean of matched pair scores
      - penalty = min(M, N) / max(M, N)  (M=GT, N=pred)
      - penalized_score = avg_score * penalty

    This penalizes BOTH missed GT (M > N) and extra pred (N > M).
    """
    pred_htmls = extract_tables_from_response(pred_text)
    gt_htmls = extract_tables_from_response(gt_text)

    M = len(gt_htmls)
    N = len(pred_htmls)

    # Edge cases: no GT or no pred
    if M == 0 and N == 0:
        return {
            "n_gt_tables": 0, "n_pred_tables": 0,
            "matched_scores": [],
            "penalty": 1.0,
            "penalized_teds": 1.0,
            "penalized_grits_con": 1.0,
            "penalized_grits_top": 1.0,
        }

    if M == 0 or N == 0:
        return {
            "n_gt_tables": M, "n_pred_tables": N,
            "matched_scores": [],
            "penalty": 0.0,
            "penalized_teds": 0.0,
            "penalized_grits_con": 0.0,
            "penalized_grits_top": 0.0,
        }

    # Compute TEDS matrix for greedy matching
    teds_matrix = {}
    for pi, ph in enumerate(pred_htmls):
        for gj, gh in enumerate(gt_htmls):
            pn = normalize_html(ph)
            gn = normalize_html(gh)
            try:
                teds_matrix[(pi, gj)] = teds_scorer.evaluate(
                    wrap_for_teds(pn), wrap_for_teds(gn))
            except Exception:
                teds_matrix[(pi, gj)] = 0.0

    matches = greedy_match_tables_by_teds(pred_htmls, gt_htmls, teds_matrix)

    # Compute full scores (TEDS + GriTS) for matched pairs
    matched_scores = []
    for pi, gj in matches:
        s = compute_scores(pred_htmls[pi], gt_htmls[gj], teds_scorer, grits_fn)
        matched_scores.append(s)

    # Average of matched pairs
    avg_teds = sum(s.get("teds", 0.0) for s in matched_scores) / len(matched_scores)
    avg_gc = sum(s.get("grits_con", 0.0) for s in matched_scores) / len(matched_scores)
    avg_gt = sum(s.get("grits_top", 0.0) for s in matched_scores) / len(matched_scores)

    # Penalty
    if penalty_mode == "underdet_only":
        penalty = N / M if N < M else 1.0  # only penalize under-detection
    else:
        penalty = min(M, N) / max(M, N)  # penalize both missed and extra

    return {
        "n_gt_tables": M,
        "n_pred_tables": N,
        "matched_scores": matched_scores,
        "penalty": round(penalty, 4),
        "penalized_teds": round(avg_teds * penalty, 6),
        "penalized_grits_con": round(avg_gc * penalty, 6),
        "penalized_grits_top": round(avg_gt * penalty, 6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen2.5-VL on test set")
    parser.add_argument("--model", required=True, help="Base model directory")
    parser.add_argument("--adapter", default=None, help="LoRA adapter checkpoint path (omit for base model)")
    parser.add_argument("--test_jsonl", required=True, help="Path to test.jsonl (SWIFT format)")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--teds_dir", default="/path/to/eval_deps",
                        help="Directory containing teds.py")
    parser.add_argument("--grits_dir", default="/path/to/eval_deps",
                        help="Directory containing grits.py")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (for quick test)")
    parser.add_argument("--infer_batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--score_workers", type=int, default=8,
                        help="Threads for parallel TEDS + GriTS scoring")
    parser.add_argument("--penalty_mode", choices=["both", "underdet_only"], default="both",
                        help="Penalty mode: 'both' penalizes missed+extra (default), "
                             "'underdet_only' only penalizes when N_pred < M_gt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load scorers (verify imports work) ──
    print(f"Loading TEDS from: {args.teds_dir}", flush=True)
    print(f"Loading GriTS from: {args.grits_dir}", flush=True)
    _ = load_teds(args.teds_dir)  # verify import works; threads create their own
    grits_fn = load_grits(args.grits_dir)
    print("Scorers loaded.", flush=True)

    # ── Load test data ──
    print(f"\nLoading test data from: {args.test_jsonl}", flush=True)
    samples = []
    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
    print(f"  Test samples: {len(samples)}", flush=True)

    # ── Load model ──
    print(f"\nLoading model: {args.model}", flush=True)
    print(f"Loading adapter: {args.adapter}", flush=True)
    pt_engine, InferRequest, RequestConfig = load_swift_engine(
        args.model, args.adapter, infer_batch_size=args.infer_batch_size)
    print("Model loaded.", flush=True)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Batch Inference (with resume support)
    # ══════════════════════════════════════════════════════════════════════
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    print(f"\nPhase 1: Batch inference on {len(samples)} samples "
          f"(batch_size={args.infer_batch_size})...", flush=True)
    t_infer_start = time.time()

    all_pred_texts = run_batch_inference(
        pt_engine, InferRequest, RequestConfig, samples,
        batch_size=args.infer_batch_size, pred_path=pred_path)

    t_infer = time.time() - t_infer_start
    print(f"Inference done: {t_infer:.0f}s  Predictions at: {pred_path}", flush=True)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Multi-threaded Scoring
    # ══════════════════════════════════════════════════════════════════════

    # Collect GT and pred tables for all images
    gt_texts = [s["messages"][2]["content"] for s in samples]

    # Step 2a: Extract tables from all predictions and GTs
    print(f"\nPhase 2: Scoring {len(samples)} samples "
          f"(workers={args.score_workers})...", flush=True)

    all_pred_htmls = [extract_tables_from_response(p) for p in all_pred_texts]
    all_gt_htmls = [extract_tables_from_response(g) for g in gt_texts]

    # Step 2b: Build TEDS tasks for all N×M pairs across all images
    # image_info[i] = (M, N, pred_htmls, gt_htmls)
    image_info = []
    teds_tasks = []  # (img_idx, pi, gj, pred_html, gt_html)
    for i in range(len(samples)):
        ph = all_pred_htmls[i]
        gh = all_gt_htmls[i]
        M, N = len(gh), len(ph)
        image_info.append((M, N))
        if M > 0 and N > 0:
            for pi in range(N):
                for gj in range(M):
                    teds_tasks.append((i, pi, gj, ph[pi], gh[gj]))

    print(f"  TEDS matching tasks: {len(teds_tasks)} pairs", flush=True)

    # Thread-local TEDS scorers (each thread gets its own)
    _thread_local = threading.local()

    def _get_thread_teds():
        if not hasattr(_thread_local, 'teds'):
            _thread_local.teds = load_teds(args.teds_dir)
        return _thread_local.teds

    # Step 2c: Compute all TEDS scores in parallel
    teds_all = {}  # (img_idx, pi, gj) -> score

    def _teds_one(task):
        img_idx, pi, gj, pred_html, gt_html = task
        if not pred_html or not gt_html:
            return (img_idx, pi, gj, 0.0)
        thread_teds = _get_thread_teds()
        pn = normalize_html(pred_html)
        gn = normalize_html(gt_html)
        try:
            score = thread_teds.evaluate(wrap_for_teds(pn), wrap_for_teds(gn))
        except Exception:
            score = 0.0
        return (img_idx, pi, gj, score)

    t_score_start = time.time()

    with ThreadPoolExecutor(max_workers=args.score_workers) as pool:
        futures = {pool.submit(_teds_one, task): task for task in teds_tasks}
        done_count = 0
        for future in as_completed(futures):
            img_idx, pi, gj, score = future.result()
            teds_all[(img_idx, pi, gj)] = score
            done_count += 1
            if done_count % 100 == 0 or done_count == len(teds_tasks):
                elapsed = time.time() - t_score_start
                rate = done_count / elapsed if elapsed > 0 else 0
                print(f"  TEDS: {done_count}/{len(teds_tasks)} pairs "
                      f"({rate:.1f} pairs/s)", flush=True)

    # Step 2d: Greedy matching + compute GriTS for matched pairs
    grits_tasks = []  # (img_idx, pi, gj, pred_html, gt_html)
    per_image_matches = {}  # img_idx -> [(pi, gj), ...]

    for i in range(len(samples)):
        M, N = image_info[i]
        if M == 0 or N == 0:
            per_image_matches[i] = []
            continue
        teds_matrix = {}
        for pi in range(N):
            for gj in range(M):
                teds_matrix[(pi, gj)] = teds_all.get((i, pi, gj), 0.0)
        matches = greedy_match_tables_by_teds(all_pred_htmls[i], all_gt_htmls[i], teds_matrix)
        per_image_matches[i] = matches
        for pi, gj in matches:
            grits_tasks.append((i, pi, gj, all_pred_htmls[i][pi], all_gt_htmls[i][gj]))

    print(f"  GriTS tasks: {len(grits_tasks)} matched pairs", flush=True)

    # Compute GriTS in parallel
    grits_all = {}  # (img_idx, pi, gj) -> {grits_con, grits_top}

    def _grits_one(task):
        img_idx, pi, gj, pred_html, gt_html = task
        pn = normalize_html(pred_html)
        gn = normalize_html(gt_html)
        try:
            g = grits_fn(gn, pn)
            return (img_idx, pi, gj, {"grits_con": g["grits_con"], "grits_top": g["grits_top"]})
        except Exception:
            return (img_idx, pi, gj, {"grits_con": 0.0, "grits_top": 0.0})

    with ThreadPoolExecutor(max_workers=args.score_workers) as pool:
        futures = {pool.submit(_grits_one, task): task for task in grits_tasks}
        done_count = 0
        for future in as_completed(futures):
            img_idx, pi, gj, scores = future.result()
            grits_all[(img_idx, pi, gj)] = scores
            done_count += 1
            if done_count % 100 == 0 or done_count == len(grits_tasks):
                elapsed = time.time() - t_score_start
                print(f"  GriTS: {done_count}/{len(grits_tasks)} pairs", flush=True)

    t_score = time.time() - t_score_start
    print(f"Scoring done: {t_score:.0f}s", flush=True)

    # Step 2e: Assemble per-image penalized scores
    all_image_scores = []
    for i in range(len(samples)):
        M, N = image_info[i]
        if M == 0 and N == 0:
            all_image_scores.append({
                "n_gt_tables": 0, "n_pred_tables": 0, "penalty": 1.0,
                "penalized_teds": 1.0, "penalized_grits_con": 1.0, "penalized_grits_top": 1.0,
            })
            continue
        if M == 0 or N == 0:
            all_image_scores.append({
                "n_gt_tables": M, "n_pred_tables": N, "penalty": 0.0,
                "penalized_teds": 0.0, "penalized_grits_con": 0.0, "penalized_grits_top": 0.0,
            })
            continue

        matches = per_image_matches[i]
        pair_teds = []
        pair_gc = []
        pair_gt = []
        for pi, gj in matches:
            pair_teds.append(teds_all.get((i, pi, gj), 0.0))
            g = grits_all.get((i, pi, gj), {"grits_con": 0.0, "grits_top": 0.0})
            pair_gc.append(g["grits_con"])
            pair_gt.append(g["grits_top"])

        avg_teds = sum(pair_teds) / len(pair_teds)
        avg_gc = sum(pair_gc) / len(pair_gc)
        avg_gt = sum(pair_gt) / len(pair_gt)
        if args.penalty_mode == "underdet_only":
            penalty = N / M if N < M else 1.0
        else:
            penalty = min(M, N) / max(M, N)

        all_image_scores.append({
            "n_gt_tables": M, "n_pred_tables": N,
            "penalty": round(penalty, 4),
            "penalized_teds": round(avg_teds * penalty, 6),
            "penalized_grits_con": round(avg_gc * penalty, 6),
            "penalized_grits_top": round(avg_gt * penalty, 6),
        })

    total_time = time.time() - t_infer_start

    # ── Aggregate (per-image penalized scores, identical to run_llm_on_generated.py) ──
    image_teds = [s["penalized_teds"] for s in all_image_scores]
    image_grits_con = [s["penalized_grits_con"] for s in all_image_scores]
    image_grits_top = [s["penalized_grits_top"] for s in all_image_scores]

    total_gt = sum(s["n_gt_tables"] for s in all_image_scores)
    total_pred = sum(s["n_pred_tables"] for s in all_image_scores)
    n_exact_det = sum(1 for s in all_image_scores if s["n_gt_tables"] == s["n_pred_tables"])
    n_no_det = sum(1 for s in all_image_scores if s["n_pred_tables"] == 0 and s["n_gt_tables"] > 0)

    # By GT table count
    by_n_tables = defaultdict(lambda: {"teds": [], "grits_con": [], "grits_top": []})
    for s in all_image_scores:
        n_gt = s["n_gt_tables"]
        by_n_tables[n_gt]["teds"].append(s["penalized_teds"])
        by_n_tables[n_gt]["grits_con"].append(s["penalized_grits_con"])
        by_n_tables[n_gt]["grits_top"].append(s["penalized_grits_top"])

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # ── Print summary ──
    print("\n" + "=" * 70, flush=True)
    print("EVALUATION RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"  Model:              {args.model}", flush=True)
    print(f"  Adapter:            {args.adapter}", flush=True)
    print(f"  Test samples:       {len(samples)}", flush=True)
    print(f"  GT tables:          {total_gt}", flush=True)
    print(f"  Pred tables:        {total_pred}", flush=True)
    print(f"  No detection:       {n_no_det}", flush=True)
    print(f"  Detection exact:    {n_exact_det}/{len(samples)} ({100*n_exact_det/len(samples):.1f}%)", flush=True)
    print(f"  Inference time:     {t_infer:.0f}s ({t_infer/60:.1f}min)", flush=True)
    print(f"  Scoring time:       {t_score:.0f}s ({t_score/60:.1f}min)", flush=True)
    print(f"  Wall clock:         {total_time:.0f}s ({total_time/60:.1f}min)", flush=True)

    print(f"\n  --- Per-image penalized averages (N={len(samples)}) ---", flush=True)
    print(f"  avg TEDS (penalized):      {mean(image_teds):.4f}", flush=True)
    print(f"  avg GriTS_Con (penalized): {mean(image_grits_con):.4f}", flush=True)
    print(f"  avg GriTS_Top (penalized): {mean(image_grits_top):.4f}", flush=True)

    print(f"\n  --- By GT table count (penalized) ---", flush=True)
    for n_gt in sorted(by_n_tables.keys()):
        b = by_n_tables[n_gt]
        print(f"  {n_gt}-table pages (n={len(b['teds'])}): "
              f"TEDS={mean(b['teds']):.4f}  "
              f"GriTS_Con={mean(b['grits_con']):.4f}  "
              f"GriTS_Top={mean(b['grits_top']):.4f}", flush=True)

    print("=" * 70, flush=True)

    # ── Save ──
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "n_samples": len(samples),
        "n_gt_tables": total_gt,
        "n_pred_tables": total_pred,
        "n_no_detection": n_no_det,
        "detection_exact_match": f"{n_exact_det}/{len(samples)}",
        "avg_teds_penalized": round(mean(image_teds), 6),
        "avg_grits_con_penalized": round(mean(image_grits_con), 6),
        "avg_grits_top_penalized": round(mean(image_grits_top), 6),
        "inference_s": round(t_infer, 1),
        "scoring_s": round(t_score, 1),
        "wall_clock_s": round(total_time, 1),
        "by_n_tables": {
            str(k): {
                "avg_teds": round(mean(v["teds"]), 6),
                "avg_grits_con": round(mean(v["grits_con"]), 6),
                "avg_grits_top": round(mean(v["grits_top"]), 6),
                "n_images": len(v["teds"]),
            }
            for k, v in sorted(by_n_tables.items())
        },
    }

    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}", flush=True)

    # Save per-image detail
    detail_path = os.path.join(args.output_dir, "eval_detail.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for i, img_scores in enumerate(all_image_scores):
            f.write(json.dumps({"idx": i, **img_scores}, ensure_ascii=False) + "\n")
    print(f"Detail saved to: {detail_path}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
