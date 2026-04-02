"""Evaluate fine-tuned Qwen2.5-VL on PubTables test set — 7-GPU parallel version.

Two-step workflow:
  Step 1: Launch 7 inference shards in parallel (one per GPU)
  Step 2: Merge all shard predictions + run TEDS/GriTS scoring (CPU only)

Usage (7x GPU):

    # Step 1: Run all 7 shards in parallel
    for i in 0 1 2 3 4 5 6; do
        CUDA_VISIBLE_DEVICES=$i python eval_finetuned_7gpu.py infer \
            --shard_id $i --n_shards 7 \
            --model /path/to/... \
            --adapter /path/to/... \
            --test_jsonl /path/to/... \
            --output_dir /path/to/... \
            --infer_batch_size 4 &
    done
    wait
    echo "All inference shards done"

    # Step 2: Merge + score (single process, CPU only)
    python eval_finetuned_7gpu.py score \
        --n_shards 7 \
        --test_jsonl /path/to/... \
        --output_dir /path/to/... \
        --teds_dir /path/to/... \
        --grits_dir /path/to/... \
        --score_workers 16
"""

import argparse
import html as html_mod
import json
import os
import re
import sys
import time
import threading
from typing import List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ═══════════════════════════════════════════════════════════════════════════════
# HTML EXTRACTION FROM MODEL RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

def extract_tables_from_response(text: str) -> List[str]:
    """Extract ALL <table>...</table> fragments from model response."""
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
# HTML NORMALIZATION & SCORING
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


def greedy_match_tables_by_teds(pred_htmls, gt_htmls, teds_matrix):
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
    if teds_dir not in sys.path:
        sys.path.insert(0, teds_dir)
    from teds import TEDS
    return TEDS(n_jobs=1, ignore_nodes="", structure_only=False)


def load_grits(grits_dir):
    if grits_dir not in sys.path:
        sys.path.insert(0, grits_dir)
    from grits import grits_from_html
    return grits_from_html


# ═══════════════════════════════════════════════════════════════════════════════
# SWIFT INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def load_swift_engine(model_dir, adapter_dir=None, infer_batch_size=4):
    os.environ.setdefault("MAX_PIXELS", "1003520")
    from swift import TransformersEngine, InferRequest, RequestConfig
    kwargs = {"torch_dtype": "bfloat16", "max_batch_size": infer_batch_size}
    if adapter_dir:
        kwargs["adapters"] = [adapter_dir]
    engine = TransformersEngine(model_dir, **kwargs)
    return engine, InferRequest, RequestConfig


def build_infer_request(InferRequest, sample):
    msgs = sample["messages"]
    images = sample.get("images", [])
    infer_msgs = [
        {"role": msgs[0]["role"], "content": msgs[0]["content"]},
        {"role": msgs[1]["role"], "content": msgs[1]["content"]},
    ]
    return InferRequest(messages=infer_msgs, images=images)


def run_shard_inference(engine, InferRequest, RequestConfig, samples,
                        batch_size=4, start_idx=0, pred_path=None):
    """Run inference on a list of samples. Write results with global indices."""
    config = RequestConfig(
        max_tokens=8192, temperature=0.0, top_p=1.0, repetition_penalty=1.0,
    )

    # Resume support: load existing predictions from this shard file
    existing = {}
    if pred_path and os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                existing[rec["idx"]] = rec["pred_text"]
        if existing:
            print(f"  Resume: {len(existing)} existing predictions found", flush=True)

    all_responses = [""] * len(samples)
    for idx, text in existing.items():
        local_i = idx - start_idx
        if 0 <= local_i < len(all_responses):
            all_responses[local_i] = text

    pred_file = open(pred_path, "a", encoding="utf-8") if pred_path else None
    t0 = time.time()
    n_inferred = 0
    n_skipped = len(existing)

    for start in range(0, len(samples), batch_size):
        batch_indices = list(range(start, min(start + batch_size, len(samples))))

        # Skip already-completed
        todo_indices = [i for i in batch_indices if (start_idx + i) not in existing]
        if not todo_indices:
            continue

        todo_samples = [samples[i] for i in todo_indices]
        reqs = []
        for s in todo_samples:
            try:
                reqs.append(build_infer_request(InferRequest, s))
            except Exception as e:
                print(f"  ERROR building request: {e}", flush=True)
                reqs.append(None)

        valid_local = [j for j, r in enumerate(reqs) if r is not None]
        valid_reqs = [reqs[j] for j in valid_local]

        if valid_reqs:
            try:
                resp_list = engine.infer(valid_reqs, config)
            except Exception as e:
                print(f"  ERROR batch inference: {e}", flush=True)
                resp_list = [None] * len(valid_reqs)
        else:
            resp_list = []

        for k, local_j in enumerate(valid_local):
            sample_i = todo_indices[local_j]
            global_idx = start_idx + sample_i
            if resp_list and k < len(resp_list) and resp_list[k] is not None:
                content = resp_list[k].choices[0].message.content
                all_responses[sample_i] = content if content else ""
            if pred_file:
                pred_file.write(json.dumps(
                    {"idx": global_idx, "pred_text": all_responses[sample_i]},
                    ensure_ascii=False) + "\n")
                pred_file.flush()

        n_inferred += len(todo_indices)
        done = n_skipped + n_inferred
        if done % 20 == 0 or done == len(samples):
            elapsed = time.time() - t0
            rate = n_inferred / elapsed if elapsed > 0 else 0
            remaining = len(samples) - n_skipped - n_inferred
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{done}/{len(samples)}] "
                  f"(skipped={n_skipped}, new={n_inferred})  "
                  f"{rate:.2f} img/s  ETA {eta/60:.1f}min", flush=True)

    if pred_file:
        pred_file.close()
    return all_responses


# ═══════════════════════════════════════════════════════════════════════════════
# SUBCOMMAND: infer
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_infer(args, samples):
    """Run inference on one shard."""
    total = len(samples)
    shard_size = (total + args.n_shards - 1) // args.n_shards
    start = args.shard_id * shard_size
    end = min(start + shard_size, total)
    shard_samples = samples[start:end]

    print(f"\n{'='*60}", flush=True)
    print(f"[Shard {args.shard_id}/{args.n_shards}] "
          f"Samples [{start}, {end}) — {len(shard_samples)} images", flush=True)
    print(f"{'='*60}", flush=True)

    # Load model
    print(f"Loading model: {args.model}", flush=True)
    if args.adapter:
        print(f"Loading adapter: {args.adapter}", flush=True)
    pt_engine, InferRequest, RequestConfig = load_swift_engine(
        args.model, args.adapter, infer_batch_size=args.infer_batch_size)
    print("Model loaded.\n", flush=True)

    # Run inference
    shard_pred_path = os.path.join(args.output_dir, f"predictions_shard{args.shard_id}.jsonl")
    t0 = time.time()

    run_shard_inference(
        pt_engine, InferRequest, RequestConfig, shard_samples,
        batch_size=args.infer_batch_size, start_idx=start, pred_path=shard_pred_path)

    elapsed = time.time() - t0
    print(f"\n[Shard {args.shard_id}] Done: {len(shard_samples)} samples "
          f"in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Saved to: {shard_pred_path}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SUBCOMMAND: score
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_score(args, samples):
    """Merge shard predictions and run TEDS + GriTS scoring."""

    # ── Merge predictions ──
    print(f"\nMerging predictions from {args.n_shards} shards...", flush=True)
    all_preds = {}

    for sid in range(args.n_shards):
        shard_path = os.path.join(args.output_dir, f"predictions_shard{sid}.jsonl")
        if not os.path.exists(shard_path):
            print(f"  WARNING: {shard_path} not found!", flush=True)
            continue
        count = 0
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                all_preds[rec["idx"]] = rec["pred_text"]
                count += 1
        print(f"  Shard {sid}: {count} predictions", flush=True)

    # Also check for old-format predictions.jsonl (from previous runs)
    old_pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    if os.path.exists(old_pred_path):
        old_count = 0
        with open(old_pred_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["idx"] not in all_preds:
                    all_preds[rec["idx"]] = rec["pred_text"]
                    old_count += 1
        if old_count:
            print(f"  Old predictions.jsonl: added {old_count} extra", flush=True)

    total = len(samples)
    print(f"\n  Total merged: {len(all_preds)}/{total} predictions", flush=True)

    missing = [i for i in range(total) if i not in all_preds]
    if missing:
        print(f"  WARNING: {len(missing)} samples have no prediction!", flush=True)
        if len(missing) <= 20:
            print(f"  Missing indices: {missing}", flush=True)

    # Build full prediction list
    all_pred_texts = [all_preds.get(i, "") for i in range(total)]
    gt_texts = [s["messages"][2]["content"] for s in samples]

    # Save merged
    merged_path = os.path.join(args.output_dir, "predictions_merged.jsonl")
    with open(merged_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(all_pred_texts):
            f.write(json.dumps({"idx": i, "pred_text": text}, ensure_ascii=False) + "\n")
    print(f"  Merged saved to: {merged_path}", flush=True)

    # ── Load scorers ──
    print(f"\nLoading scorers...", flush=True)
    _ = load_teds(args.teds_dir)
    grits_fn = load_grits(args.grits_dir)
    print("Scorers ready.", flush=True)

    # ── Scoring ──
    all_pred_htmls = [extract_tables_from_response(p) for p in all_pred_texts]
    all_gt_htmls = [extract_tables_from_response(g) for g in gt_texts]

    # Build TEDS tasks
    image_info = []
    teds_tasks = []
    for i in range(total):
        ph = all_pred_htmls[i]
        gh = all_gt_htmls[i]
        M, N = len(gh), len(ph)
        image_info.append((M, N))
        if M > 0 and N > 0:
            for pi in range(N):
                for gj in range(M):
                    teds_tasks.append((i, pi, gj, ph[pi], gh[gj]))

    print(f"\nScoring {len(teds_tasks)} TEDS pairs with {args.score_workers} threads...",
          flush=True)

    _thread_local = threading.local()

    def _get_thread_teds():
        if not hasattr(_thread_local, 'teds'):
            _thread_local.teds = load_teds(args.teds_dir)
        return _thread_local.teds

    teds_all = {}

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
            if done_count % 200 == 0 or done_count == len(teds_tasks):
                elapsed = time.time() - t_score_start
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (len(teds_tasks) - done_count) / rate if rate > 0 else 0
                print(f"  TEDS: {done_count}/{len(teds_tasks)} "
                      f"({rate:.1f} pairs/s, ETA {eta/60:.1f}min)", flush=True)

    # Greedy matching + GriTS
    grits_tasks = []
    per_image_matches = {}

    for i in range(total):
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

    print(f"  GriTS: {len(grits_tasks)} matched pairs...", flush=True)

    grits_all = {}

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
            if done_count % 200 == 0 or done_count == len(grits_tasks):
                print(f"  GriTS: {done_count}/{len(grits_tasks)} pairs", flush=True)

    t_score = time.time() - t_score_start
    print(f"Scoring done: {t_score:.0f}s ({t_score/60:.1f}min)", flush=True)

    # ── Assemble per-image penalized scores ──
    all_image_scores = []
    for i in range(total):
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
        pair_teds, pair_gc, pair_gt = [], [], []
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

    # ── Aggregate & print ──
    image_teds = [s["penalized_teds"] for s in all_image_scores]
    image_grits_con = [s["penalized_grits_con"] for s in all_image_scores]
    image_grits_top = [s["penalized_grits_top"] for s in all_image_scores]

    total_gt = sum(s["n_gt_tables"] for s in all_image_scores)
    total_pred = sum(s["n_pred_tables"] for s in all_image_scores)
    n_exact_det = sum(1 for s in all_image_scores if s["n_gt_tables"] == s["n_pred_tables"])
    n_no_det = sum(1 for s in all_image_scores if s["n_pred_tables"] == 0 and s["n_gt_tables"] > 0)

    by_n_tables = defaultdict(lambda: {"teds": [], "grits_con": [], "grits_top": []})
    for s in all_image_scores:
        n_gt = s["n_gt_tables"]
        by_n_tables[n_gt]["teds"].append(s["penalized_teds"])
        by_n_tables[n_gt]["grits_con"].append(s["penalized_grits_con"])
        by_n_tables[n_gt]["grits_top"].append(s["penalized_grits_top"])

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print("\n" + "=" * 70, flush=True)
    print("EVALUATION RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"  Model:              {args.model}", flush=True)
    print(f"  Adapter:            {args.adapter}", flush=True)
    print(f"  Test samples:       {total}", flush=True)
    print(f"  GT tables:          {total_gt}", flush=True)
    print(f"  Pred tables:        {total_pred}", flush=True)
    print(f"  No detection:       {n_no_det}", flush=True)
    print(f"  Detection exact:    {n_exact_det}/{total} ({100*n_exact_det/total:.1f}%)", flush=True)
    print(f"  Scoring time:       {t_score:.0f}s ({t_score/60:.1f}min)", flush=True)

    print(f"\n  --- Per-image penalized averages (N={total}) ---", flush=True)
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
        "n_samples": total,
        "n_gt_tables": total_gt,
        "n_pred_tables": total_pred,
        "n_no_detection": n_no_det,
        "detection_exact_match": f"{n_exact_det}/{total}",
        "avg_teds_penalized": round(mean(image_teds), 6),
        "avg_grits_con_penalized": round(mean(image_grits_con), 6),
        "avg_grits_top_penalized": round(mean(image_grits_top), 6),
        "scoring_s": round(t_score, 1),
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

    detail_path = os.path.join(args.output_dir, "eval_detail.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for i, img_scores in enumerate(all_image_scores):
            f.write(json.dumps({"idx": i, **img_scores}, ensure_ascii=False) + "\n")
    print(f"Detail saved to: {detail_path}", flush=True)
    print("\nDone.", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="7-GPU parallel eval for fine-tuned Qwen2.5-VL")
    sub = parser.add_subparsers(dest="command", help="Sub-command")

    # ── infer ──
    p_infer = sub.add_parser("infer", help="Run inference on one shard (one GPU)")
    p_infer.add_argument("--shard_id", type=int, required=True, help="Shard index (0-based)")
    p_infer.add_argument("--n_shards", type=int, required=True, help="Total shards")
    p_infer.add_argument("--model", required=True, help="Base model directory")
    p_infer.add_argument("--adapter", default=None, help="LoRA adapter checkpoint")
    p_infer.add_argument("--test_jsonl", required=True, help="Path to test.jsonl")
    p_infer.add_argument("--output_dir", required=True, help="Output directory")
    p_infer.add_argument("--infer_batch_size", type=int, default=4)

    # ── score ──
    p_score = sub.add_parser("score", help="Merge shards + run TEDS/GriTS scoring (CPU only)")
    p_score.add_argument("--n_shards", type=int, required=True, help="Total shards to merge")
    p_score.add_argument("--model", default=None, help="Model path (for summary only)")
    p_score.add_argument("--adapter", default=None, help="Adapter path (for summary only)")
    p_score.add_argument("--test_jsonl", required=True, help="Path to test.jsonl")
    p_score.add_argument("--output_dir", required=True, help="Output directory")
    p_score.add_argument("--teds_dir", default="/path/to/eval_deps")
    p_score.add_argument("--grits_dir", default="/path/to/eval_deps")
    p_score.add_argument("--score_workers", type=int, default=16)
    p_score.add_argument("--penalty_mode", choices=["both", "underdet_only"], default="both")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load test data
    print(f"Loading test data: {args.test_jsonl}", flush=True)
    samples = []
    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"  Total: {len(samples)} samples", flush=True)

    if args.command == "infer":
        cmd_infer(args, samples)
    elif args.command == "score":
        cmd_score(args, samples)


if __name__ == "__main__":
    main()
