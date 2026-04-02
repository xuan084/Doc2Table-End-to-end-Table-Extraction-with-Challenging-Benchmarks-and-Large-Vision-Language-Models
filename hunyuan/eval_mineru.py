"""Evaluate MinerU (Pipeline: DocLayout-YOLO + StructTable-InternVL2-1B) on table extraction.

MinerU is a detect-then-recognize pipeline:
  1. DocLayout-YOLO detects tables on the full page
  2. StructTable-InternVL2-1B recognizes table structure from cropped images
  3. Outputs HTML tables

Two-phase design (same as eval_hunyuan.py):
  Phase 1: Inference — process images through MinerU, save predictions
  Phase 2: Scoring — same scoring engine as eval_hunyuan.py

Usage:
    # Phase 1: Inference (supports sharding for multi-GPU)
    CUDA_VISIBLE_DEVICES=0 python eval_mineru.py infer \
        --dataset generated \
        --test_jsonl /path/to/test.jsonl \
        --image_base_dir /path/to/generated \
        --output_dir /path/to/results \
        --shard_id 0 --n_shards 6

    # Phase 2: Score (CPU, same as eval_hunyuan.py)
    python eval_mineru.py score \
        --dataset generated \
        --test_jsonl /path/to/test.jsonl \
        --output_dir /path/to/results \
        --teds_dir /path/to/eval_deps \
        --grits_dir /path/to/eval_deps \
        --n_shards 6 \
        --score_workers 64 \
        --manifest /path/to/manifest.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Import scoring functions from eval_hunyuan.py (same scoring engine)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_hunyuan import (
    get_image_path, get_gt_text,
    extract_tables_from_response, normalize_html, count_cells,
    run_scoring,  # reuse the entire scoring pipeline
)


def extract_html_tables_from_mineru_output(output_dir: str, doc_name: str) -> str:
    """Extract HTML tables from MinerU's output files.

    MinerU output structure:
        output_dir/{doc_name}/auto/{doc_name}_content_list.json
        output_dir/{doc_name}/auto/{doc_name}.md

    In content_list.json, tables appear as:
        {"type": "table", "table_body": "<table>...</table>", ...}

    In .md files, tables are embedded as raw HTML:
        \n<table>...</table>\n

    Args:
        output_dir: MinerU output directory
        doc_name: document name (without extension)

    Returns:
        Raw text containing all <table>...</table> blocks
    """
    # Priority 1: content_list.json — most structured, has table_body field
    json_candidates = [
        os.path.join(output_dir, doc_name, "auto", f"{doc_name}_content_list.json"),
        os.path.join(output_dir, doc_name, f"{doc_name}_content_list.json"),
        os.path.join(output_dir, f"{doc_name}_content_list.json"),
    ]

    for json_path in json_candidates:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                content_list = json.load(f)
            tables = []
            for item in content_list:
                if isinstance(item, dict):
                    # Check type field (may be string "table" or enum)
                    item_type = str(item.get("type", ""))
                    if "table" in item_type.lower():
                        # table_body holds the HTML (from source code: span['html'])
                        html = item.get("table_body", "")
                        if not html:
                            html = item.get("html", "")
                        if html and "<table" in html.lower():
                            tables.append(html)
            if tables:
                return "\n".join(tables)

    # Priority 2: markdown file — tables embedded as raw HTML
    md_candidates = [
        os.path.join(output_dir, doc_name, "auto", f"{doc_name}.md"),
        os.path.join(output_dir, doc_name, f"{doc_name}.md"),
        os.path.join(output_dir, f"{doc_name}.md"),
    ]

    for md_path in md_candidates:
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            if "<table" in content.lower():
                return content

    # Priority 3: brute-force search all files under output_dir
    for root, dirs, files in os.walk(output_dir):
        for fname in files:
            if fname.endswith(("_content_list.json",)):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        cl = json.load(f)
                    tables = []
                    for item in cl:
                        if isinstance(item, dict) and "table" in str(item.get("type", "")).lower():
                            html = item.get("table_body", item.get("html", ""))
                            if html and "<table" in html.lower():
                                tables.append(html)
                    if tables:
                        return "\n".join(tables)
                except Exception:
                    pass

    return ""


def run_mineru_inference(args, samples):
    """Run MinerU inference for one shard."""

    total = len(samples)
    shard_size = (total + args.n_shards - 1) // args.n_shards
    start = args.shard_id * shard_size
    end = min(start + shard_size, total)
    shard_samples = samples[start:end]

    print(f"Shard {args.shard_id}: [{start}, {end}) = {len(shard_samples)} samples", flush=True)

    pred_path = os.path.join(args.output_dir, f"predictions_shard{args.shard_id}.jsonl")

    # Resume
    existing = {}
    if os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                existing[rec["idx"]] = rec["pred_text"]
        print(f"  Resumed: {len(existing)} existing predictions", flush=True)

    # Import MinerU
    try:
        from mineru import parse_doc
    except ImportError:
        print("ERROR: MinerU not installed. Run: pip install -U 'mineru[all]'", flush=True)
        return

    t_start = time.time()
    new_count = 0
    mineru_tmp = os.path.join(args.output_dir, f"mineru_tmp_shard{args.shard_id}")
    os.makedirs(mineru_tmp, exist_ok=True)

    with open(pred_path, "a", encoding="utf-8") as fout:
        for local_i, sample in enumerate(shard_samples):
            global_idx = start + local_i
            if global_idx in existing:
                continue

            img_path = get_image_path(sample, getattr(args, "image_base_dir", None))
            if not img_path or not os.path.exists(img_path):
                fout.write(json.dumps({"idx": global_idx, "pred_text": ""}) + "\n")
                new_count += 1
                continue

            try:
                # Create per-image output dir
                img_out = os.path.join(mineru_tmp, f"img_{global_idx}")
                os.makedirs(img_out, exist_ok=True)

                # Run MinerU on single image
                parse_doc(
                    [Path(img_path)],
                    img_out,
                    backend="pipeline",
                    method="auto",
                )

                # Extract tables from output
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                raw_output = extract_html_tables_from_mineru_output(img_out, img_name)

                # If no tables found in expected locations, search all files
                if not raw_output or "<table" not in raw_output.lower():
                    for root, dirs, files in os.walk(img_out):
                        for fname in files:
                            if fname.endswith((".md", ".html", ".json")):
                                fpath = os.path.join(root, fname)
                                with open(fpath, "r", encoding="utf-8") as rf:
                                    content = rf.read()
                                if "<table" in content.lower():
                                    raw_output = content
                                    break
                        if raw_output and "<table" in raw_output.lower():
                            break

                pred_text = raw_output if raw_output else ""

            except Exception as e:
                print(f"  ERROR idx={global_idx}: {e}", flush=True)
                pred_text = ""

            fout.write(json.dumps({"idx": global_idx, "pred_text": pred_text}, ensure_ascii=False) + "\n")
            fout.flush()
            new_count += 1

            done = local_i + 1
            if done % 20 == 0 or done == len(shard_samples):
                skipped = done - new_count
                elapsed = time.time() - t_start
                rate = new_count / elapsed if elapsed > 0 else 0
                remaining = len(shard_samples) - done
                eta = remaining / (done / elapsed) if elapsed > 0 and done > 0 else 0
                print(f"  [{done}/{len(shard_samples)}] "
                      f"(skipped={skipped}, new={new_count}) "
                      f"{rate:.2f} img/s | "
                      f"Elapsed: {elapsed/60:.1f}min | "
                      f"ETA: {eta/60:.1f}min", flush=True)

    print(f"Shard {args.shard_id} done.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MinerU on table extraction")
    sub = parser.add_subparsers(dest="cmd")

    # Infer subcommand
    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--dataset", required=True, choices=["generated", "fintabnet", "pubtables"])
    p_infer.add_argument("--test_jsonl", required=True)
    p_infer.add_argument("--output_dir", required=True)
    p_infer.add_argument("--image_base_dir", default=None)
    p_infer.add_argument("--shard_id", type=int, default=0)
    p_infer.add_argument("--n_shards", type=int, default=1)

    # Score subcommand — reuse eval_hunyuan.py's scoring entirely
    p_score = sub.add_parser("score")
    p_score.add_argument("--dataset", required=True, choices=["generated", "fintabnet", "pubtables"])
    p_score.add_argument("--test_jsonl", required=True)
    p_score.add_argument("--output_dir", required=True)
    p_score.add_argument("--teds_dir", required=True)
    p_score.add_argument("--grits_dir", required=True)
    p_score.add_argument("--n_shards", type=int, default=1)
    p_score.add_argument("--score_workers", type=int, default=16)
    p_score.add_argument("--manifest", default=None)

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
        run_mineru_inference(args, samples)
    elif args.cmd == "score":
        # Reuse eval_hunyuan.py's scoring pipeline directly
        run_scoring(args, samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
