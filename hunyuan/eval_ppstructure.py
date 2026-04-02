"""Evaluate PP-StructureV3 (Pipeline: Baidu PaddleOCR) on table extraction.

PP-StructureV3 is a detect-then-recognize pipeline (Baidu, 44k star):
  1. PP-DocLayout detects tables on the full page
  2. PP-LCNet classifies tables (wired vs wireless)
  3. SLANeXt/SLANet-plus recognizes table structure
  4. PP-OCRv5 fills cell text
  5. Outputs HTML tables in pred_html field

Input: image file path (PNG/JPG) — no OCR needed (built-in PP-OCRv5)
Output: result[0]['table_res_list'][i]['pred_html'] = '<html><body><table>...</table></body></html>'

Two-phase design (same as eval_hunyuan.py):
  Phase 1: Inference — process images through PP-StructureV3, save predictions
  Phase 2: Scoring — same scoring engine as eval_hunyuan.py

Usage:
    # Phase 1: Inference (supports sharding for multi-GPU)
    CUDA_VISIBLE_DEVICES=0 PADDLE_PDX_MODEL_SOURCE=aistudio PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    python eval_ppstructure.py infer \
        --dataset generated \
        --test_jsonl /path/to/test.jsonl \
        --image_base_dir /path/to/generated \
        --output_dir /path/to/results \
        --shard_id 0 --n_shards 5

    # Phase 2: Score (CPU, same as eval_hunyuan.py)
    python eval_ppstructure.py score \
        --dataset generated \
        --test_jsonl /path/to/test.jsonl \
        --output_dir /path/to/results \
        --teds_dir /path/to/eval_deps \
        --grits_dir /path/to/eval_deps \
        --n_shards 5 \
        --score_workers 64 \
        --manifest /path/to/manifest.jsonl
"""

import argparse
import json
import os
import re
import sys
import time

# Import scoring functions from eval_hunyuan.py (same scoring engine)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_hunyuan import (
    get_image_path, get_gt_text,
    extract_tables_from_response, normalize_html, count_cells,
    run_scoring,  # reuse the entire scoring pipeline
)


def run_ppstructure_inference(args, samples):
    """Run PP-StructureV3 inference for one shard."""

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

    # Import PP-StructureV3
    try:
        from paddleocr import PPStructureV3
    except ImportError:
        print("ERROR: PaddleOCR not installed. Run: pip install paddleocr", flush=True)
        return

    # Initialize engine once (loads all models)
    print("  Loading PP-StructureV3 models...", flush=True)
    engine = PPStructureV3()
    print("  PP-StructureV3 ready.", flush=True)

    t_start = time.time()
    new_count = 0

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
                # Run PP-StructureV3 on single image
                result = engine.predict(img_path)

                # Extract HTML tables from result
                all_tables_html = []
                for r in result:
                    table_res_list = r.get("table_res_list", [])
                    for table_res in table_res_list:
                        if isinstance(table_res, dict):
                            pred_html = table_res.get("pred_html", "")
                        else:
                            pred_html = getattr(table_res, "pred_html", "")

                        if pred_html:
                            # pred_html is '<html><body><table>...</table></body></html>'
                            # Extract just the <table>...</table> part
                            table_matches = re.findall(
                                r"(<table[^>]*>.*?</table>)",
                                pred_html,
                                re.DOTALL | re.IGNORECASE,
                            )
                            all_tables_html.extend(table_matches)

                if all_tables_html:
                    pred_text = "\n".join(all_tables_html)
                else:
                    pred_text = ""

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
    parser = argparse.ArgumentParser(description="Evaluate PP-StructureV3 on table extraction")
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
        run_ppstructure_inference(args, samples)
    elif args.cmd == "score":
        # Reuse eval_hunyuan.py's scoring pipeline directly
        run_scoring(args, samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
