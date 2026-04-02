"""
Build SFT dataset for Qwen3-VL fine-tuning.

Reads the split files (split_train.json, split_val.json, split_test.json),
then for each image:
  - Extracts GT HTML from gt_json_v2 (tables sorted by y_min)
  - Extracts OCR text from ocr/*.json
  - Assembles into Qwen-VL SFT message format

Output:
  qwen_sft/
  ├── images/              # symlinks or copies of 20K PNGs
  ├── train.jsonl          # 17,000 samples
  ├── val.jsonl            # 1,000 samples
  └── test.jsonl           # 2,000 samples

Usage:
    python build_sft_dataset.py [--copy-images]

    --copy-images   Copy images instead of symlinking (for upload to cloud)

Deterministic: output order matches sorted image_ids in split files.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PIPELINE_DIR = PROJECT_ROOT / "data" / "WildDocTables"
GT_V2_DIR = PIPELINE_DIR / "gt_json_v2"
OCR_DIR = PIPELINE_DIR / "ocr"
IMAGES_DIR = PIPELINE_DIR / "images"

OUTPUT_DIR = SCRIPT_DIR / "qwen_sft"

SPLIT_FILES = {
    "train": SCRIPT_DIR / "split_train.json",
    "val": SCRIPT_DIR / "split_val.json",
    "test": SCRIPT_DIR / "split_test.json",
}

# ============================================================
# Prompt templates
# ============================================================

# Aligned with agent_eval/run_llm_on_generated.py SYSTEM_PROMPT
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

## Common Mistakes to AVOID

DO NOT:
- Add extra rows beyond what's visible in the image
- Create cells for empty spaces outside the table
- Treat paragraph text as table rows
- Merge two separate tables into one
- Split one table into two

DO:
- Stop exactly where each table ends
- Leave cells empty (`<td></td>`) if they appear empty in the image
- Match the exact structure you see
- Output each table as a separate `<table>...</table>` block

## Output Format

- Return one or more bare `<table>...</table>` HTML fragments
- If there are multiple tables, output them in top-to-bottom order
- Do NOT include <html>, <head>, <body>, <thead>, <tbody> wrappers
- Use <td> for ALL cells (never <th>)
- Separate multiple tables with a blank line"""

# Aligned with agent_eval/run_llm_on_generated.py user prompt structure
USER_TEMPLATE = """**OCR Reference (for text accuracy)**:

The following OCR text was extracted from the target document page. You can use it as a reference to improve text recognition accuracy:

```
{ocr_text}
```

**Note**: The OCR text is provided as a reference to help you recognize text content more accurately. However, you should still rely primarily on the visual structure from the image for determining table layout, rows, columns, and cell boundaries.

---

**Your Task**:

Extract ALL tables from the image and convert each to HTML following the format described above.

**Before you generate the HTML**:
1. First, determine how many tables are on the page — do NOT merge separate tables into one or split one table into two
2. Count the exact number of rows in each table
3. Count the exact number of columns in each table
4. Identify where each table starts and ends
5. Do NOT add any extra rows or columns beyond what you see
6. Use the OCR reference text above to help with accurate text recognition, but determine the table structure from the visual image."""


# ============================================================
# Core functions
# ============================================================

def load_split(split_name):
    """Load image IDs from a split file."""
    path = SPLIT_FILES[split_name]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["image_ids"]


def extract_gt_html(img_id):
    """Extract sorted GT HTML strings from gt_json_v2."""
    path = GT_V2_DIR / f"{img_id}_v2.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tables = [b for b in data["page"] if b["type"] == "table"]
    # Sort by y_min (bbox[1]) for deterministic top-to-bottom order
    tables.sort(key=lambda t: t["bbox"][1])

    htmls = [t["html"] for t in tables]
    return htmls


def extract_ocr_text(img_id):
    """Extract OCR text (line-level, joined by newlines)."""
    path = OCR_DIR / f"{img_id}_ocr.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = [entry["text"] for entry in data["ocr"]]
    return "\n".join(lines)


def build_assistant_content(htmls):
    """Build assistant response: bare <table> blocks separated by blank lines.

    Aligned with eval script's extract_tables_from_response() which uses
    regex to find <table>...</table> blocks in the raw response.
    """
    return "\n\n".join(htmls)


def build_sample(img_id, image_rel_path):
    """Build one SFT sample in Qwen-VL message format."""
    htmls = extract_gt_html(img_id)
    ocr_text = extract_ocr_text(img_id)

    assistant_content = build_assistant_content(htmls)
    user_text = USER_TEMPLATE.format(ocr_text=ocr_text)

    sample = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rel_path},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ]
    }
    return sample


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset for Qwen-VL")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images instead of symlinking")
    args = parser.parse_args()

    # Create output directories
    out_images = OUTPUT_DIR / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)

    # Collect all image IDs across splits
    all_ids = set()
    split_ids = {}
    for split_name in ["train", "val", "test"]:
        ids = load_split(split_name)
        split_ids[split_name] = ids
        all_ids.update(ids)
        print(f"  {split_name}: {len(ids)} images", flush=True)
    print(f"  Total unique: {len(all_ids)}", flush=True)

    # Step 1: Link/copy images
    print(f"\nStep 1: {'Copying' if args.copy_images else 'Symlinking'} {len(all_ids)} images...",
          flush=True)
    done = 0
    for img_id in sorted(all_ids):
        src = IMAGES_DIR / f"{img_id}.png"
        dst = out_images / f"{img_id}.png"
        if dst.exists():
            done += 1
            continue
        if args.copy_images:
            shutil.copy2(src, dst)
        else:
            # On Windows, symlinks may require elevated permissions;
            # fall back to copy if symlink fails
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
        done += 1
        if done % 5000 == 0:
            print(f"  {done}/{len(all_ids)} images...", flush=True)
    print(f"  Done: {done} images.", flush=True)

    # Step 2: Build JSONL for each split
    for split_name in ["train", "val", "test"]:
        ids = split_ids[split_name]
        out_path = OUTPUT_DIR / f"{split_name}.jsonl"
        print(f"\nStep 2: Building {split_name}.jsonl ({len(ids)} samples)...", flush=True)

        errors = 0
        with open(out_path, "w", encoding="utf-8") as fout:
            for i, img_id in enumerate(ids):
                try:
                    # Image path relative to the JSONL location (qwen_sft/)
                    image_rel_path = f"images/{img_id}.png"
                    sample = build_sample(img_id, image_rel_path)
                    line = json.dumps(sample, ensure_ascii=False)
                    fout.write(line + "\n")
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"  ERROR {img_id}: {e}", flush=True)

                if (i + 1) % 5000 == 0:
                    print(f"  {i + 1}/{len(ids)}...", flush=True)

        print(f"  Done: {len(ids) - errors} ok, {errors} errors.", flush=True)

    # Step 3: Summary stats
    print("\n" + "=" * 60, flush=True)
    print("Summary", flush=True)
    print("=" * 60, flush=True)

    for split_name in ["train", "val", "test"]:
        jsonl_path = OUTPUT_DIR / f"{split_name}.jsonl"
        n_lines = 0
        total_assistant_chars = 0
        total_tables = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                n_lines += 1
                sample = json.loads(line)
                asst = sample["messages"][2]["content"]
                total_assistant_chars += len(asst)
                total_tables += asst.count("<table>")

        file_size = jsonl_path.stat().st_size
        print(f"\n  {split_name}.jsonl:", flush=True)
        print(f"    Samples: {n_lines}", flush=True)
        print(f"    Tables: {total_tables}", flush=True)
        print(f"    Avg tables/sample: {total_tables / max(n_lines, 1):.2f}", flush=True)
        print(f"    Avg assistant chars: {total_assistant_chars / max(n_lines, 1):.0f}", flush=True)
        print(f"    File size: {file_size / 1024 / 1024:.1f} MB", flush=True)

    # Images dir size
    img_total = sum(f.stat().st_size for f in out_images.iterdir() if f.is_file())
    print(f"\n  images/ total: {img_total / 1024 / 1024 / 1024:.2f} GB "
          f"({sum(1 for _ in out_images.iterdir())} files)", flush=True)

    print(f"\nAll done. Output at: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
