"""Convert PubTables-1M and FinTabNet test 2K datasets to SWIFT JSONL format.

This produces test.jsonl files that can be used with eval_finetuned_on_dsw.py
to evaluate Qwen fine-tuned and base models on these external datasets.

Output format per line (SWIFT standard):
{
  "messages": [
    {"role": "system", "content": "...system prompt..."},
    {"role": "user", "content": "<image>...OCR + task instructions..."},
    {"role": "assistant", "content": "<table>...</table>\n\n<table>...</table>"}
  ],
  "images": ["/absolute/path/to/image.ext"]
}
"""

import json
import os
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT (identical to training data)
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


def build_user_content(ocr_text):
    """Build user content string with <image> tag and OCR text."""
    parts = ["<image>"]

    parts.append("**OCR Reference (for text accuracy)**:\n")
    parts.append("The following OCR text was extracted from the target document page. "
                 "You can use it as a reference to improve text recognition accuracy:\n")
    parts.append("```")
    parts.append(ocr_text)
    parts.append("```\n")
    parts.append("**Note**: The OCR text is provided as a reference to help you recognize "
                 "text content more accurately. However, you should still rely primarily on "
                 "the visual structure from the image for determining table layout, rows, "
                 "columns, and cell boundaries.\n")
    parts.append("---\n")
    parts.append("**Your Task**:\n")
    parts.append("Extract ALL tables from the image and convert each to HTML following "
                 "the format described above.\n")
    parts.append("**Before you generate the HTML**:")
    parts.append("1. First, determine how many tables are on the page — do NOT merge "
                 "separate tables into one or split one table into two")
    parts.append("2. Count the exact number of rows in each table")
    parts.append("3. Count the exact number of columns in each table")
    parts.append("4. Identify where each table starts and ends")
    parts.append("5. Do NOT add any extra rows or columns beyond what you see")
    parts.append("6. Use the OCR reference text above to help with accurate text "
                 "recognition, but determine the table structure from the visual image.")

    return "\n".join(parts)


def convert_dataset(data_dir, output_jsonl, image_ext, dsw_image_base):
    """Convert a dataset to SWIFT JSONL format.

    Args:
        data_dir: Local dataset directory (e.g. dataset/pubtables_test2k)
        output_jsonl: Output JSONL path
        image_ext: Image extension ('.jpg' or '.png')
        dsw_image_base: Absolute image path prefix on DSW
    """
    manifest = json.load(open(os.path.join(data_dir, "manifest.json"), encoding="utf-8"))
    print(f"Converting {len(manifest)} pages from {data_dir}")

    n_written = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for m in manifest:
            page_name = m["page_name"]

            # Load OCR
            ocr_path = os.path.join(data_dir, "ocr", f"{page_name}_ocr.json")
            ocr = json.load(open(ocr_path, encoding="utf-8"))
            ocr_text = "\n".join(entry["text"] for entry in ocr["ocr"])

            # Load GT HTML
            gt_path = os.path.join(data_dir, "gt_html", f"{page_name}_gt.json")
            gt = json.load(open(gt_path, encoding="utf-8"))
            gt_htmls = [t["html"] for t in gt["tables"]]
            assistant_content = "\n\n".join(gt_htmls)

            # Build user content
            user_content = build_user_content(ocr_text)

            # Image path on DSW
            image_abs = f"{dsw_image_base}/{page_name}{image_ext}"

            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "images": [image_abs],
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"  Written: {n_written} lines to {output_jsonl}")
    return n_written


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ── PubTables-1M ──
    pub_dir = os.path.join(SCRIPT_DIR, "pubtables_test2k")
    pub_out = os.path.join(pub_dir, "test_swift.jsonl")
    convert_dataset(
        data_dir=pub_dir,
        output_jsonl=pub_out,
        image_ext=".jpg",
        dsw_image_base="/path/to/pubtables_test2k/images",
    )

    # ── FinTabNet ──
    fin_dir = os.path.join(SCRIPT_DIR, "fintabnet_test2k")
    fin_out = os.path.join(fin_dir, "test_swift.jsonl")
    convert_dataset(
        data_dir=fin_dir,
        output_jsonl=fin_out,
        image_ext=".png",
        dsw_image_base="/path/to/fintabnet_test2k/images",
    )

    print("\nDone. Upload to server:")
    print(f"  {pub_out}")
    print(f"  {fin_out}")
    print("  + corresponding images/ directories")
