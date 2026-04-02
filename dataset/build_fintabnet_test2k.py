"""Build a 2000-page evaluation dataset from FinTabNet test set.

Output: dataset/fintabnet_test2k/
  ├── images/          # 2000 whole-page PNGs (rendered from PDF at 150 DPI)
  ├── ocr/             # 2000 page-level OCR JSONs (extracted from PDF, line-by-line with bbox)
  ├── gt_html/         # per-page GT: list of HTML tables (reconstructed from tokenized annotation)
  ├── manifest.json    # metadata
  └── split_info.json  # sampling info

Usage:
    python dataset/build_fintabnet_test2k.py
"""

import json
import os
import sys
import random
import fitz  # pymupdf
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

FINTABNET_DIR = "/path/to/fintabnet"
ANNOTATION = os.path.join(FINTABNET_DIR, "FinTabNet_1.0.0_cell_test.jsonl")
PDF_DIR = os.path.join(FINTABNET_DIR, "test")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "fintabnet_test2k")

SEED = 42
N_SAMPLES = 2000
RENDER_DPI = 150


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_html(rec):
    """Reconstruct HTML from FinTabNet tokenized annotation."""
    struct_tokens = rec["html"]["structure"]["tokens"]
    cells = rec["html"]["cells"]
    cell_idx = 0
    html_parts = []
    for token in struct_tokens:
        if token == "</td>":
            if cell_idx < len(cells):
                cell_text = "".join(cells[cell_idx].get("tokens", []))
                html_parts.append(escape_for_xml(cell_text))
                cell_idx += 1
            html_parts.append("</td>")
        else:
            html_parts.append(token)
    return "".join(html_parts)


def escape_for_xml(text):
    """Escape text for XML/HTML compatibility."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    # Convert non-ASCII to numeric references for XML safety
    result = []
    for ch in text:
        if ord(ch) > 127:
            result.append(f"&#{ord(ch)};")
        else:
            result.append(ch)
    return "".join(result)


def render_pdf_page(pdf_path, dpi=150):
    """Render first page of PDF to PIL-compatible PNG bytes."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    width, height = pix.width, pix.height
    doc.close()
    return img_bytes, width, height


def extract_ocr_from_pdf(pdf_path):
    """Extract line-by-line text with bbox from PDF (vector text, not OCR).

    Returns list of {text, bbox} dicts, same format as pubtables_test2k.
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    blocks = page.get_text("dict")["blocks"]
    page_width = page.rect.width
    page_height = page.rect.height

    # Get pixmap dimensions for coordinate scaling
    pix = page.get_pixmap(dpi=RENDER_DPI)
    img_w, img_h = pix.width, pix.height
    scale_x = img_w / page_width
    scale_y = img_h / page_height

    lines = []
    for block in blocks:
        for line in block.get("lines", []):
            text_parts = []
            for span in line["spans"]:
                text_parts.append(span["text"])
            line_text = "".join(text_parts).strip()
            if not line_text:
                continue
            # Scale bbox from PDF coords to image pixel coords
            bbox = line["bbox"]
            scaled_bbox = [
                round(bbox[0] * scale_x, 1),
                round(bbox[1] * scale_y, 1),
                round(bbox[2] * scale_x, 1),
                round(bbox[3] * scale_y, 1),
            ]
            lines.append({"text": line_text, "bbox": scaled_bbox})

    doc.close()
    return lines, [img_w, img_h]


def has_fffd(text):
    """Check for replacement character."""
    return "\ufffd" in text


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "ocr"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "gt_html"), exist_ok=True)

    print("=" * 70)
    print("Building FinTabNet Test 2K Dataset")
    print("=" * 70)

    # ── Step 1: Load annotations and group by page ──
    print("\nStep 1: Loading annotations...")
    page_records = defaultdict(list)  # filename -> [records]
    with open(ANNOTATION, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            page_records[rec["filename"]].append(rec)

    print(f"  Total records: {sum(len(v) for v in page_records.values())}")
    print(f"  Unique pages: {len(page_records)}")

    # ── Step 2: Filter valid pages ──
    print("\nStep 2: Filtering valid pages...")
    valid_pages = []
    for filename, recs in page_records.items():
        pdf_path = os.path.join(PDF_DIR, filename)
        if not os.path.exists(pdf_path):
            continue

        # Check for U+FFFD in cell text
        has_bad = False
        for rec in recs:
            for cell in rec["html"]["cells"]:
                cell_text = "".join(cell.get("tokens", []))
                if has_fffd(cell_text):
                    has_bad = True
                    break
            if has_bad:
                break
        if has_bad:
            continue

        valid_pages.append(filename)

    print(f"  Valid pages (PDF exists + no U+FFFD): {len(valid_pages)}")

    # ── Step 3: Sample 2000 pages ──
    print("\nStep 3: Sampling...")
    rng = random.Random(SEED)
    sampled = sorted(rng.sample(valid_pages, min(N_SAMPLES, len(valid_pages))))
    print(f"  Sampled: {len(sampled)} pages")

    # ── Step 4: Process each page ──
    print(f"\nStep 4: Processing {len(sampled)} pages...")
    manifest = []
    n_ok = 0
    n_err = 0
    total_tables = 0

    for i, filename in enumerate(sampled):
        try:
            pdf_path = os.path.join(PDF_DIR, filename)
            # Page name: HAL/2015/page_43.pdf -> HAL_2015_page_43
            page_name = filename.replace("/", "_").replace("\\", "_").replace(".pdf", "")

            # 1. Render image
            img_bytes, img_w, img_h = render_pdf_page(pdf_path, dpi=RENDER_DPI)
            img_path = os.path.join(OUTPUT_DIR, "images", f"{page_name}.png")
            with open(img_path, "wb") as f:
                f.write(img_bytes)

            # 2. Extract OCR (line-by-line from PDF)
            ocr_lines, img_size = extract_ocr_from_pdf(pdf_path)
            # Filter U+FFFD from OCR too
            ocr_lines = [l for l in ocr_lines if not has_fffd(l["text"])]

            ocr_data = {
                "image_id": page_name,
                "image_size": img_size,
                "ocr": ocr_lines,
            }
            ocr_path = os.path.join(OUTPUT_DIR, "ocr", f"{page_name}_ocr.json")
            with open(ocr_path, "w", encoding="utf-8") as f:
                json.dump(ocr_data, f, ensure_ascii=False)

            # 3. Reconstruct GT HTML for all tables on this page
            recs = page_records[filename]
            # Sort tables by y-position (top to bottom)
            recs.sort(key=lambda r: r["bbox"][1])

            gt_tables = []
            for rec in recs:
                html = reconstruct_html(rec)
                gt_tables.append({
                    "table_id": rec["table_id"],
                    "html": html,
                    "bbox": rec["bbox"],
                })

            gt_data = {
                "page_name": page_name,
                "tables": gt_tables,
            }
            gt_path = os.path.join(OUTPUT_DIR, "gt_html", f"{page_name}_gt.json")
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gt_data, f, ensure_ascii=False)

            manifest.append({
                "page_name": page_name,
                "filename": filename,
                "n_tables": len(gt_tables),
                "table_ids": [t["table_id"] for t in gt_tables],
            })

            n_ok += 1
            total_tables += len(gt_tables)

        except Exception as e:
            n_err += 1
            if n_err <= 5:
                print(f"  ERROR {filename}: {e}")

        if (i + 1) % 200 == 0 or (i + 1) == len(sampled):
            print(f"  [{i+1}/{len(sampled)}] ok={n_ok} err={n_err} tables={total_tables}")

    # ── Step 5: Save manifest and split info ──
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    split_info = {
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "n_written": n_ok,
        "source": "FinTabNet cell_test",
        "pages": [m["page_name"] for m in manifest],
    }
    split_path = os.path.join(OUTPUT_DIR, "split_info.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    # Summary
    from collections import Counter
    dist = Counter(m["n_tables"] for m in manifest)
    print(f"\n  Written: {n_ok} pages, {total_tables} tables")
    print(f"  Errors: {n_err}")
    print(f"  Table distribution:")
    for k in sorted(dist):
        print(f"    {k} table(s): {dist[k]} pages")
    print(f"\n  Output: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
