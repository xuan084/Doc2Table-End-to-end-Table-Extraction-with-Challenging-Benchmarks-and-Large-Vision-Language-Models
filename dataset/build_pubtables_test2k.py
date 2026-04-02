"""Build a 2000-page end-to-end evaluation dataset from PubTables-1M test set.

Output: dataset/pubtables_test2k/
  ├── images/             # 2000 whole-page JPGs (copied from detection test)
  ├── ocr/                # 2000 page-level OCR JSONs (text + bbox per word)
  ├── gt_html/            # per-page GT: list of HTML tables (reconstructed from XML + table_words)
  ├── manifest.json       # metadata: page_name, n_tables, table_names, etc.
  └── split_info.json     # sampling seed + page list

GT HTML reconstruction:
  XML annotation (rows/cols/cells bbox) + table_words (OCR per word)
  -> objects_to_cells() -> cells_to_html()

Usage:
    python dataset/build_pubtables_test2k.py
"""

import json
import os
import sys
import shutil
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "detr"))

DET_IMG_DIR = os.path.join(SCRIPT_DIR, "image", "detection", "detection_image_test")
DET_ANN_DIR = os.path.join(SCRIPT_DIR, "annotation", "detection", "detection_annotation_test")
STRUCT_ANN_DIR = os.path.join(SCRIPT_DIR, "annotation", "recognition", "test")
STRUCT_IMG_DIR = os.path.join(SCRIPT_DIR, "PubTables-1M-Structure_Images_Test")
TABLE_WORDS_DIR = os.path.join(SCRIPT_DIR, "PubTables-1M-Structure_Table_Words")
PAGE_WORDS_DIR = os.path.join(SCRIPT_DIR, "annotation", "page_word")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "pubtables_test2k")
OUT_IMAGES = os.path.join(OUTPUT_DIR, "images")
OUT_OCR = os.path.join(OUTPUT_DIR, "ocr")
OUT_GT_HTML = os.path.join(OUTPUT_DIR, "gt_html")

SEED = 42
N_SAMPLES = 2000
NUM_WORKERS = 12

# Import project's postprocessing code
from postprocess import (objects_to_cells, objects_to_table_structures,
                         table_structure_to_cells, extract_text_from_spans,
                         sort_objects_left_to_right)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Class maps from structure_config.json
STRUCTURE_CLASS_NAMES = [
    'table', 'table column', 'table row', 'table column header',
    'table projected row header', 'table spanning cell', 'no object'
]
# index -> name (used by objects_to_cells to look up label names)
STRUCTURE_CLASS_MAP = {v: k for v, k in enumerate(STRUCTURE_CLASS_NAMES)}
# name -> index (used by parse_structure_xml to convert string labels)
STRUCTURE_CLASS_NAME_TO_IDX = {k: v for v, k in enumerate(STRUCTURE_CLASS_NAMES)}
STRUCTURE_CLASS_THRESHOLDS = {
    "table": 0.5,
    "table column": 0.5,
    "table row": 0.5,
    "table column header": 0.5,
    "table projected row header": 0.5,
    "table spanning cell": 0.5,
    "no object": 10
}


def parse_structure_xml(xml_path):
    """Parse structure annotation XML into objects list.

    Returns objects with integer label indices (as expected by objects_to_cells).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)

    objects = []
    for obj in root.findall("object"):
        label_str = obj.find("name").text
        # Convert string label to index
        label_idx = STRUCTURE_CLASS_NAME_TO_IDX.get(label_str, 6)  # 6 = no object
        bbox_elem = obj.find("bndbox")
        bbox = [
            float(bbox_elem.find("xmin").text),
            float(bbox_elem.find("ymin").text),
            float(bbox_elem.find("xmax").text),
            float(bbox_elem.find("ymax").text),
        ]
        objects.append({
            "label": label_idx,
            "score": 1.0,
            "bbox": bbox,
        })

    return objects, img_w, img_h


def objects_to_html(objects, table_words):
    """Convert structure objects + table words into HTML string.

    Uses the project's postprocess.py functions.
    """
    # Find the table object (label index 0 = "table")
    table_obj = None
    for obj in objects:
        if obj["label"] == 0:
            table_obj = obj
            break

    if table_obj is None:
        return "<table></table>"

    # Add page_num required by objects_to_table_structures
    table_obj["page_num"] = 0

    try:
        _, cells, _ = objects_to_cells(
            table_obj, objects, table_words,
            STRUCTURE_CLASS_MAP, STRUCTURE_CLASS_THRESHOLDS
        )
    except Exception:
        cells = []

    if not cells:
        return "<table></table>"

    return cells_to_html(cells)


def escape_for_xml(text):
    """Escape text to be safe inside XML/HTML tags.

    Converts characters that break xml.etree.ElementTree parsing:
    - & -> &amp;  (must be first)
    - < -> &lt;
    - > -> &gt;
    - Non-ASCII chars that may cause XML parse errors -> HTML numeric entities
    """
    import html as html_mod
    # First escape the 3 critical XML chars
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    # Convert non-ASCII to numeric character references for XML safety
    result = []
    for ch in text:
        if ord(ch) > 127:
            result.append(f"&#{ord(ch)};")
        else:
            result.append(ch)
    return "".join(result)


def cells_to_html(cells):
    """Convert cell list to HTML table string.

    Cell text is XML-escaped to ensure compatibility with both
    TEDS (lxml) and GriTS (xml.etree.ElementTree) parsers.
    """
    if not cells:
        return "<table></table>"

    # Find grid dimensions
    max_row = max(max(c["row_nums"]) for c in cells)
    max_col = max(max(c["column_nums"]) for c in cells)

    # Build grid: (row, col) -> cell index
    grid = {}
    for ci, cell in enumerate(cells):
        for r in cell["row_nums"]:
            for c in cell["column_nums"]:
                grid[(r, c)] = ci

    # Generate HTML
    html_parts = ["<table>"]
    visited = set()

    for row in range(max_row + 1):
        html_parts.append("<tr>")
        for col in range(max_col + 1):
            ci = grid.get((row, col))
            if ci is None or ci in visited:
                continue
            visited.add(ci)
            cell = cells[ci]

            rowspan = len(cell["row_nums"])
            colspan = len(cell["column_nums"])

            tag = "<td"
            if colspan > 1:
                tag += f' colspan="{colspan}"'
            if rowspan > 1:
                tag += f' rowspan="{rowspan}"'
            tag += ">"

            text = escape_for_xml(cell.get("cell_text", ""))
            html_parts.append(f"{tag}{text}</td>")

        html_parts.append("</tr>")

    html_parts.append("</table>")
    return "".join(html_parts)


def group_words_into_lines(words, y_tolerance=5):
    """Group words into lines by clustering on y-center coordinate."""
    if not words:
        return []

    for w in words:
        w["_y_center"] = (w["bbox"][1] + w["bbox"][3]) / 2

    sorted_words = sorted(words, key=lambda w: (w["_y_center"], w["bbox"][0]))

    lines = []
    current_line = [sorted_words[0]]

    for w in sorted_words[1:]:
        if abs(w["_y_center"] - current_line[-1]["_y_center"]) <= y_tolerance:
            current_line.append(w)
        else:
            current_line.sort(key=lambda w: w["bbox"][0])
            lines.append(current_line)
            current_line = [w]

    if current_line:
        current_line.sort(key=lambda w: w["bbox"][0])
        lines.append(current_line)

    return lines


def build_page_ocr(page_words):
    """Build line-by-line OCR data from page words.

    Returns list of {text, bbox} dicts, one per line,
    matching the format of data/WildDocTables/ocr/.
    """
    if not page_words:
        return []

    lines = group_words_into_lines(page_words)
    ocr_lines = []
    for line_words in lines:
        text = " ".join(w["text"] for w in line_words)
        # Compute line-level bbox (union of all word bboxes)
        x_min = min(w["bbox"][0] for w in line_words)
        y_min = min(w["bbox"][1] for w in line_words)
        x_max = max(w["bbox"][2] for w in line_words)
        y_max = max(w["bbox"][3] for w in line_words)
        ocr_lines.append({
            "text": text,
            "bbox": [round(x_min, 1), round(y_min, 1), round(x_max, 1), round(y_max, 1)],
        })

    return ocr_lines


def map_tables_to_page(page_name, pmc_to_struct_tables):
    """Find which structure test tables belong to this detection page."""
    pmc = page_name.rsplit("_", 1)[0]
    page_num = page_name.rsplit("_", 1)[1]

    struct_tables = pmc_to_struct_tables.get(pmc, [])
    if not struct_tables:
        return []

    # Load page words
    pw_path = os.path.join(PAGE_WORDS_DIR, f"{page_name}_words.json")
    if not os.path.exists(pw_path):
        return []
    with open(pw_path, "r") as f:
        pw = json.load(f)

    pw_words = pw["words"]
    matched = []

    for table_name in struct_tables:
        tw_path = os.path.join(TABLE_WORDS_DIR, f"{table_name}_words.json")
        if not os.path.exists(tw_path):
            continue
        with open(tw_path, "r") as f:
            tw = json.load(f)
        if not tw:
            continue

        span_num = tw[0].get("span_num", -1)
        if 0 <= span_num < len(pw_words):
            if pw_words[span_num]["text"] == tw[0]["text"]:
                matched.append(table_name)

    return matched


def process_one_page(page_name, pmc_to_struct_tables):
    """Process one page: map tables, reconstruct GT HTML, build OCR."""
    result = {
        "page_name": page_name,
        "tables": [],
        "error": None,
    }

    try:
        # Map structure tables to this page
        table_names = map_tables_to_page(page_name, pmc_to_struct_tables)
        if not table_names:
            result["error"] = "no_tables_mapped"
            return result

        # Load page words for OCR
        pw_path = os.path.join(PAGE_WORDS_DIR, f"{page_name}_words.json")
        with open(pw_path, "r") as f:
            pw = json.load(f)

        # For each mapped table, reconstruct GT HTML
        gt_htmls = []
        for table_name in sorted(table_names):
            # Load structure annotation
            ann_path = os.path.join(STRUCT_ANN_DIR, f"{table_name}.xml")
            if not os.path.exists(ann_path):
                continue

            objects, img_w, img_h = parse_structure_xml(ann_path)

            # Load table words
            tw_path = os.path.join(TABLE_WORDS_DIR, f"{table_name}_words.json")
            with open(tw_path, "r") as f:
                table_words = json.load(f)

            # Reconstruct HTML
            html = objects_to_html(objects, table_words)
            gt_htmls.append({
                "table_name": table_name,
                "html": html,
            })

        result["tables"] = gt_htmls

        # Build line-by-line OCR
        ocr_lines = build_page_ocr(pw["words"])
        result["ocr_lines"] = ocr_lines
        result["n_page_words"] = len(pw["words"])

    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_IMAGES, exist_ok=True)
    os.makedirs(OUT_OCR, exist_ok=True)
    os.makedirs(OUT_GT_HTML, exist_ok=True)

    print("=" * 70)
    print("Building PubTables-1M Test 2K Dataset")
    print("=" * 70)

    # ── Step 1: Build structure table index ──
    print("\nStep 1: Building structure table index...")
    struct_tables = sorted(f.replace(".jpg", "") for f in os.listdir(STRUCT_IMG_DIR))
    pmc_to_struct_tables = defaultdict(list)
    for t in struct_tables:
        pmc = t.split("_table_")[0]
        pmc_to_struct_tables[pmc].append(t)
    print(f"  Structure test tables: {len(struct_tables)}")
    print(f"  Unique PMCs: {len(pmc_to_struct_tables)}")

    # ── Step 2: Find valid pages (can map to at least 1 structure table) ──
    print("\nStep 2: Finding valid pages (with mappable structure tables)...")
    det_pages = sorted(f.replace(".jpg", "") for f in os.listdir(DET_IMG_DIR))
    print(f"  Detection test pages: {len(det_pages)}")

    # Build PMC -> page_words index
    pmc_to_pw_pages = defaultdict(list)
    for f in os.listdir(PAGE_WORDS_DIR):
        if f.endswith("_words.json"):
            parts = f.replace("_words.json", "").rsplit("_", 1)
            if len(parts) == 2:
                pmc_to_pw_pages[parts[0]].append((parts[1], f))

    valid_pages = []
    for page_name in det_pages:
        pmc = page_name.rsplit("_", 1)[0]
        if pmc not in pmc_to_struct_tables:
            continue

        pw_path = os.path.join(PAGE_WORDS_DIR, f"{page_name}_words.json")
        if not os.path.exists(pw_path):
            continue

        # Quick check: can at least 1 structure table map to this page?
        with open(pw_path, "r") as f:
            pw = json.load(f)
        pw_words = pw["words"]

        for table_name in pmc_to_struct_tables[pmc]:
            tw_path = os.path.join(TABLE_WORDS_DIR, f"{table_name}_words.json")
            if not os.path.exists(tw_path):
                continue
            with open(tw_path, "r") as f:
                tw = json.load(f)
            if not tw:
                continue
            span_num = tw[0].get("span_num", -1)
            if 0 <= span_num < len(pw_words):
                if pw_words[span_num]["text"] == tw[0]["text"]:
                    valid_pages.append(page_name)
                    break

    print(f"  Valid pages (mappable): {len(valid_pages)}")

    rng = random.Random(SEED)
    sampled_pages = sorted(rng.sample(valid_pages, N_SAMPLES))
    print(f"  Sampled: {N_SAMPLES} pages (seed={SEED})")

    # ── Step 2b: Pre-filter — exclude pages whose GT contains U+FFFD ──
    print("\nStep 2b: Filtering out pages with U+FFFD (broken font mapping)...")
    filtered_pages = []
    n_fffd = 0
    for page_name in sampled_pages:
        # Quick process to check for U+FFFD
        pmc = page_name.rsplit("_", 1)[0]
        table_names = map_tables_to_page(page_name, pmc_to_struct_tables)
        has_fffd = False
        for table_name in table_names:
            tw_path = os.path.join(TABLE_WORDS_DIR, f"{table_name}_words.json")
            if not os.path.exists(tw_path):
                continue
            with open(tw_path, "r") as f:
                tw = json.load(f)
            for w in tw:
                if "\ufffd" in w.get("text", ""):
                    has_fffd = True
                    break
            if has_fffd:
                break
        if has_fffd:
            n_fffd += 1
        else:
            filtered_pages.append(page_name)

    print(f"  Removed {n_fffd} pages with U+FFFD")
    print(f"  Remaining: {len(filtered_pages)} pages")

    # If we lost pages, sample replacements from the valid pool
    if len(filtered_pages) < N_SAMPLES:
        already_used = set(filtered_pages)
        candidates = [p for p in valid_pages if p not in already_used]
        rng2 = random.Random(SEED + 1)
        rng2.shuffle(candidates)
        for cand in candidates:
            if len(filtered_pages) >= N_SAMPLES:
                break
            # Check candidate for U+FFFD
            table_names = map_tables_to_page(cand, pmc_to_struct_tables)
            has_fffd = False
            for table_name in table_names:
                tw_path = os.path.join(TABLE_WORDS_DIR, f"{table_name}_words.json")
                if not os.path.exists(tw_path):
                    continue
                with open(tw_path, "r") as f:
                    tw = json.load(f)
                for w in tw:
                    if "\ufffd" in w.get("text", ""):
                        has_fffd = True
                        break
                if has_fffd:
                    break
            if not has_fffd:
                filtered_pages.append(cand)

        print(f"  After replacement: {len(filtered_pages)} pages")

    sampled_pages = sorted(filtered_pages[:N_SAMPLES])

    # ── Step 3: Process each page ──
    print(f"\nStep 3: Processing {N_SAMPLES} pages...")
    results = []
    n_ok = 0
    n_no_tables = 0
    n_errors = 0
    total_tables = 0

    for i, page_name in enumerate(sampled_pages):
        r = process_one_page(page_name, pmc_to_struct_tables)
        results.append(r)

        if r["error"]:
            n_errors += 1
        elif not r["tables"]:
            n_no_tables += 1
        else:
            n_ok += 1
            total_tables += len(r["tables"])

        if (i + 1) % 200 == 0 or (i + 1) == N_SAMPLES:
            print(f"  [{i+1}/{N_SAMPLES}] ok={n_ok} no_tables={n_no_tables} errors={n_errors} "
                  f"total_tables={total_tables}")

    print(f"\n  Final: {n_ok} pages with tables, {total_tables} total tables")
    print(f"  No tables mapped: {n_no_tables}")
    print(f"  Errors: {n_errors}")

    # ── Step 4: Write output ──
    print(f"\nStep 4: Writing output to {OUTPUT_DIR}...")

    manifest = []
    written = 0

    for r in results:
        page_name = r["page_name"]

        if r["error"] or not r["tables"]:
            continue

        # Copy image
        src_img = os.path.join(DET_IMG_DIR, f"{page_name}.jpg")
        dst_img = os.path.join(OUT_IMAGES, f"{page_name}.jpg")
        shutil.copy2(src_img, dst_img)

        # Write OCR (line-by-line format, same as generated dataset)
        from PIL import Image as PILImage
        img = PILImage.open(os.path.join(OUT_IMAGES, f"{page_name}.jpg"))
        ocr_data = {
            "image_id": page_name,
            "image_size": list(img.size),
            "ocr": r.get("ocr_lines", []),
        }
        ocr_path = os.path.join(OUT_OCR, f"{page_name}_ocr.json")
        with open(ocr_path, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False)

        # Write GT HTML
        gt_data = {
            "page_name": page_name,
            "tables": [{"table_name": t["table_name"], "html": t["html"]}
                       for t in r["tables"]],
        }
        gt_path = os.path.join(OUT_GT_HTML, f"{page_name}_gt.json")
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, ensure_ascii=False)

        manifest.append({
            "page_name": page_name,
            "n_tables": len(r["tables"]),
            "table_names": [t["table_name"] for t in r["tables"]],
        })
        written += 1

    # Write manifest
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Write split info
    split_info = {
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "n_written": written,
        "source": "PubTables-1M detection test",
        "pages": [m["page_name"] for m in manifest],
    }
    split_path = os.path.join(OUTPUT_DIR, "split_info.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    # Summary
    table_counts = defaultdict(int)
    for m in manifest:
        table_counts[m["n_tables"]] += 1

    print(f"\n  Written: {written} pages")
    print(f"  Table count distribution:")
    for n in sorted(table_counts):
        print(f"    {n} table(s): {table_counts[n]} pages")

    print(f"\nOutput:")
    print(f"  Images:   {OUT_IMAGES}")
    print(f"  OCR:      {OUT_OCR}")
    print(f"  GT HTML:  {OUT_GT_HTML}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Split:    {split_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
