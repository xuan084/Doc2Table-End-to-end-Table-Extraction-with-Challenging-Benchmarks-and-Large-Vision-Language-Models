# WildDocTables Dataset

Place the WildDocTables dataset here with the following structure:

```
WildDocTables/
├── images/          # Document page images (PNG)
├── gt_json_v1/      # Ground truth with bboxes (for DETR pipeline)
├── gt_json_v2/      # Ground truth HTML tables (for scoring)
├── ocr/             # OCR transcripts (JSON)
└── manifest.jsonl   # Image metadata with subset (S1-S4) and attributes (T1-T4)
```

The full dataset will be released separately.
