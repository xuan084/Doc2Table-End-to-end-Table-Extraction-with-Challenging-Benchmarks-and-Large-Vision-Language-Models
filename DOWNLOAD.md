# Full Dataset & Model Download

The full WildDocTables dataset and the fine-tuned model checkpoint are available at:

**[Google Drive](https://drive.google.com/drive/folders/1HkafPW5CexffQMtpJObXAZtfzKadq-pv?usp=sharing)**

## Contents

| File | Size | Description |
|------|------|-------------|
| `images.rar` | 9.0 GB | 225K document page images (.png) |
| `gt_json_v1.rar` | 1.5 GB | Ground truth with bounding boxes |
| `gt_json_v2.rar` | 1.5 GB | Ground truth HTML (without thead/tbody) |
| `ocr.rar` | 339 MB | OCR transcripts per page |
| `manifest.jsonl` | 41 MB | Table attribute metadata (230K entries) |
| `checkpoint-1100.tar.gz` | 213 MB | Qwen2.5-VL-7B LoRA checkpoint, fine-tuned on WildDocTables |

## Setup

After downloading, extract and place files as follows:

```
data/WildDocTables/
  ├── images/          # extract from images.rar
  ├── ocr/             # extract from ocr.rar
  ├── gt_json_v1/      # extract from gt_json_v1.rar
  ├── gt_json_v2/      # extract from gt_json_v2.rar
  └── manifest.jsonl   # place directly
```

The LoRA checkpoint (`checkpoint-1100.tar.gz`) can be used with the Qwen2.5-VL-7B base model for fine-tuned inference:
```bash
python agent_eval/eval_finetuned_on_dsw_tuned.py \
    --model /path/to/Qwen2.5-VL-7B-Instruct \
    --adapter /path/to/checkpoint-1100 \
    --test_jsonl /path/to/test.jsonl \
    --output_dir /path/to/output
```

## Note

The 2K test splits included in this repository (`data/WildDocTables/`, `dataset/fintabnet_test2k/`, `dataset/pubtables_test2k/`) are sufficient for reproducing all experiments reported in the paper. The full dataset above is provided for comprehensive evaluation and model training.
