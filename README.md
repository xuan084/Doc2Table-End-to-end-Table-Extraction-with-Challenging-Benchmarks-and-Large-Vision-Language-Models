# Doc2Table: End-to-End Table Extraction with Large Vision-Language Models

This repository contains the code and sample data for reproducing the experiments in our paper.

## Repository Structure

```
.
├── src/                          # Core evaluation & inference utilities
│   ├── grits.py                  # GriTS metric implementation
│   ├── eval.py                   # Evaluation functions (TEDS + GriTS)
│   ├── inference.py              # DETR+UniTable pipeline
│   ├── postprocess.py            # Post-processing utilities
│   ├── table_datasets.py         # Dataset loading utilities
│   ├── main.py                   # Entry point for table structure recognition
│   ├── detection_config.json     # DETR detection config
│   └── structure_config.json     # DETR structure config
│
├── agent_eval/                   # Large VLM inference & evaluation
│   ├── run_llm_on_generated.py   # GPT-5.2 on WildDocTables
│   ├── run_llm_on_fintabnet.py   # GPT-5.2 on FinTabNet
│   ├── run_llm_on_pubtables.py   # GPT-5.2 on PubTables-1M
│   ├── run_llm_on_test.py        # GPT-5.2 (quick test)
│   ├── eval_finetuned_on_dsw.py  # Qwen2.5-VL zero-shot
│   ├── eval_finetuned_on_dsw_tuned.py  # Qwen2.5-VL fine-tuned (LoRA)
│   ├── eval_finetuned_7gpu.py    # Multi-GPU fine-tuned Qwen
│   └── sample/
│       ├── build_sft_dataset.py  # Build SFT training data for Qwen
│       └── split_test.json       # Test split image IDs (2K)
│
├── pipeline_eval/                # DETR+UniTable pipeline evaluation
│   ├── run_pipeline_on_generated.py
│   ├── run_pipeline_on_generated_batch.py
│   └── run_pipeline_on_fintabnet.py
│
├── hunyuan/                      # Baseline model evaluation
│   ├── eval_hunyuan.py           # HunyuanOCR (Tencent, 1B VLM)
│   ├── eval_mineru.py            # MinerU (document parser)
│   └── eval_ppstructure.py       # PP-StructureV3 (PaddleOCR)
│
├── unitable/                     # TEDS metric (tree-edit-distance scorer)
│
├── detr/                         # DETR table detection model code
│
├── data/WildDocTables/            # WildDocTables test split (2K pages)
│   ├── images/                   # 2000 document page images (.png)
│   ├── ocr/                      # OCR transcripts per page
│   ├── gt_json_v1/               # Ground truth with bboxes
│   ├── gt_json_v2/               # Ground truth HTML (no thead/tbody)
│   └── manifest.jsonl            # Table attribute metadata (230K entries)
│
├── dataset/                      # Dataset preparation scripts + test data
│   ├── fintabnet_test2k/         # FinTabNet test split (2K pages)
│   │   ├── images/               # 2000 page images (.png)
│   │   ├── gt_html/              # Ground truth HTML per page
│   │   ├── ocr/                  # OCR transcripts
│   │   └── manifest.json         # Page index
│   ├── pubtables_test2k/         # PubTables-1M test split (2K pages)
│   │   ├── images/               # 2000 page images (.jpg)
│   │   ├── gt_html/              # Ground truth HTML per page
│   │   ├── ocr/                  # OCR transcripts
│   │   └── manifest.json         # Page index
│   ├── build_fintabnet_test2k.py
│   ├── build_pubtables_test2k.py
│   └── convert_to_swift_jsonl.py
│
└── environment.yml               # Conda environment specification
```

## Dataset

This repository includes test splits (2K pages each) for all three datasets used in our experiments, ready to run:
- **WildDocTables** (2K test pages): `data/WildDocTables/`
- **FinTabNet** (2K test pages): `dataset/fintabnet_test2k/`
- **PubTables-1M** (2K test pages): `dataset/pubtables_test2k/`

The full WildDocTables dataset (230K pages, 386K tables) will be released separately.

## Setup

```bash
conda env create -f environment.yml
conda activate table-transformer
```

## API Keys

Before running LLM-based experiments, set your API key:
- For GPT-5.2: Fill `API_KEY` in `agent_eval/run_llm_on_*.py`

## Running Experiments

### Large VLM Inference (GPT-5.2)
```bash
# On WildDocTables (generated dataset)
python agent_eval/run_llm_on_generated.py

# On FinTabNet
python agent_eval/run_llm_on_fintabnet.py

# On PubTables-1M
python agent_eval/run_llm_on_pubtables.py
```

### Qwen2.5-VL Inference
```bash
# Zero-shot
python agent_eval/eval_finetuned_on_dsw.py

# Fine-tuned (LoRA)
python agent_eval/eval_finetuned_on_dsw_tuned.py
```

### Baseline Models
```bash
# HunyuanOCR (requires vLLM serving)
python hunyuan/eval_hunyuan.py infer --dataset generated
python hunyuan/eval_hunyuan.py score --dataset generated

# MinerU
python hunyuan/eval_mineru.py infer --dataset generated
python hunyuan/eval_mineru.py score --dataset generated

# PP-StructureV3
python hunyuan/eval_ppstructure.py infer --dataset generated
python hunyuan/eval_ppstructure.py score --dataset generated
```

### DETR+UniTable Pipeline
```bash
python pipeline_eval/run_pipeline_on_generated_batch.py
```

## Evaluation Metrics

All experiments are evaluated using our proposed **PMT** (Page-level Multi-Table) evaluation protocol, which extends TEDS and GriTS to the page level with:
- **Hungarian matching** for optimal table-to-table assignment
- **Cell-count weighting** for size-aware scoring
- **Weighted Precision/Recall/F_beta** for separate detection and recognition assessment

Scores are reported on a 0-100 scale. Results are saved as `eval_summary.json` (aggregated) and `eval_detail.jsonl` (per-image).

## Hardware Requirements

- **Training**: 7x NVIDIA A800-80GB GPUs (for Qwen LoRA fine-tuning)
- **Inference**: 1-2 GPUs sufficient for most models; HunyuanOCR/MinerU/PP-Structure can run on single GPU
- **GPT-5.2**: Requires OpenAI API access (no local GPU needed)
