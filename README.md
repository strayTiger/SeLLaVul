# SeLLaVul: Selective LLM-Audited Boundary Refinement for Function-Level Vulnerability Detection

SeLLaVul is a function-level software vulnerability detection framework that combines a pretrained code model with selective LLM auditing and constrained supervision injection.

The key idea is simple: instead of asking an LLM to judge every function, SeLLaVul first trains a neural vulnerability detector, finds the most uncertain **gray-zone** samples near the decision boundary, audits only those difficult samples with an LLM, and then injects LLM-derived information back into model training only when it is high-confidence and label-consistent.

## Overview

Existing pretrained code models, such as GraphCodeBERT, can learn useful syntax and semantic patterns from source code. However, they may still make unstable predictions for samples near the decision boundary, especially when vulnerability evidence depends on subtle control-flow logic, input validation, bounds checking, error handling, or other security-sensitive patterns.

SeLLaVul addresses this problem through a three-stage framework:

1. **Stage 1: Baseline detector training**  
   Train a GraphCodeBERT-based vulnerability detector and export prediction artifacts, including out-of-fold predictions for the training set.

2. **Stage 2: Selective LLM auditing**  
   Select gray-zone samples near the baseline decision boundary and query an LLM only for these uncertain samples. The LLM returns conservative binary judgments and structured auxiliary attributes.

3. **Stage 3: Constrained refinement training**  
   Build a refined training set. LLM information is used as soft-label supervision or auxiliary supervision only when the LLM output is valid, confident, and consistent with the original gold label. The refined GraphCodeBERT model is then trained from the Stage-1 checkpoint.

## Design of SeLLaVul
![Pipeline](./sevel_flow_01.png)

## Dataset
The dataset can be downloaded at: https://drive.google.com/file/d/1EU3wztfOpmbQdvZoMDAqtnvSEmLT1iL9/view?usp=sharing

## Main Features

- Function-level vulnerability detection for C/C++ code.
- GraphCodeBERT-based baseline and refinement model.
- Gray-zone sample selection based on prediction uncertainty.
- LLM auditing for only a small subset of difficult samples.
- Optional CFG-aware prompting in the LLM audit stage.
- Conservative gating to suppress speculative or low-evidence LLM outputs.
- Label-consistent soft-label learning.
- Five auxiliary supervision signals:
  - input validation
  - bounds checking
  - untrusted input usage
  - null checking
  - error-handling path
- Evaluation with F1, precision, recall,ACC, AUC, AP, and P@K/R@K.

## Repository Structure

A recommended repository layout is shown below. You may rename the scripts according to your actual file names.

```text
SeLLaVul/
├── README.md
├── requirements.txt
├── data/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── models/
│   └── graphcodebert_model/
├── scripts/
│   ├── stage1_baselines.py
│   ├── stage2_llm_features.py
│   ├── build_refine_dataset.py
│   └── train_refine.py
├── artifacts/
│   └── .gitkeep
├── result/
│   └── .gitkeep
└── runs/
    └── .gitkeep
```

## Requirements

The code was developed with Python and PyTorch. A CUDA-enabled GPU is recommended for model training.

```bash
pip install torch transformers scikit-learn numpy pandas tqdm requests
```

A minimal `requirements.txt` may contain:

```text
torch
transformers
scikit-learn
numpy
pandas
tqdm
requests
```

## Dataset Format

Each dataset split should be stored as a JSONL file. Each line represents one function-level sample.

```json
{"idx": "sample_0001", "func": "int foo(...) { ... }", "target": 0}
```

Required fields:

| Field | Description |
|---|---|
| `idx` | Unique function/sample ID. Other supported names may include `id`, `uid`, `func_id`, or `func_name`. |
| `func` | Source code of the function. |
| `target` | Ground-truth label, where `1` means vulnerable and `0` means non-vulnerable. |

For CFG-aware LLM auditing, the `func` field may contain both code and CFG information:

```json
{
  "idx": "sample_0001",
  "func": "<CODE>\nint foo(...) { ... }\n</CODE>\n<CFG>\n...\n</CFG>",
  "target": 1
}
```

If CFG information is not available, the LLM auditing script can still run with code-only input by setting `--cfg_keep_nodes 0 --cfg_keep_edges 0`.

## Running the Pipeline

### Step 1: Train the Stage-1 Baseline

Train the GraphCodeBERT baseline and export prediction artifacts.

```bash
python scripts/stage1_baselines_unified.py \
  --models graphcodebert
```

To generate out-of-fold predictions for the training set, run:

```bash
python scripts/stage1_baselines_unified.py \
  --models graphcodebert \
  --make_oof_train \
  --oof_folds 5 \
  --oof_inner_valid_ratio 0.1
```

Expected Stage-1 outputs include:

```text
result/*_graphcodebert.pt
artifacts/*train_oof*graphcodebert*.npz
artifacts/*valid*graphcodebert*.npz
artifacts/*test*graphcodebert*.npz
```

The OOF artifact is used to select gray-zone training samples for LLM auditing.

### Step 2: Select Gray-Zone Samples and Run LLM Auditing

For the training set, use the Stage-1 OOF artifact:

```bash
python scripts/stage2_llm_with_cfg_api_features.py \
  --artefacts ./artifacts/reveal_baseline_artifacts_train_oof_graphcodebert.npz \
  --force_jsonl ./data/reveal_train.jsonl \
  --gray_on_pred neg_only \
  --gray_top_percent 3 \
  --code_clip 6000 \
  --cfg_keep_nodes 60 \
  --cfg_keep_edges 80 \
  --save_dir ./runs/stage2_train_negonly_top3
```

For code-only auditing without CFG input:

```bash
python scripts/stage2_llm_with_cfg_api_features.py \
  --artefacts ./artifacts/reveal_baseline_artifacts_train_oof_graphcodebert.npz \
  --force_jsonl ./data/reveal_train.jsonl \
  --gray_on_pred neg_only \
  --gray_top_percent 3 \
  --code_clip 6000 \
  --cfg_keep_nodes 0 \
  --cfg_keep_edges 0 \
  --save_dir ./runs/stage2_train_nocfg_negonly_top3
```

Expected Stage-2 outputs:

```text
runs/stage2_train_negonly_top3/gray_features_llm_with_cfg_api_*.jsonl
runs/stage2_train_negonly_top3/gray_features_llm_with_cfg_api_*.csv
runs/stage2_train_negonly_top3/run_meta_*.json
```

Each gray-zone record contains baseline predictions, LLM judgments, confidence scores, conservative gate information, code summary features, CFG summary features, and auxiliary attributes.

### Step 3: Build the Refine Training Dataset

Merge the raw training set, Stage-1 OOF predictions, and Stage-2 LLM audit features.

```bash
python scripts/build_refine_dataset.py \
  --train_json ./data/reveal_train.jsonl \
  --train_oof_npz ./artifacts/reveal_baseline_artifacts_train_oof_graphcodebert.npz \
  --stage2_train_gray_jsonl ./runs/stage2_train_negonly_top3/gray_features_llm_with_cfg_api_xxx.jsonl \
  --save_dir ./runs/refine_dataset_graphcodebert_top3 \
  --train_id_field idx \
  --gray_id_field func_id \
  --alpha_gold 0.85 \
  --min_soft_conf 0.60 \
  --min_aux_conf 0.60 \
  --gray_weight_agree 1.20 \
  --gray_weight_disagree 1.00 \
  --gray_weight_no_llm 1.00
```

Expected outputs:

```text
runs/refine_dataset_graphcodebert_top3/refine_train_*.jsonl
runs/refine_dataset_graphcodebert_top3/refine_train_*.csv
runs/refine_dataset_graphcodebert_top3/refine_targets_*.npz
runs/refine_dataset_graphcodebert_top3/refine_summary_*.json
```

The refined JSONL file keeps all original training samples. Non-gray samples use the original hard label. Gray samples may receive soft labels, sample weights, and auxiliary labels only when the LLM output passes the reliability constraints.

### Step 4: Train the Refined GraphCodeBERT Model

```bash
python scripts/train_refine_graphcodebert.py \
  --train_refine_json ./runs/refine_dataset_graphcodebert_top3/refine_train_xxx.jsonl \
  --valid_json ./data/reveal_valid.jsonl \
  --test_json ./data/reveal_test.jsonl \
  --stage1_ckpt ./result/stage1_graphcodebert.pt \
  --save_dir ./runs/refine_graphcodebert_top3 \
  --model_name ./models/graphcodebert_model \
  --preprocess_mode stage1_compatible \
  --max_len 512 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --epochs 15 \
  --encoder_lr 8e-6 \
  --head_lr 4e-5 \
  --aux_loss_weight 0.05 \
  --warmup_ratio 0.06 \
  --patience 10 \
  --max_pos_scale 20 \
  --seed 42
```

Expected outputs:

```text
runs/refine_graphcodebert_top3/*/best_refine_model.pt
runs/refine_graphcodebert_top3/*/train_history.csv
runs/refine_graphcodebert_top3/*/valid_predictions_best.csv
runs/refine_graphcodebert_top3/*/test_predictions_best.csv
runs/refine_graphcodebert_top3/*/refine_test_artifacts.npz
runs/refine_graphcodebert_top3/*/refine_summary.json
```

## Evaluation Metrics

The final script reports:

- Average Precision (AP)
- ROC-AUC
- F1 score
- Precision
- Recall
- ACC
- Best validation threshold
- P@K and R@K for top-ranked suspicious functions

Example final-test result reported in the manuscript:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| SeLLaVul | 0.82 | 0.65 | 0.73 |

Please update this table according to the exact dataset split, random seed, and checkpoint used in your released experiments.

## Key Design Choices

### Conservative LLM Use

The LLM is not used to replace the original labels directly. Its output is used only when it satisfies strict reliability conditions:

1. The output can be parsed successfully.
2. The confidence score is high enough.
3. The LLM prediction is consistent with the gold label.
4. The conservative gate does not suppress the judgment as speculative or context-insufficient.

### Soft-Label Construction

For reliable gray-zone samples, the refined target is constructed as:

```text
soft_label = alpha_gold * gold_label + (1 - alpha_gold) * llm_probability
```

If the LLM output is unreliable or inconsistent with the gold label, the original hard label is preserved.

### Auxiliary Supervision

The refined model includes one main vulnerability classification head and five auxiliary binary heads. Auxiliary loss is calculated only for samples with `aux_mask = 1`.

## Reproducibility Tips

- Keep sample IDs consistent across raw JSONL files, Stage-1 NPZ artifacts, and Stage-2 gray-zone outputs.
- Use `idx` for raw training data and `func_id` for Stage-2 gray-zone alignment unless your script has been modified.
- Fix the random seed for all training runs.
- Save all generated `refine_summary.json` files for later comparison.
- Do not mix Stage-1 artifacts from one dataset split with JSONL files from another split.

## Suggested `.gitignore`

```gitignore
__pycache__/
*.pyc
.env
*.pt
*.bin
*.npz
*.csv
*.jsonl
artifacts/*
result/*
runs/*
data/*
models/*
!artifacts/.gitkeep
!result/.gitkeep
!runs/.gitkeep
```

## Acknowledgement

This project builds on pretrained code representation learning and LLM-assisted software security analysis. The implementation uses PyTorch, Hugging Face Transformers, and common machine learning utilities from scikit-learn.
