# -*- coding: utf-8 -*-
"""
train_refine_graphcodebert.py

python train_refine_graphcodebert_project.py ^
    --train_refine_json ./refine_dataset_openssl_graphcodebert_top1/refine_train_2026-04-21_13-49-15.jsonl ^
    --valid_json ./openssl_val.jsonl ^
    --test_json ./openssl_test.jsonl ^
    --stage1_ckpt E:/XLNetLLM/SeLLaVul/Stage1_Baseline/result/openssl_2026-04-20_19-10-17_graphcodebert.pt ^
    --save_dir ./refine_openssl_runs_graphcodebert_top1 ^
    --model_name E:/XLNetLLM/SeLLaVul/Stage1_Baseline/graphcodebert_model ^
    --preprocess_mode stage1_compatible ^
    --max_len 512 ^
    --train_batch_size 16 ^
    --eval_batch_size 32 ^
    --epochs 15 ^
    --encoder_lr 8e-6 ^
    --head_lr 4e-5 ^
    --aux_loss_weight 0.05 ^
    --warmup_ratio 0.06 ^
    --patience 10 ^
    --max_pos_scale 20 ^
    --seed 42


python train_refine_graphcodebert.py ^
    --train_refine_json ./refine_dataset_graphcodebert/refine_train_2026-03-23_14-25-03.jsonl ^
    --valid_json ./valid.jsonl ^
    --test_json ./test.jsonl ^
    --stage1_ckpt E:/XLNetLLM/SeLLaVul/Stage1_Baseline/result/2026-03-22_13-57-38_graphcodebert.pt ^
    --disable_stage1_init ^
    --save_dir ./ablation_runs/B1_no_stage1_init ^
    --model_name E:/XLNetLLM/SeLLaVul/Stage1_Baseline/graphcodebert_model ^
    --preprocess_mode stage1_compatible ^
    --max_len 512 ^
    --train_batch_size 16 ^
    --eval_batch_size 32 ^
    --epochs 6 ^
    --encoder_lr 8e-6 ^
    --head_lr 4e-5 ^
    --aux_loss_weight 0.05 ^
    --warmup_ratio 0.06 ^
    --patience 6 ^
    --max_pos_scale 12 ^
    --seed 42
"""

from __future__ import absolute_import, division, print_function

import os
import re
import json
import math
import time
import random
import argparse
import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)


# =========================================================
# Utils
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_id(v):
    if v is None:
        return None
    return str(v)


def safe_int(v, default=None):
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def safe_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def safe_int01(v, default=None):
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return 1 if v else 0
        iv = int(float(str(v).strip()))
        return iv if iv in (0, 1) else default
    except Exception:
        return default


def load_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSONL at line {line_no} in {path}: {e}")
    return rows


def get_row_id(js: Dict[str, Any]) -> Optional[str]:
    for k in ["idx", "id", "uid", "func_id", "func_name"]:
        if k in js and js[k] is not None:
            return normalize_id(js[k])
    return None


def preprocess_code(code: str, mode: str = "stage1_compatible") -> str:
    if code is None:
        code = ""
    code = str(code)
    if mode == "raw":
        return code
    code = " ".join(code.split())
    if mode == "space_normalize":
        return code
    if mode == "stage1_compatible":
        pattern = (
            r"[A-Za-z_]\w*"
            r"|\d+"
            r"|==|!=|<=|>=|->|\+\+|--|\|\||&&|<<=|>>=|<<|>>"
            r"|\+=|-=|\*=|/=|%=|&=|\|=|\^=|::"
            r"|[{}()\[\];,.\+\-\*/%<>=!&|^~?:]"
        )
        toks = re.findall(pattern, code)
        return " ".join(toks) if toks else code
    raise ValueError(f"Unknown preprocess_mode: {mode}")


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    masked_hidden = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return masked_hidden.sum(dim=1) / denom


def compute_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float, float, float]:
    thresholds = np.linspace(0.0, 1.0, 2001)
    best_thr = 0.5
    best_f1 = -1.0
    best_p = 0.0
    best_r = 0.0
    for thr in thresholds:
        preds = (probs > thr).astype(int)
        if preds.sum() == 0:
            f1 = 0.0
            p = 0.0
            r = 0.0
        else:
            f1 = f1_score(y_true, preds, zero_division=0)
            p = precision_score(y_true, preds, zero_division=0)
            r = recall_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
            best_p = float(p)
            best_r = float(r)
    return best_thr, best_f1, best_p, best_r


def safe_average_precision(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, probs))
    except Exception:
        return float("nan")


def safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return float("nan")


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def compute_precision_recall_at_k(
    y_true: np.ndarray,
    probs: np.ndarray,
    k_list: List[int],
) -> Dict[str, float]:
    """
    计算 P@K 和 R@K 指标。

    Args:
        y_true: 真实标签 (0/1)
        probs: 预测概率
        k_list: K 值列表，例如 [1, 5, 10, 20, 50]

    Returns:
        包含 P@K 和 R@K 的字典，例如 {"P@1": 0.8, "R@1": 0.2, ...}
    """
    if len(y_true) == 0:
        return {f"P@{k}": float("nan") for k in k_list} | {f"R@{k}": float("nan") for k in k_list}

    # 按预测概率降序排序
    sorted_indices = np.argsort(-probs)
    y_sorted = y_true[sorted_indices]

    total_positives = int(np.sum(y_true))
    if total_positives == 0:
        # 没有正样本时，召回率定义为 0（或 nan）
        metrics = {}
        for k in k_list:
            metrics[f"P@{k}"] = 0.0
            metrics[f"R@{k}"] = 0.0
        return metrics

    metrics = {}
    for k in k_list:
        top_k = y_sorted[:k]
        tp_k = int(np.sum(top_k))
        precision_k = tp_k / k if k > 0 else 0.0
        recall_k = tp_k / total_positives
        metrics[f"P@{k}"] = float(precision_k)
        metrics[f"R@{k}"] = float(recall_k)

    return metrics


# =========================================================
# Dataset
# =========================================================

AUX_KEYS = [
    "aux_has_input_validation",
    "aux_has_bounds_check",
    "aux_uses_untrusted_input",
    "aux_has_null_check",
    "aux_has_error_handling_path",
]


class RefineCodeDataset(Dataset):
    def __init__(self, path: str, split_name: str, preprocess_mode: str = "stage1_compatible"):
        self.path = path
        self.split_name = split_name
        self.preprocess_mode = preprocess_mode
        self.rows = load_jsonl_rows(path)
        if len(self.rows) == 0:
            raise RuntimeError(f"Dataset is empty: {path}")

        self.samples = []
        seen_ids = set()
        for i, js in enumerate(self.rows):
            rid = get_row_id(js)
            if rid is None:
                raise RuntimeError(f"Missing id field in row #{i} of {path}")
            if rid in seen_ids:
                raise RuntimeError(f"Duplicate id in {path}: {rid}")
            seen_ids.add(rid)

            code = js.get("func", "")
            code = preprocess_code(code, preprocess_mode)

            hard_label = safe_int(js.get("hard_label", js.get("target")), default=None)
            if hard_label not in (0, 1):
                raise RuntimeError(f"Invalid hard label for id={rid} in {path}")

            soft_label = safe_float(js.get("soft_label", hard_label), default=float(hard_label))
            use_soft_label = safe_int01(js.get("use_soft_label", 0), default=0)
            sample_weight = safe_float(js.get("sample_weight", 1.0), default=1.0)
            aux_mask = safe_int01(js.get("aux_mask", 0), default=0)

            aux_targets = []
            for k in AUX_KEYS:
                aux_targets.append(safe_int01(js.get(k, -1), default=-1))

            self.samples.append({
                "idx": rid,
                "code": code,
                "hard_label": int(hard_label),
                "soft_label": float(soft_label),
                "use_soft_label": int(use_soft_label),
                "sample_weight": float(sample_weight),
                "aux_mask": int(aux_mask),
                "aux_targets": aux_targets,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class RefineCollator:
    def __init__(self, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        codes = [x["code"] for x in batch]
        enc = self.tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        out = {
            "idx": [x["idx"] for x in batch],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "hard_label": torch.tensor([x["hard_label"] for x in batch], dtype=torch.float32),
            "soft_label": torch.tensor([x["soft_label"] for x in batch], dtype=torch.float32),
            "use_soft_label": torch.tensor([x["use_soft_label"] for x in batch], dtype=torch.float32),
            "sample_weight": torch.tensor([x["sample_weight"] for x in batch], dtype=torch.float32),
            "aux_mask": torch.tensor([x["aux_mask"] for x in batch], dtype=torch.float32),
            "aux_targets": torch.tensor([x["aux_targets"] for x in batch], dtype=torch.float32),
        }
        return out


# =========================================================
# Model
# =========================================================

class GraphCodeBERTRefineModel(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1, num_aux_tasks: int = 5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        self.aux_classifier = nn.Linear(hidden_size, num_aux_tasks)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = masked_mean_pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        main_logits = self.classifier(pooled).squeeze(-1)
        aux_logits = self.aux_classifier(pooled)
        return {
            "main_logits": main_logits,
            "aux_logits": aux_logits,
            "pooled": pooled,
        }


# =========================================================
# Checkpoint helpers
# =========================================================

def extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise RuntimeError("Unable to extract state_dict from checkpoint.")


def remap_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("roberta."):
            nk = "encoder." + nk[len("roberta."):]
        if nk.startswith("backbone."):
            nk = "encoder." + nk[len("backbone."):]
        if nk.startswith("base."):
            nk = "encoder." + nk[len("base."):]
        new_sd[nk] = v
    return new_sd


def flexible_load_stage1_checkpoint(model: nn.Module, ckpt_path: str) -> Dict[str, Any]:
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = extract_state_dict(ckpt_obj)
    state_dict = remap_checkpoint_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    info = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    print(f"[CKPT] Loaded from: {ckpt_path}")
    print(f"[CKPT] Missing keys ({len(missing)}): {missing[:20]}")
    print(f"[CKPT] Unexpected keys ({len(unexpected)}): {unexpected[:20]}")
    return info


# =========================================================
# Loss
# =========================================================

def compute_train_losses(
    batch: Dict[str, Any],
    outputs: Dict[str, torch.Tensor],
    pos_scale: float,
    aux_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    main_logits = outputs["main_logits"]
    aux_logits = outputs["aux_logits"]

    hard_label = batch["hard_label"]
    soft_label = batch["soft_label"]
    use_soft_label = batch["use_soft_label"]
    sample_weight = batch["sample_weight"]
    aux_mask = batch["aux_mask"]
    aux_targets = batch["aux_targets"]

    y_main = torch.where(use_soft_label > 0.5, soft_label, hard_label)

    bce_main = nn.functional.binary_cross_entropy_with_logits(
        main_logits, y_main, reduction="none",
    )
    class_weight = torch.where(
        hard_label > 0.5,
        torch.full_like(hard_label, pos_scale),
        torch.ones_like(hard_label)
    )
    main_loss = (bce_main * sample_weight * class_weight).mean()

    if aux_mask.sum().item() > 0:
        aux_targets_clamped = aux_targets.clamp(min=0.0, max=1.0)
        bce_aux = nn.functional.binary_cross_entropy_with_logits(
            aux_logits, aux_targets_clamped, reduction="none",
        )
        aux_sample_mask = aux_mask.unsqueeze(1).expand_as(aux_targets_clamped)
        aux_loss = (bce_aux * aux_sample_mask).sum() / aux_sample_mask.sum().clamp(min=1.0)
    else:
        aux_loss = torch.zeros((), device=main_logits.device, dtype=main_logits.dtype)

    total_loss = main_loss + aux_loss_weight * aux_loss

    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "main_loss": float(main_loss.detach().cpu().item()),
        "aux_loss": float(aux_loss.detach().cpu().item()),
    }
    return total_loss, metrics


def compute_eval_loss(batch: Dict[str, Any], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    main_logits = outputs["main_logits"]
    hard_label = batch["hard_label"]
    return nn.functional.binary_cross_entropy_with_logits(main_logits, hard_label, reduction="mean")


# =========================================================
# Train / Eval
# =========================================================

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    pos_scale: float,
    aux_loss_weight: float,
    max_grad_norm: float,
    use_fp16: bool,
) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    main_sum = 0.0
    aux_sum = 0.0
    n_steps = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_fp16):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss, metrics = compute_train_losses(
                batch=batch,
                outputs=outputs,
                pos_scale=pos_scale,
                aux_loss_weight=aux_loss_weight,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_sum += metrics["loss"]
        main_sum += metrics["main_loss"]
        aux_sum += metrics["aux_loss"]
        n_steps += 1

    return {
        "train_loss": loss_sum / max(n_steps, 1),
        "train_main_loss": main_sum / max(n_steps, 1),
        "train_aux_loss": aux_sum / max(n_steps, 1),
    }


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> Dict[str, Any]:
    model.eval()
    all_ids = []
    all_labels = []
    all_probs = []
    loss_list = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=use_fp16):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = compute_eval_loss(batch, outputs)
        probs = torch.sigmoid(outputs["main_logits"]).detach().cpu().numpy()
        labels = batch["hard_label"].detach().cpu().numpy()
        all_ids.extend(batch["idx"])
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())
        loss_list.append(float(loss.detach().cpu().item()))

    return {
        "ids": all_ids,
        "labels": np.asarray(all_labels, dtype=np.int32),
        "probs": np.asarray(all_probs, dtype=np.float32),
        "loss": float(np.mean(loss_list)) if len(loss_list) > 0 else float("nan"),
    }


def evaluate_binary(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    ap = safe_average_precision(y_true, probs)
    auc = safe_auc(y_true, probs)
    best_thr, best_f1, best_p, best_r = compute_best_threshold(y_true, probs)
    return {
        "ap": ap,
        "auc": auc,
        "best_thr": best_thr,
        "best_f1": best_f1,
        "best_precision": best_p,
        "best_recall": best_r,
    }


def save_pred_csv(ids: List[str], probs: np.ndarray, labels: np.ndarray, thr: float, path: str, pred_col: str):
    preds = (probs > thr).astype(int)
    df = pd.DataFrame({
        "Func_id": ids,
        "prob": probs,
        "Label": labels.astype(int),
        pred_col: preds.astype(int),
    })
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[Saved] {path}")


def save_test_npz(ids: List[str], probs: np.ndarray, labels: np.ndarray, thr: float, path: str, meta: Dict[str, Any]):
    np.savez_compressed(
        path,
        pred_probs=np.asarray(probs, dtype=np.float32),
        test_labels=np.asarray(labels, dtype=np.int32),
        test_ids=np.asarray(ids, dtype=object),
        valid_thr=np.asarray(thr, dtype=np.float32),
        meta=np.asarray(meta, dtype=object),
    )
    print(f"[Saved] {path}")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_refine_json", required=True)
    parser.add_argument("--valid_json", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--save_dir", default="./refine_runs_graphcodebert")
    parser.add_argument("--model_name", default="microsoft/graphcodebert-base")
    parser.add_argument("--preprocess_mode", default="stage1_compatible",
                        choices=["raw", "space_normalize", "stage1_compatible"])
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--encoder_lr", type=float, default=8e-6)
    parser.add_argument("--head_lr", type=float, default=4e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aux_loss_weight", type=float, default=0.10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_pos_scale", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--disable_fp16", action="store_true")
    parser.add_argument("--disable_stage1_init", action="store_true",
                        help="do not load Stage-1 checkpoint initialization")
    parser.add_argument("--p_at_k", type=str, default="10,20,30,40,50,100,150,200",
                        help="Comma-separated list of K values for P@K and R@K metrics")

    args = parser.parse_args()

    # 解析 K 值列表
    p_at_k_list = [int(x.strip()) for x in args.p_at_k.split(",") if x.strip()]
    print(f"[P@K / R@K] Will compute for K = {p_at_k_list}")

    ensure_dir(args.save_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = (device.type == "cuda") and (not args.disable_fp16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.save_dir, f"refine_graphcodebert_{stamp}")
    ensure_dir(run_dir)

    print(f"[Device] {device}")
    print(f"[FP16] {use_fp16}")
    print(f"[RunDir] {run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = RefineCodeDataset(args.train_refine_json, "train_refine", args.preprocess_mode)
    valid_ds = RefineCodeDataset(args.valid_json, "valid", args.preprocess_mode)
    test_ds = RefineCodeDataset(args.test_json, "test", args.preprocess_mode)

    collator = RefineCollator(tokenizer, args.max_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collator, pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collator, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collator, pin_memory=(device.type == "cuda"),
    )

    # 更强一点的正类权重，优先抬 recall
    hard_labels = np.asarray([x["hard_label"] for x in train_ds.samples], dtype=np.int32)
    pos_n = int(hard_labels.sum())
    neg_n = int(len(hard_labels) - pos_n)
    imbalance = (neg_n + 1.0) / (pos_n + 1.0)
    raw_pos_scale = imbalance ** 0.65
    pos_scale = float(min(max(raw_pos_scale, 1.0), args.max_pos_scale))

    print(f"[Train] n={len(train_ds)} | pos={pos_n} | neg={neg_n} | pos_scale={pos_scale:.4f}")
    print(f"[Valid] n={len(valid_ds)}")
    print(f"[Test ] n={len(test_ds)}")

    model = GraphCodeBERTRefineModel(
        model_name=args.model_name,
        dropout=args.dropout,
        num_aux_tasks=len(AUX_KEYS),
    )

    if args.disable_stage1_init:
        ckpt_info = {
            "missing_keys": [],
            "unexpected_keys": [],
            "note": "Stage-1 initialization disabled"
        }
        print("[CKPT] Stage-1 initialization is disabled.")
    else:
        ckpt_info = flexible_load_stage1_checkpoint(model, args.stage1_ckpt)

    model.to(device)

    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay},
        ]
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_valid_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(run_dir, "best_refine_model.pt")
    history = []
    bad_rounds = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            pos_scale=pos_scale,
            aux_loss_weight=args.aux_loss_weight,
            max_grad_norm=args.max_grad_norm,
            use_fp16=use_fp16,
        )

        valid_pred = predict_dataset(
            model=model,
            loader=valid_loader,
            device=device,
            use_fp16=use_fp16,
        )
        valid_metrics = evaluate_binary(valid_pred["labels"], valid_pred["probs"])

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "train_main_loss": train_stats["train_main_loss"],
            "train_aux_loss": train_stats["train_aux_loss"],
            "valid_loss": valid_pred["loss"],
            "valid_ap": valid_metrics["ap"],
            "valid_auc": valid_metrics["auc"],
            "valid_best_thr": valid_metrics["best_thr"],
            "valid_best_f1": valid_metrics["best_f1"],
            "valid_best_precision": valid_metrics["best_precision"],
            "valid_best_recall": valid_metrics["best_recall"],
            "minutes": (time.time() - t0) / 60.0,
        }
        history.append(epoch_info)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={epoch_info['train_loss']:.6f} | "
            f"main={epoch_info['train_main_loss']:.6f} | "
            f"aux={epoch_info['train_aux_loss']:.6f} | "
            f"valid_loss={epoch_info['valid_loss']:.6f} | "
            f"valid_ap={epoch_info['valid_ap']:.6f} | "
            f"valid_auc={epoch_info['valid_auc']:.6f} | "
            f"valid_f1={epoch_info['valid_best_f1']:.6f} @ thr={epoch_info['valid_best_thr']:.4f}"
        )

        improved = epoch_info["valid_best_f1"] > best_valid_f1
        if improved:
            best_valid_f1 = epoch_info["valid_best_f1"]
            best_epoch = epoch
            bad_rounds = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_valid_f1": best_valid_f1,
                "args": vars(args),
                "stage1_ckpt_info": ckpt_info,
                "pos_scale": pos_scale,
            }, best_ckpt_path)
            print(f"[Saved] Best checkpoint -> {best_ckpt_path}")
        else:
            bad_rounds += 1
            if bad_rounds >= args.patience:
                print(f"[EarlyStop] no improvement for {args.patience} epoch(s).")
                break

    history_csv = os.path.join(run_dir, "train_history.csv")
    pd.DataFrame(history).to_csv(history_csv, index=False, encoding="utf-8")
    print(f"[Saved] {history_csv}")

    if best_epoch < 0:
        raise RuntimeError("No best checkpoint was saved.")

    best_obj = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(best_obj["model_state_dict"], strict=True)
    model.to(device)

    valid_pred = predict_dataset(model, valid_loader, device, use_fp16)
    valid_metrics = evaluate_binary(valid_pred["labels"], valid_pred["probs"])
    best_thr = valid_metrics["best_thr"]

    test_pred = predict_dataset(model, test_loader, device, use_fp16)
    test_probs = test_pred["probs"]
    test_labels = test_pred["labels"]
    test_preds = (test_probs > best_thr).astype(int)

    # 计算基础指标
    test_metrics = {
        "test_loss": test_pred["loss"],
        "test_ap": safe_average_precision(test_labels, test_probs),
        "test_auc": safe_auc(test_labels, test_probs),
        "test_f1": f1_score(test_labels, test_preds, zero_division=0),
        "test_precision": precision_score(test_labels, test_preds, zero_division=0),
        "test_recall": recall_score(test_labels, test_preds, zero_division=0),
        "test_thr": best_thr,
    }

    # 计算 P@K 和 R@K 指标
    pk_rk_metrics = compute_precision_recall_at_k(test_labels, test_probs, p_at_k_list)
    test_metrics.update(pk_rk_metrics)

    valid_csv = os.path.join(run_dir, "valid_predictions_best.csv")
    test_csv = os.path.join(run_dir, "test_predictions_best.csv")
    save_pred_csv(valid_pred["ids"], valid_pred["probs"], valid_pred["labels"], best_thr, valid_csv, "pred_refine")
    save_pred_csv(test_pred["ids"], test_pred["probs"], test_pred["labels"], best_thr, test_csv, "pred_refine")

    test_npz = os.path.join(run_dir, "refine_test_artifacts.npz")
    save_test_npz(
        ids=test_pred["ids"],
        probs=test_pred["probs"],
        labels=test_pred["labels"],
        thr=best_thr,
        path=test_npz,
        meta={
            "model_name": args.model_name,
            "stage1_ckpt": os.path.abspath(args.stage1_ckpt),
            "preprocess_mode": args.preprocess_mode,
            "best_epoch": best_epoch,
            "best_valid_f1": best_valid_f1,
            "valid_best_thr": best_thr,
            "task": "graphcodebert_refine",
        }
    )

    summary = {
        "config": vars(args),
        "device": str(device),
        "use_fp16": bool(use_fp16),
        "dataset": {
            "train_n": len(train_ds),
            "valid_n": len(valid_ds),
            "test_n": len(test_ds),
            "train_pos_n": pos_n,
            "train_neg_n": neg_n,
            "pos_scale": pos_scale,
        },
        "best_epoch": best_epoch,
        "best_valid_f1": best_valid_f1,
        "valid_metrics": {
            "valid_loss": valid_pred["loss"],
            "valid_ap": valid_metrics["ap"],
            "valid_auc": valid_metrics["auc"],
            "valid_best_thr": valid_metrics["best_thr"],
            "valid_best_f1": valid_metrics["best_f1"],
            "valid_best_precision": valid_metrics["best_precision"],
            "valid_best_recall": valid_metrics["best_recall"],
        },
        "test_metrics": test_metrics,
        "files": {
            "best_ckpt": os.path.abspath(best_ckpt_path),
            "history_csv": os.path.abspath(history_csv),
            "valid_csv": os.path.abspath(valid_csv),
            "test_csv": os.path.abspath(test_csv),
            "test_npz": os.path.abspath(test_npz),
        },
        "stage1_ckpt_info": ckpt_info,
    }

    summary_path = os.path.join(run_dir, "refine_summary.json")
    save_json(summary, summary_path)
    print(f"[Saved] {summary_path}")

    # 打印最终结果，包含 P@K 和 R@K
    print("\n=== Final Summary ===")
    final_print = {
        "best_epoch": best_epoch,
        "best_valid_f1": best_valid_f1,
        "valid_best_thr": best_thr,
        "test_ap": test_metrics["test_ap"],
        "test_auc": test_metrics["test_auc"],
        "test_f1": test_metrics["test_f1"],
        "test_precision": test_metrics["test_precision"],
        "test_recall": test_metrics["test_recall"],
    }
    for k in p_at_k_list:
        final_print[f"P@{k}"] = test_metrics.get(f"P@{k}", float("nan"))
        final_print[f"R@{k}"] = test_metrics.get(f"R@{k}", float("nan"))

    print(json.dumps(final_print, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
