# -*- coding: utf-8 -*-
"""

train OOF artifacts：
python stage1_baselines_unified1.py --models graphcodebert --make_oof_train --oof_folds 5 --oof_inner_valid_ratio 0.1

pip install torch transformers scikit-learn numpy pandas tqdm
"""

from __future__ import absolute_import, division, print_function

import os
import time
import json
import argparse
import datetime
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModel, AdamW
from transformers import T5EncoderModel

from tqdm import trange
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, balanced_accuracy_score, cohen_kappa_score
)
from sklearn.metrics import auc as sk_auc
from sklearn.model_selection import StratifiedKFold, train_test_split


# =========================
# GLOBAL CONFIG
# =========================
SEED = 43
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

THR_GRID = 2001
USE_SPLIT_CHARS = True

# 改动点：按 valid_f1 选 best checkpoint
SELECT_BEST_BY = "valid_f1"      # "valid_f1" or "valid_loss"
TIE_BREAK_BY = "valid_loss"      # tie-break: "valid_loss" or "valid_f1"

USE_AMP = True

DATA_DIR = "./"
TRAIN_JSON = "reveal_train.jsonl"
VALID_JSON = "reveal_valid.jsonl"
TEST_JSON = "reveal_test.jsonl"

RESULT_DIR = "./result"
ARTIFACTS_DIR = "./artifacts"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODELS_TO_RUN = ["graphcodebert"]

MODEL_SPECS = {
    "xlnet": {
        "name": "./xlnet_model",
        "family": "auto",
    },
    "codebert": {
        "name": "./codebert_model",
        "family": "auto",
    },
    "graphcodebert": {
        "name": "./graphcodebert_model",
        "family": "auto",
    },
    "codet5": {
        "name": "./codeT5_model",
        "family": "t5_encoder",
    },
    "roberta": {
        "name": "./roberta_model",
        "family": "auto",
    },
    "gpt2": {
        "name": "./gpt2_model",
        "family": "auto",
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Runtime
# =========================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def compute_pos_weight(labels_np: np.ndarray) -> float:
    labels_np = np.asarray(labels_np).astype(int)
    pos = int((labels_np == 1).sum())
    neg = int((labels_np == 0).sum())
    if pos <= 0:
        return 1.0
    return float(neg / max(pos, 1))


# =========================
# Data preprocessing
# =========================
def _fallback_split_characters(tok: str) -> str:
    if tok is None:
        return ""
    tok = str(tok)
    if tok == "":
        return ""

    tok = tok.replace("->", " -> ").replace(">>", " >> ").replace("<<", " << ")

    for t in ['(', ')', '{', '}', '*', '/', '+', '-', '=', ';', ',', '[', ']', '>', '<', '"']:
        tok = tok.replace(t, f" {t} ")
    return " ".join(tok.split())


try:
    from DataLoader import SplitCharacters as SplitCharacters  # noqa
    _SPLIT_IMPL = "DataLoader.SplitCharacters"
except Exception as e:
    SplitCharacters = _fallback_split_characters
    _SPLIT_IMPL = f"fallback_split_characters (reason: {type(e).__name__})"


def generate_id_label_code(json_file_path: str) -> Tuple[List[Any], List[int], List[str]]:
    id_list, label_list, code_list = [], [], []
    with open(json_file_path, encoding="utf-8") as f:
        for line in f:
            js = json.loads(line.strip())
            fid = js.get("idx", None)
            if fid is None:
                fid = js.get("func_name") or js.get("id") or js.get("uid")

            y = int(js["target"])
            func = js.get("func", "")
            tokens = func.split()

            if USE_SPLIT_CHARS:
                tokens = [SplitCharacters(tok) for tok in tokens]

            code = " ".join(tokens)

            id_list.append(fid)
            label_list.append(y)
            code_list.append(code)

    return id_list, label_list, code_list


def check_unique_ids(name, ids):
    from collections import Counter
    c = Counter(ids)
    dups = [k for k, v in c.items() if v > 1]
    if dups:
        print(f"[WARN] {name}: found {len(dups)} duplicated IDs (showing up to 10): {dups[:10]}")
    else:
        print(f"[OK] {name}: all IDs unique ({len(ids)} items).")


def ids_to_np(id_list):
    if all(isinstance(x, (int, np.integer)) for x in id_list):
        return np.asarray(id_list, dtype=np.int64)
    return np.asarray(id_list, dtype=object)


def tokenize_batch(text_list: List[str], tokenizer, max_len: int = MAX_LEN):
    enc = tokenizer(
        text_list,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return enc["input_ids"], enc["attention_mask"]


def build_tensor_dataset(text_list: List[str], labels_list: List[int], tokenizer):
    inputs, masks = tokenize_batch(text_list, tokenizer, MAX_LEN)
    labels = torch.tensor(labels_list, dtype=torch.float32)
    return TensorDataset(inputs, masks, labels)


# =========================
# Metrics / threshold search
# =========================
def find_best_threshold(y_true: np.ndarray, probs: np.ndarray, grid: int = THR_GRID) -> Tuple[float, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probs).astype(float)
    best_thr, best_f1 = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, grid):
        pred = (p > t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)
    return best_thr, best_f1


def evaluate_at_threshold(labels: np.ndarray, probs: np.ndarray, thr: float, title: str = "") -> Dict[str, Any]:
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs).flatten()
    pred = (p > thr).astype(int)

    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)

    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    bal_acc = balanced_accuracy_score(y, pred)
    kappa = cohen_kappa_score(y, pred)

    precision_arr, recall_arr, _ = precision_recall_curve(y, p)
    pr_auc = sk_auc(recall_arr, precision_arr) if len(recall_arr) > 1 else float("nan")

    if title:
        print(f"\n=== {title} ===")
    print("Confusion Matrix:\n", cm)
    print(f"TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")
    print(f"AUC:{auc:.6f}  AP:{ap:.6f}  PR-AUC:{pr_auc:.6f}")
    print(f"P:{prec:.6f}  R:{rec:.6f}  F1:{f1:.6f}  BalAcc:{bal_acc:.6f}  Kappa:{kappa:.6f}")
    print("\n", classification_report(y, pred, target_names=["Non-vulnerable", "Vulnerable"], digits=4))

    return {
        "thr": float(thr),
        "P": float(prec),
        "R": float(rec),
        "F1": float(f1),
        "AUC": float(auc),
        "AP": float(ap),
        "PR_AUC": float(pr_auc),
        "BalAcc": float(bal_acc),
        "Kappa": float(kappa),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# =========================
# Model
# =========================
class MeanPoolClassifier(nn.Module):
    def __init__(self, base_model, hidden_size: int, pos_weight: float = 1.0):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_normal_(self.classifier.weight)

        pw = torch.tensor([float(max(pos_weight, 1.0))], dtype=torch.float32)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            h = out.last_hidden_state
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            h = out[0]
        else:
            raise RuntimeError("Unexpected model output: cannot find hidden states.")

        if attention_mask is None:
            pooled = h.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1e-6)
            pooled = (h * mask).sum(dim=1) / denom

        logits = self.classifier(pooled).squeeze(-1)
        if labels is not None:
            loss = self.criterion(logits, labels.float())
            return loss, logits
        return logits


def load_tokenizer_and_model(model_key: str, pos_weight: float = 1.0):
    spec = MODEL_SPECS[model_key]
    name = spec["name"]
    family = spec.get("family", "auto")

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)

    added_new_pad = False
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
            added_new_pad = True

    if family == "t5_encoder":
        base = T5EncoderModel.from_pretrained(name)
        hidden = int(base.config.d_model)
    else:
        base = AutoModel.from_pretrained(name)
        hidden = int(getattr(base.config, "hidden_size", None) or getattr(base.config, "d_model", None))

    if added_new_pad and hasattr(base, "resize_token_embeddings"):
        base.resize_token_embeddings(len(tok))

    model = MeanPoolClassifier(base, hidden, pos_weight=pos_weight)
    model.to(device)

    if hasattr(model.base, "config"):
        if getattr(model.base.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
            model.base.config.pad_token_id = tok.pad_token_id
        if getattr(model.base.config, "eos_token_id", None) is None and tok.eos_token_id is not None:
            model.base.config.eos_token_id = tok.eos_token_id

    return tok, model


# =========================
# Training / inference
# =========================
@torch.no_grad()
def infer_probs(model: nn.Module, loader: DataLoader, use_amp: bool = True) -> np.ndarray:
    model.eval()
    probs = []
    amp_ok = (use_amp and device.type == "cuda" and USE_AMP)
    for batch in loader:
        input_ids, attn_mask, _labels = (t.to(device) for t in batch)
        if amp_ok:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(input_ids=input_ids, attention_mask=attn_mask)
        else:
            logits = model(input_ids=input_ids, attention_mask=attn_mask)
        p = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        probs.extend(p)
    return np.asarray(probs, dtype=np.float32)


def train_one_model(
    model_key: str,
    train_dataset: TensorDataset,
    valid_dataset: TensorDataset,
    train_labels_np: np.ndarray,
    valid_labels_np: np.ndarray,
    tokenizer,
    model: nn.Module,
    stamp: str,
) -> Dict[str, Any]:
    # 改动点：去掉 WeightedRandomSampler，改普通 shuffle
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=BATCH_SIZE)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and USE_AMP))

    best = {
        "epoch": -1,
        "score_f1": -1.0,
        "score_loss": float("inf"),
        "best_thr": 0.5,
        "ckpt_path": None,
    }

    t0 = time.time()
    for ep in trange(EPOCHS, desc=f"Epoch({model_key})"):
        model.train()
        tr_loss = 0.0
        n = 0

        for batch in train_loader:
            input_ids, attn_mask, labels = (t.to(device) for t in batch)
            optimizer.zero_grad(set_to_none=True)

            amp_ok = (device.type == "cuda" and USE_AMP)
            if amp_ok:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    loss, _ = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, _ = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

            bs = labels.size(0)
            tr_loss += loss.item() * bs
            n += bs

        train_loss = tr_loss / max(1, n)

        model.eval()
        ev_loss = 0.0
        nv = 0
        amp_ok = (device.type == "cuda" and USE_AMP)
        with torch.no_grad():
            for batch in valid_loader:
                input_ids, attn_mask, labels = (t.to(device) for t in batch)
                if amp_ok:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        loss, _ = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                else:
                    loss, _ = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                bs = labels.size(0)
                ev_loss += loss.item() * bs
                nv += bs
        valid_loss = ev_loss / max(1, nv)

        valid_probs = infer_probs(model, valid_loader, use_amp=True)
        best_thr, best_f1 = find_best_threshold(valid_labels_np, valid_probs, grid=THR_GRID)

        def is_better(curr_f1, curr_loss):
            if SELECT_BEST_BY == "valid_f1":
                if curr_f1 > best["score_f1"]:
                    return True
                if curr_f1 == best["score_f1"]:
                    return (curr_loss < best["score_loss"]) if TIE_BREAK_BY == "valid_loss" else False
                return False
            elif SELECT_BEST_BY == "valid_loss":
                if curr_loss < best["score_loss"]:
                    return True
                if curr_loss == best["score_loss"]:
                    return (curr_f1 > best["score_f1"]) if TIE_BREAK_BY == "valid_f1" else False
                return False
            else:
                raise ValueError("Unknown SELECT_BEST_BY")

        improved = is_better(best_f1, valid_loss)
        if improved:
            ckpt_path = os.path.join(RESULT_DIR, f"{stamp}_{model_key}.pt")
            torch.save(
                {
                    "model_key": model_key,
                    "model_name": MODEL_SPECS[model_key]["name"],
                    "epoch": ep,
                    "state_dict": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
                    "train_loss": float(train_loss),
                    "valid_loss": float(valid_loss),
                    "best_thr": float(best_thr),
                    "best_f1_valid": float(best_f1),
                    "config": {
                        "MAX_LEN": MAX_LEN,
                        "BATCH_SIZE": BATCH_SIZE,
                        "EPOCHS": EPOCHS,
                        "LR": LR,
                        "SEED": SEED,
                        "USE_SPLIT_CHARS": USE_SPLIT_CHARS,
                        "SELECT_BEST_BY": SELECT_BEST_BY,
                        "TIE_BREAK_BY": TIE_BREAK_BY,
                        "THR_GRID": THR_GRID,
                        "split_impl": _SPLIT_IMPL,
                    },
                },
                ckpt_path,
            )
            best.update(
                {
                    "epoch": ep,
                    "score_f1": float(best_f1),
                    "score_loss": float(valid_loss),
                    "best_thr": float(best_thr),
                    "ckpt_path": ckpt_path,
                }
            )

        print(
            f"[{model_key}] ep={ep:02d} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
            f"best_thr(valid)={best_thr:.4f} F1(valid@thr)={best_f1:.6f} "
            f"{'<< BEST' if improved else ''}"
        )

    elapsed = time.time() - t0
    best["elapsed_sec"] = float(elapsed)
    print(
        f"[{model_key}] Training done. Best epoch={best['epoch']}  "
        f"best F1(valid)={best['score_f1']:.6f} best_thr={best['best_thr']:.4f} time={elapsed:.1f}s"
    )
    return best


def load_best_checkpoint_into_model(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return ckpt


# =========================
# Save artifacts
# =========================
def save_artifacts_for_split(
    split_name: str,
    model_key: str,
    pred_probs: np.ndarray,
    labels_np: np.ndarray,
    id_list: List[Any],
    json_path: str,
    thr: float,
    stamp: str,
    best_ckpt_path: str,
    best_epoch: int,
):
    if split_name == "":
        arte_path = os.path.join(ARTIFACTS_DIR, f"baseline_artifacts_{model_key}_{stamp}.npz")
    else:
        arte_path = os.path.join(ARTIFACTS_DIR, f"baseline_artifacts_{split_name}_{model_key}_{stamp}.npz")

    meta = {
        "test_jsonl_path": os.path.abspath(json_path),
        "split_name": split_name if split_name != "" else "test",
        "model_key": model_key,
        "model_name_or_path": MODEL_SPECS[model_key]["name"],
        "best_ckpt": os.path.abspath(best_ckpt_path) if best_ckpt_path else None,
        "best_epoch": int(best_epoch),
        "max_len": int(MAX_LEN),
        "batch_size": int(BATCH_SIZE),
        "epochs": int(EPOCHS),
        "lr": float(LR),
        "seed": int(SEED),
        "use_split_chars": bool(USE_SPLIT_CHARS),
        "split_impl": str(_SPLIT_IMPL),
        "select_best_by": str(SELECT_BEST_BY),
        "tie_break_by": str(TIE_BREAK_BY),
        "thr_grid": int(THR_GRID),
        "time": stamp,
        "id_field": "idx",
    }

    np.savez_compressed(
        arte_path,
        pred_probs=np.asarray(pred_probs, dtype=np.float32),
        test_labels=np.asarray(labels_np, dtype=np.int32),
        test_ids=ids_to_np(id_list),
        valid_thr=np.float32(thr),
        meta=np.array([meta], dtype=object),
    )
    print(f"[Saved] {split_name or 'test'} artifacts -> {arte_path}")
    return arte_path


# =========================
# OOF helpers
# =========================
def safe_train_valid_split_indices(y: np.ndarray, valid_ratio: float, seed: int):
    idx = np.arange(len(y))
    try:
        tr_idx, va_idx = train_test_split(
            idx,
            test_size=valid_ratio,
            random_state=seed,
            stratify=y
        )
    except Exception:
        tr_idx, va_idx = train_test_split(
            idx,
            test_size=valid_ratio,
            random_state=seed,
            shuffle=True
        )
    return tr_idx, va_idx


def generate_oof_train_artifacts(
    model_key: str,
    train_ids: List[Any],
    train_y: List[int],
    train_code: List[str],
    full_best_thr: float,
    full_best_ckpt_path: str,
    full_best_epoch: int,
    train_json_path: str,
    stamp: str,
    oof_folds: int = 5,
    inner_valid_ratio: float = 0.1,
):
    """
    为 selector / refine dataset 生成 OOF(train) baseline artifacts：
    - 每个 train 样本的概率来自“没见过该样本”的子模型
    - 阈值仍沿用 full model 在 valid 上选出的 best_thr
    """
    y_np = np.asarray(train_y, dtype=np.int64)
    n = len(y_np)

    skf = StratifiedKFold(n_splits=oof_folds, shuffle=True, random_state=SEED)

    oof_probs = np.zeros(n, dtype=np.float32)
    filled = np.zeros(n, dtype=np.int32)

    print("\n" + "=" * 90)
    print(f"[OOF] Start generating OOF train artifacts for {model_key} | folds={oof_folds}")

    for fold, (dev_idx, hold_idx) in enumerate(skf.split(np.zeros(n), y_np), 1):
        print(f"\n[OOF] Fold {fold}/{oof_folds} | dev={len(dev_idx)} | holdout={len(hold_idx)}")

        dev_y = y_np[dev_idx]
        tr_rel, va_rel = safe_train_valid_split_indices(
            dev_y,
            valid_ratio=inner_valid_ratio,
            seed=SEED + fold
        )

        inner_train_idx = dev_idx[tr_rel]
        inner_valid_idx = dev_idx[va_rel]

        inner_train_code = [train_code[i] for i in inner_train_idx]
        inner_valid_code = [train_code[i] for i in inner_valid_idx]
        hold_code = [train_code[i] for i in hold_idx]

        inner_train_y = y_np[inner_train_idx]
        inner_valid_y = y_np[inner_valid_idx]
        hold_y = y_np[hold_idx]

        inner_pos_weight = compute_pos_weight(inner_train_y)
        print(f"[OOF] Fold {fold} pos_weight={inner_pos_weight:.6f}")

        tokenizer, model = load_tokenizer_and_model(model_key, pos_weight=inner_pos_weight)

        train_ds = build_tensor_dataset(inner_train_code, inner_train_y.tolist(), tokenizer)
        valid_ds = build_tensor_dataset(inner_valid_code, inner_valid_y.tolist(), tokenizer)
        hold_ds = build_tensor_dataset(hold_code, hold_y.tolist(), tokenizer)

        best_fold = train_one_model(
            model_key=model_key,
            train_dataset=train_ds,
            valid_dataset=valid_ds,
            train_labels_np=inner_train_y,
            valid_labels_np=inner_valid_y,
            tokenizer=tokenizer,
            model=model,
            stamp=f"{stamp}_oof_fold{fold}"
        )

        if best_fold["ckpt_path"] is None:
            raise RuntimeError(f"[OOF] No checkpoint saved for fold {fold}")

        _ = load_best_checkpoint_into_model(model, best_fold["ckpt_path"])

        hold_loader = DataLoader(
            hold_ds,
            sampler=SequentialSampler(hold_ds),
            batch_size=BATCH_SIZE
        )

        hold_probs = infer_probs(model, hold_loader, use_amp=True)

        oof_probs[hold_idx] = hold_probs
        filled[hold_idx] = 1

        del model
        torch.cuda.empty_cache()

    if not np.all(filled == 1):
        missing = np.where(filled == 0)[0]
        raise RuntimeError(f"[OOF] Some train samples were not filled by OOF predictions. Missing={len(missing)}")

    print(f"[OOF] Completed. Filled {filled.sum()}/{n} train samples.")

    oof_path = save_artifacts_for_split(
        split_name="train_oof",
        model_key=model_key,
        pred_probs=oof_probs,
        labels_np=y_np.astype(np.int32),
        id_list=train_ids,
        json_path=train_json_path,
        thr=full_best_thr,
        stamp=stamp,
        best_ckpt_path=full_best_ckpt_path,
        best_epoch=full_best_epoch,
    )

    return oof_path, oof_probs


# =========================
# Main
# =========================
def main():
    global device

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None, help="e.g. --models xlnet codebert graphcodebert codet5 roberta gpt2")
    parser.add_argument("--train_json", default=os.path.join(DATA_DIR, TRAIN_JSON))
    parser.add_argument("--valid_json", default=os.path.join(DATA_DIR, VALID_JSON))
    parser.add_argument("--test_json", default=os.path.join(DATA_DIR, TEST_JSON))
    parser.add_argument("--cuda", default=None, help="Optionally set CUDA_VISIBLE_DEVICES, e.g. 0")

    parser.add_argument("--make_oof_train", action="store_true", help="Generate OOF train artifacts for selector/refine training")
    parser.add_argument("--oof_folds", type=int, default=5, help="Number of folds for OOF train prediction")
    parser.add_argument("--oof_inner_valid_ratio", type=float, default=0.1, help="Inner valid ratio inside each OOF fold")

    args = parser.parse_args()

    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    models = args.models if args.models and len(args.models) > 0 else MODELS_TO_RUN
    for m in models:
        if m not in MODEL_SPECS:
            raise ValueError(f"Unknown model key: {m}. Available: {list(MODEL_SPECS.keys())}")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    train_ids, train_y, train_code = generate_id_label_code(args.train_json)
    valid_ids, valid_y, valid_code = generate_id_label_code(args.valid_json)
    test_ids, test_y, test_code = generate_id_label_code(args.test_json)

    check_unique_ids("TRAIN", train_ids)
    check_unique_ids("VALID", valid_ids)
    check_unique_ids("TEST ", test_ids)

    print(f"[Preprocess] USE_SPLIT_CHARS={USE_SPLIT_CHARS} | Split impl = {_SPLIT_IMPL}")
    print(f"Train: {len(train_y)} (pos={int(np.sum(train_y))})")
    print(f"Valid: {len(valid_y)} (pos={int(np.sum(valid_y))})")
    print(f"Test : {len(test_y)} (pos={int(np.sum(test_y))})")
    print(f"[Device] {device}")

    summary_rows = []

    for model_key in models:
        print("\n" + "=" * 90)
        print(f"[Stage1] Model = {model_key} | name/path = {MODEL_SPECS[model_key]['name']}")
        print(
            f"[Config] MAX_LEN={MAX_LEN} BATCH={BATCH_SIZE} EPOCHS={EPOCHS} LR={LR} "
            f"USE_SPLIT_CHARS={USE_SPLIT_CHARS} SELECT_BEST_BY={SELECT_BEST_BY} TIE_BREAK_BY={TIE_BREAK_BY}"
        )

        train_pos_weight = compute_pos_weight(np.asarray(train_y, dtype=np.int64))
        print(f"[Train] pos_weight={train_pos_weight:.6f}")

        tokenizer, model = load_tokenizer_and_model(model_key, pos_weight=train_pos_weight)

        train_inputs, train_masks = tokenize_batch(train_code, tokenizer, MAX_LEN)
        valid_inputs, valid_masks = tokenize_batch(valid_code, tokenizer, MAX_LEN)
        test_inputs, test_masks = tokenize_batch(test_code, tokenizer, MAX_LEN)

        train_labels = torch.tensor(train_y, dtype=torch.float32)
        valid_labels = torch.tensor(valid_y, dtype=torch.float32)
        test_labels = torch.tensor(test_y, dtype=torch.float32)

        train_ds = TensorDataset(train_inputs, train_masks, train_labels)
        valid_ds = TensorDataset(valid_inputs, valid_masks, valid_labels)
        test_ds = TensorDataset(test_inputs, test_masks, test_labels)

        train_eval_loader = DataLoader(train_ds, sampler=SequentialSampler(train_ds), batch_size=BATCH_SIZE)
        valid_loader = DataLoader(valid_ds, sampler=SequentialSampler(valid_ds), batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_ds, sampler=SequentialSampler(test_ds), batch_size=BATCH_SIZE)

        best = train_one_model(
            model_key=model_key,
            train_dataset=train_ds,
            valid_dataset=valid_ds,
            train_labels_np=np.asarray(train_y, dtype=np.int64),
            valid_labels_np=np.asarray(valid_y, dtype=np.int64),
            tokenizer=tokenizer,
            model=model,
            stamp=stamp,
        )

        if best["ckpt_path"] is None:
            raise RuntimeError(f"No checkpoint saved for {model_key}. Something went wrong.")

        ckpt = load_best_checkpoint_into_model(model, best["ckpt_path"])
        best_thr = float(ckpt["best_thr"])
        best_epoch = int(ckpt["epoch"])

        train_probs = infer_probs(model, train_eval_loader, use_amp=True)
        valid_probs = infer_probs(model, valid_loader, use_amp=True)
        test_probs = infer_probs(model, test_loader, use_amp=True)

        _ = evaluate_at_threshold(np.asarray(valid_y), valid_probs, best_thr, title=f"{model_key} | VALID (best_thr from valid)")
        test_metrics = evaluate_at_threshold(np.asarray(test_y), test_probs, best_thr, title=f"{model_key} | TEST (best_ckpt + best_thr)")

        _ = save_artifacts_for_split(
            split_name="train",
            model_key=model_key,
            pred_probs=train_probs,
            labels_np=np.asarray(train_y, dtype=np.int32),
            id_list=train_ids,
            json_path=args.train_json,
            thr=best_thr,
            stamp=stamp,
            best_ckpt_path=best["ckpt_path"],
            best_epoch=best_epoch,
        )
        _ = save_artifacts_for_split(
            split_name="valid",
            model_key=model_key,
            pred_probs=valid_probs,
            labels_np=np.asarray(valid_y, dtype=np.int32),
            id_list=valid_ids,
            json_path=args.valid_json,
            thr=best_thr,
            stamp=stamp,
            best_ckpt_path=best["ckpt_path"],
            best_epoch=best_epoch,
        )
        _ = save_artifacts_for_split(
            split_name="",
            model_key=model_key,
            pred_probs=test_probs,
            labels_np=np.asarray(test_y, dtype=np.int32),
            id_list=test_ids,
            json_path=args.test_json,
            thr=best_thr,
            stamp=stamp,
            best_ckpt_path=best["ckpt_path"],
            best_epoch=best_epoch,
        )

        if args.make_oof_train:
            _oof_path, _oof_probs = generate_oof_train_artifacts(
                model_key=model_key,
                train_ids=train_ids,
                train_y=train_y,
                train_code=train_code,
                full_best_thr=best_thr,
                full_best_ckpt_path=best["ckpt_path"],
                full_best_epoch=best_epoch,
                train_json_path=args.train_json,
                stamp=stamp,
                oof_folds=args.oof_folds,
                inner_valid_ratio=args.oof_inner_valid_ratio,
            )
            print(f"[Saved] train OOF artifacts -> {_oof_path}")

        df_out = pd.DataFrame({
            "Func_id": test_ids,
            "prob": test_probs,
            "Label": np.asarray(test_y, dtype=np.int32),
            "pred_baseline": (np.asarray(test_probs) > best_thr).astype(int),
        })
        csv_path = os.path.join(ARTIFACTS_DIR, f"baseline_only_{model_key}_{stamp}.csv")
        df_out.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[Saved] baseline csv -> {csv_path}")

        summary_rows.append({
            "model_key": model_key,
            "model_name_or_path": MODEL_SPECS[model_key]["name"],
            "best_epoch": best_epoch,
            "best_thr(valid)": best_thr,
            "F1_test": test_metrics["F1"],
            "P_test": test_metrics["P"],
            "R_test": test_metrics["R"],
            "AUC_test": test_metrics["AUC"],
            "AP_test": test_metrics["AP"],
            "elapsed_train_sec": best.get("elapsed_sec", float("nan")),
            "best_ckpt": best["ckpt_path"],
        })

        del model
        torch.cuda.empty_cache()

    df_sum = pd.DataFrame(summary_rows).sort_values(
        by=["F1_test", "R_test", "P_test"], ascending=False
    ).reset_index(drop=True)
    sum_path = os.path.join(RESULT_DIR, f"stage1_summary_{stamp}.csv")
    df_sum.to_csv(sum_path, index=False, encoding="utf-8")
    print("\n" + "=" * 90)
    print(f"[Saved] Stage-1 summary -> {sum_path}")
    if len(df_sum) > 0:
        top = df_sum.iloc[0].to_dict()
        print(f"[Best by F1_test] model={top['model_key']}  F1_test={top['F1_test']:.6f}  thr={top['best_thr(valid)']:.4f}")


if __name__ == "__main__":
    main()
