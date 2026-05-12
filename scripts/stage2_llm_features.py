# -*- coding: utf-8 -*-

python stage2_llm_with_features.py ^
  --artefacts ./artifacts/reveal_baseline_artifacts_train_oof_graphcodebert_2026-04-29_10-33-56.npz ^
  --force_jsonl ./reveal_train.jsonl ^
  --gray_on_pred neg_only ^
  --gray_top_percent 3 ^
  --code_clip 6000 ^
  --cfg_keep_nodes 60 ^
  --cfg_keep_edges 80 ^
  --save_dir ./stage2_reveal_train_with_cfg_api_negonly_top3

"""

from __future__ import absolute_import, division, print_function

import os
import re
import json
import time
import math
import argparse
import datetime
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import requests

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

# =========================================================
# API config
# =========================================================
API_BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
#API_MODEL_DEFAULT = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
API_MODEL_DEFAULT = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
API_KEY = "sk-vtkizllwdehrnzdwqpyg"

# =========================================================
# Prompt
# =========================================================
SYS_MSG = (
    "You are a senior C/C++ security auditor.\n"
    "Judge whether the function itself contains concrete local vulnerability evidence.\n"
    "Be conservative and precision-oriented.\n"
    "Do NOT speculate about missing struct layout, helper semantics, callee ownership, "
    "macro expansion details, or cross-function behavior.\n"
    "If evidence is insufficient, output final_label=0 with lower confidence.\n\n"
    "Return EXACTLY ONE JSON object only. No markdown. No code fence. No extra text.\n"
    "Use this schema exactly:\n"
    "{\n"
    "  \"final_label\": 0 or 1,\n"
    "  \"confidence\": float in [0,1],\n"
    "  \"evidence_sufficient\": 0 or 1,\n"
    "  \"needs_more_context\": 0 or 1,\n"
    "  \"context_missing_type\": \"none|struct_def|callee_semantics|type_layout|macro_expansion|global_invariant|cross_function_flow|other\",\n"
    "  \"confirmed_unsafe_write\": 0 or 1,\n"
    "  \"confirmed_unsafe_free\": 0 or 1,\n"
    "  \"confirmed_source_to_sink\": 0 or 1,\n"
    "  \"safe_parse_pattern\": 0 or 1,\n"
    "  \"speculative_only\": 0 or 1,\n"
    "  \"has_input_validation\": 0 or 1,\n"
    "  \"has_bounds_check\": 0 or 1,\n"
    "  \"uses_untrusted_input\": 0 or 1,\n"
    "  \"has_null_check\": 0 or 1,\n"
    "  \"has_error_handling_path\": 0 or 1,\n"
    "  \"vuln_type\": \"short string\",\n"
    "  \"rationale\": \"short concise rationale\"\n"
    "}\n"
)

# =========================================================
# Pretty code
# =========================================================
def restore_multiline(code: str) -> str:
    if not code:
        return code
    s = code.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r';\s*', ';\n', s)
    s = re.sub(r'\s*\{', r'\n{\n', s)
    s = re.sub(r'\}\s*', r'}\n', s)
    s = re.sub(r'\b(if|for|while|switch|case|return)\b', r'\n\1', s)
    s = re.sub(r'\n{3,}', '\n\n', s).strip()
    return s

# =========================================================
# JSON extraction
# =========================================================
def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m:
        return None
    s = m.group(0)
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r',\s*]', ']', s)
    try:
        return json.loads(s)
    except Exception:
        return None

# =========================================================
# Parse <CODE> and <CFG>
# =========================================================
CODE_RE = re.compile(r'<CODE>\s*(.*?)\s*</CODE>', re.S | re.I)
CFG_RE = re.compile(r'<CFG>\s*(.*?)\s*</CFG>', re.S | re.I)

def split_code_cfg(func_field: str) -> Tuple[str, Optional[str]]:
    if not func_field:
        return "", None
    code_text = func_field
    m_code = CODE_RE.search(func_field)
    if m_code:
        code_text = m_code.group(1).strip()
    cfg_text = None
    m_cfg = CFG_RE.search(func_field)
    if m_cfg:
        cfg_text = m_cfg.group(1).strip()
    return code_text, cfg_text

# =========================================================
# Summary feature extractors
# =========================================================
DANGEROUS_APIS = [
    "strcpy", "strcat", "gets", "sprintf", "vsprintf",
    "memcpy", "memmove", "scanf", "fscanf", "sscanf"
]

def extract_code_summary(code: str) -> Dict[str, Any]:
    if not code:
        return {
            "token_count": 0,
            "line_count": 0,
            "branch_count": 0,
            "return_count": 0,
            "pointer_count": 0,
            "array_access_count": 0,
            "dangerous_api_count": 0,
        }

    return {
        "token_count": int(len(code.split())),
        "line_count": int(len([ln for ln in code.splitlines() if ln.strip()])),
        "branch_count": int(len(re.findall(r"\b(if|else\s+if|switch|case|for|while)\b", code))),
        "return_count": int(len(re.findall(r"\breturn\b", code))),
        "pointer_count": int(code.count("*")),
        "array_access_count": int(code.count("[")),
        "dangerous_api_count": int(sum(code.count(api) for api in DANGEROUS_APIS)),
    }


def extract_cfg_summary(cfg_text: Optional[str]) -> Dict[str, Any]:
    if not cfg_text:
        return {
            "cfg_is_real": 0,
            "cfg_is_global": 0,
            "cfg_node_count": 0,
            "cfg_edge_count": 0,
            "has_validation_path_cfg": 0,
            "has_error_handling_path_cfg": 0,
            "has_null_check_cfg": 0,
        }

    cfg_upper = cfg_text.upper()
    cfg_lower = cfg_text.lower()

    cfg_is_real = 1 if "TYPE: REAL" in cfg_upper else 0
    cfg_is_global = 1 if "TYPE: GLOBAL" in cfg_upper else 0

    cfg_node_count = len(re.findall(r"\bN\d+:", cfg_text))

    cfg_edge_count = 0
    m = re.search(r"Edges:\s*(.*)", cfg_text, flags=re.S)
    if m:
        edge_items = [x.strip() for x in m.group(1).split(";") if x.strip()]
        cfg_edge_count = len(edge_items)

    has_validation_path_cfg = int(
        ("check" in cfg_lower) or
        ("validate" in cfg_lower) or
        ("assert" in cfg_lower) or
        ("bounds" in cfg_lower) or
        ("len" in cfg_lower) or
        ("size" in cfg_lower)
    )

    has_error_handling_path_cfg = int(
        ("error" in cfg_lower) or
        ("fail" in cfg_lower) or
        ("cleanup" in cfg_lower) or
        ("return" in cfg_lower)
    )

    has_null_check_cfg = int(
        ("null" in cfg_lower) or
        ("nil" in cfg_lower)
    )

    return {
        "cfg_is_real": int(cfg_is_real),
        "cfg_is_global": int(cfg_is_global),
        "cfg_node_count": int(cfg_node_count),
        "cfg_edge_count": int(cfg_edge_count),
        "has_validation_path_cfg": int(has_validation_path_cfg),
        "has_error_handling_path_cfg": int(has_error_handling_path_cfg),
        "has_null_check_cfg": int(has_null_check_cfg),
    }

# =========================================================
# Helpers
# =========================================================
def _safe_int01(v, default=None):
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return 1 if v else 0
        s = str(v).strip().lower()
        if s in {"0", "1"}:
            return int(s)
        if s in {"true", "yes"}:
            return 1
        if s in {"false", "no"}:
            return 0
        iv = int(float(s))
        return iv if iv in (0, 1) else default
    except Exception:
        return default


def _safe_float(v, default=None):
    try:
        if v is None:
            return default
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return default
        return fv
    except Exception:
        return default


def to_logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def parse_label_strict(js: Dict[str, Any]) -> Optional[int]:
    return _safe_int01(js.get("final_label"), default=None)


def parse_confidence_strict(js: Dict[str, Any]) -> Optional[float]:
    c = _safe_float(js.get("confidence"), default=None)
    if c is None:
        return None
    return min(1.0, max(0.0, c))


def normalize_context_missing_type(x: Any) -> str:
    allowed = {
        "none", "struct_def", "callee_semantics", "type_layout",
        "macro_expansion", "global_invariant", "cross_function_flow", "other"
    }
    if x is None:
        return "none"
    s = str(x).strip().lower()
    return s if s in allowed else "other"


def lower_text(x: Any) -> str:
    return "" if x is None else str(x).strip().lower()

# =========================================================
# Heuristics for gate
# =========================================================
def looks_like_logging_only(code: str) -> bool:
    code_l = lower_text(code)
    if not code_l:
        return False

    logging_hits = sum([
        code_l.count("printk("),
        code_l.count("gprintk("),
        code_l.count("msg_dbg("),
        code_l.count("msg_err("),
        code_l.count("printf("),
        code_l.count("fprintf("),
        code_l.count("pr_info("),
        code_l.count("pr_err("),
        code_l.count("pr_warn("),
    ])
    if logging_hits <= 0:
        return False

    risky_ops = [
        "memcpy(", "memmove(", "strcpy(", "strcat(", "gets(",
        "malloc(", "calloc(", "realloc(", "free(",
        "copy_from_user", "copy_to_user", "write(", "read("
    ]
    risky_present = any(op in code_l for op in risky_ops)
    if risky_present:
        return False

    return True


def helper_semantics_unknown(code: str, vuln_type: str, rationale: str) -> bool:
    vt = lower_text(vuln_type)
    ra = lower_text(rationale)
    code_l = lower_text(code)

    target_vt = (
        "double free" in vt or
        "invalid free" in vt or
        "use-after-free" in vt or
        "uaf" in vt
    )
    if not target_vt:
        return False

    has_custom_free = re.search(r'\b[a-zA-Z_][a-zA-Z0-9_]*free[a-zA-Z0-9_]*\s*\(', code_l) is not None
    has_plain_free = "free(" in code_l

    explicit_ownership = any(x in ra for x in [
        "ownership explicit", "ownership clear", "same pointer definitely freed",
        "callee definitely frees", "confirmed double free"
    ])

    return has_custom_free and has_plain_free and (not explicit_ownership)


def safe_parse_suppress_heuristic(rec: Dict[str, Any], code: str) -> bool:
    code_l = lower_text(code)
    vt = lower_text(rec.get("vuln_type", ""))

    parser_keywords = [
        "read", "load", "parse", "header", "chunk", "cookie", "demux",
        "peek", "stream_", "copy_from_guest", "extent", "decode"
    ]
    parser_like = any(k in code_l for k in parser_keywords)

    has_iv = int(rec.get("has_input_validation", 0) or 0) == 1
    has_bc = int(rec.get("has_bounds_check", 0) or 0) == 1
    has_eh = int(rec.get("has_error_handling_path", 0) or 0) == 1
    has_unsafe_write = int(rec.get("confirmed_unsafe_write", 0) or 0) == 1
    has_src_sink = int(rec.get("confirmed_source_to_sink", 0) or 0) == 1
    safe_parse = int(rec.get("safe_parse_pattern", 0) or 0) == 1

    overflow_like = (
        "overflow" in vt or
        "out-of-bounds" in vt or
        "oob" in vt or
        "buffer" in vt or
        vt == "none" or
        vt == ""
    )

    return (
        overflow_like and
        (safe_parse or parser_like) and
        has_iv and has_bc and has_eh and
        (not has_unsafe_write) and
        (not has_src_sink)
    )


def overflow_needs_layout_context(rec: Dict[str, Any]) -> bool:
    vt = lower_text(rec.get("vuln_type", ""))
    cmt = normalize_context_missing_type(rec.get("context_missing_type"))
    has_unsafe_write = int(rec.get("confirmed_unsafe_write", 0) or 0) == 1

    overflow_like = (
        "overflow" in vt or
        "out-of-bounds" in vt or
        "oob" in vt
    )
    return overflow_like and ((not has_unsafe_write) or cmt in {"struct_def", "type_layout"})

# =========================================================
# Conservative gate
# =========================================================
def apply_conservative_gate(rec: Dict[str, Any], code: str) -> Dict[str, Any]:
    raw_label = rec.get("llm_final_label_raw", None)
    raw_conf = rec.get("llm_confidence_raw", None)

    if raw_label is None:
        rec["llm_final_label"] = None
        rec["llm_confidence"] = None
        rec["gate_applied"] = False
        rec["gate_reason"] = ""
        rec["evidence_gate_triggered"] = 0
        rec["context_gate_triggered"] = 0
        rec["safe_parse_suppress_triggered"] = 0
        rec["helper_semantics_suppress_triggered"] = 0
        rec["logging_only_suppress_triggered"] = 0
        rec["overflow_context_suppress_triggered"] = 0
        return rec

    adj_label = int(raw_label)
    adj_conf = 0.5 if raw_conf is None else float(raw_conf)
    gate_reasons: List[str] = []

    evidence_gate_triggered = 0
    context_gate_triggered = 0
    safe_parse_suppress_triggered = 0
    helper_semantics_suppress_triggered = 0
    logging_only_suppress_triggered = 0
    overflow_context_suppress_triggered = 0

    evidence_sufficient = _safe_int01(rec.get("evidence_sufficient"), default=None)
    needs_more_context = _safe_int01(rec.get("needs_more_context"), default=None)
    speculative_only = _safe_int01(rec.get("speculative_only"), default=0)
    confirmed_unsafe_write = _safe_int01(rec.get("confirmed_unsafe_write"), default=0)
    confirmed_unsafe_free = _safe_int01(rec.get("confirmed_unsafe_free"), default=0)

    if adj_label == 1:
        if evidence_sufficient is None:
            evidence_sufficient = 0

        if evidence_sufficient == 0:
            adj_label = 0
            adj_conf = min(adj_conf, 0.55)
            evidence_gate_triggered = 1
            gate_reasons.append("evidence_insufficient")

        if needs_more_context == 1:
            adj_label = 0
            adj_conf = min(adj_conf, 0.55)
            context_gate_triggered = 1
            gate_reasons.append(f"context_missing:{normalize_context_missing_type(rec.get('context_missing_type'))}")

        if speculative_only == 1:
            adj_label = 0
            adj_conf = min(adj_conf, 0.55)
            evidence_gate_triggered = 1
            gate_reasons.append("speculative_only")

        if helper_semantics_unknown(code, rec.get("vuln_type", ""), rec.get("rationale", "")):
            if confirmed_unsafe_free != 1:
                adj_label = 0
                adj_conf = min(adj_conf, 0.55)
                helper_semantics_suppress_triggered = 1
                gate_reasons.append("helper_semantics_unknown")

        if overflow_needs_layout_context(rec):
            adj_label = 0
            adj_conf = min(adj_conf, 0.55)
            overflow_context_suppress_triggered = 1
            gate_reasons.append("overflow_needs_layout_context")

        if safe_parse_suppress_heuristic(rec, code):
            adj_label = 0
            adj_conf = min(adj_conf, 0.55)
            safe_parse_suppress_triggered = 1
            gate_reasons.append("safe_parse_suppress")

        if looks_like_logging_only(code):
            if _safe_int01(rec.get("confirmed_source_to_sink"), default=0) != 1 and confirmed_unsafe_write != 1:
                adj_label = 0
                adj_conf = min(adj_conf, 0.55)
                logging_only_suppress_triggered = 1
                gate_reasons.append("logging_only_suppress")

    else:
        if needs_more_context == 1:
            adj_conf = min(adj_conf, 0.60)

    rec["llm_final_label"] = int(adj_label)
    rec["llm_confidence"] = float(adj_conf)

    rec["gate_applied"] = len(gate_reasons) > 0
    rec["gate_reason"] = "|".join(gate_reasons)
    rec["evidence_gate_triggered"] = int(evidence_gate_triggered)
    rec["context_gate_triggered"] = int(context_gate_triggered)
    rec["safe_parse_suppress_triggered"] = int(safe_parse_suppress_triggered)
    rec["helper_semantics_suppress_triggered"] = int(helper_semantics_suppress_triggered)
    rec["logging_only_suppress_triggered"] = int(logging_only_suppress_triggered)
    rec["overflow_context_suppress_triggered"] = int(overflow_context_suppress_triggered)

    return rec

# =========================================================
# User prompt assembly
# =========================================================
def assemble_user_block(code: str,
                        cfg_text: Optional[str],
                        cfg_keep_nodes: int = 60,
                        cfg_keep_edges: int = 80,
                        code_clip: int = 6000) -> Tuple[str, str, str]:
    code_used_inner = code if code_clip is None or code_clip <= 0 else code[:code_clip]
    code_used = "<CODE>\n" + code_used_inner + "\n</CODE>\n"

    cfg_used = ""
    if cfg_text and "TYPE: REAL" in cfg_text.upper():
        lines_all = [ln for ln in cfg_text.splitlines()]
        type_lines = [ln for ln in lines_all if ln.strip().upper().startswith("TYPE")]
        node_lines = [ln for ln in lines_all if ln.strip().startswith("N")]
        edge_lines = [ln for ln in lines_all if ln.strip().startswith("Edges:")]

        node_keep = node_lines if cfg_keep_nodes is None or cfg_keep_nodes <= 0 else node_lines[:cfg_keep_nodes]
        edge_keep = []
        if edge_lines:
            m = re.match(r"^Edges:\s*(.*)$", edge_lines[0].strip())
            if m:
                items = [x.strip() for x in m.group(1).split(";") if x.strip()]
                edge_keep = items if cfg_keep_edges is None or cfg_keep_edges <= 0 else items[:cfg_keep_edges]

        out_lines = []
        if type_lines:
            out_lines.extend(type_lines[:1])
        out_lines.extend(node_keep)
        edges_str = "Edges: " + "; ".join(edge_keep) if edge_keep else "Edges: "
        out_lines.append(edges_str)
        cfg_used_inner = "\n".join(out_lines)
        cfg_used = "<CFG>\n" + cfg_used_inner + "\n</CFG>\n"

    user = "Code snippet:\n" + code_used
    if cfg_used:
        user += "CFG summary:\n" + cfg_used
    user += (
        "\nAnalyze the function itself only.\n"
        "Be conservative and evidence-based.\n"
        "If helper semantics / struct layout / cross-function flow is unknown, do not speculate.\n"
        "Output ONLY one valid JSON object."
    )
    return user, code_used, cfg_used

# =========================================================
# API call
# =========================================================
def call_llm_chat(user_content: str,
                  model_name: str,
                  extra_hint: str = "",
                  max_tokens: int = 256,
                  temperature: float = 0.0,
                  timeout: int = 240,
                  retry: int = 3,
                  sleep_sec: float = 2.0) -> str:
    if not API_KEY:
        raise RuntimeError("Missing SILICONFLOW_API_KEY environment variable")

    final_user = user_content + ("\n" + extra_hint if extra_hint else "")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYS_MSG},
            {"role": "user", "content": final_user}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    last_err = None
    for _ in range(retry):
        try:
            resp = requests.post(
                API_BASE_URL,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)

    raise RuntimeError(f"LLM call failed after retries: {last_err}")

# =========================================================
# Metrics
# =========================================================
def evaluate_baseline(labels_tensor, probs, thr=0.5):
    y = labels_tensor if isinstance(labels_tensor, np.ndarray) else np.asarray(labels_tensor)
    p = np.asarray(probs).flatten()
    preds = (p > thr).astype(int)
    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    print("Confusion Matrix:\n", cm)
    print(f"AUC:{auc:.4f}  F1:{f1:.4f}  P:{prec:.4f}  R:{rec:.4f}  AP:{ap:.4f}")
    print("\n", classification_report(y, preds, target_names=["Non-vulnerable", "Vulnerable"]))
    return dict(AUC=auc, F1=f1, P=prec, R=rec, AP=ap, CM=cm, TN=tn, FP=fp, FN=fn, TP=tp)

# =========================================================
# Gray selector
# =========================================================
def select_gray_indices(
    pred_probs: np.ndarray,
    base_preds: np.ndarray,
    thr: float,
    gray_on_pred: str = "all",
    gray_top_percent: float = None,
    delta_logit: float = 1.5,
    delta: float = None,
) -> np.ndarray:
    pred_probs = np.asarray(pred_probs, dtype=float).flatten()
    base_preds = np.asarray(base_preds, dtype=int).flatten()

    logits = to_logit(pred_probs)
    thr_logit = float(to_logit([thr])[0])

    if gray_on_pred == "neg_only":
        candidate = np.where(base_preds == 0)[0]
    elif gray_on_pred == "all":
        candidate = np.arange(len(pred_probs))
    else:
        raise ValueError(f"Unknown gray_on_pred: {gray_on_pred}")

    if len(candidate) == 0:
        return np.array([], dtype=int)

    cand_probs = pred_probs[candidate]
    cand_logits = logits[candidate]

    if gray_top_percent is not None:
        assert 0 < gray_top_percent <= 100, "gray_top_percent must be in (0,100]"
        d = np.abs(cand_logits - thr_logit)
        k = max(1, int(len(d) * (gray_top_percent / 100.0)))
        picked_local = np.argsort(d)[:k]
        gray = candidate[picked_local]
        print(f"[Gray] mode={gray_on_pred}, top {gray_top_percent}% within candidate pool -> {len(gray)} samples")
        return np.asarray(gray, dtype=int)

    if delta_logit is not None:
        picked_local = np.where(np.abs(cand_logits - thr_logit) <= delta_logit)[0]
        gray = candidate[picked_local]
        print(f"[Gray] mode={gray_on_pred}, delta_logit={delta_logit} -> {len(gray)} samples")
        return np.asarray(gray, dtype=int)

    if delta is not None:
        picked_local = np.where(np.abs(cand_probs - thr) <= delta)[0]
        gray = candidate[picked_local]
        print(f"[Gray] mode={gray_on_pred}, delta={delta} -> {len(gray)} samples")
        return np.asarray(gray, dtype=int)

    picked_local = np.where(np.abs(cand_logits - thr_logit) <= 1.5)[0]
    gray = candidate[picked_local]
    print(f"[Gray] mode={gray_on_pred}, default delta_logit=1.5 -> {len(gray)} samples")
    return np.asarray(gray, dtype=int)

# =========================================================
# Data loading with strict alignment
# =========================================================
def load_code_cfg_aligned(jsonl_path: str,
                          test_ids: np.ndarray,
                          pretty: bool = True,
                          id_field: Optional[str] = None) -> Tuple[List[str], List[Optional[str]]]:
    print(f"[Load Codes+CFG] from: {jsonl_path}")
    id2pair: Dict[str, Tuple[str, Optional[str]]] = {}

    def get_id(js: Dict[str, Any]) -> Optional[str]:
        if id_field and id_field in js:
            return js[id_field]
        return (js.get("func_id")
                or js.get("id")
                or js.get("uid")
                or js.get("Func_id")
                or js.get("idx"))

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            js = json.loads(line)
            raw = js.get('func') or js.get('code') or ""
            code, cfg = split_code_cfg(raw)
            if pretty:
                code = restore_multiline(code)
            fid = get_id(js)
            if fid is None:
                raise RuntimeError("Missing id/idx in CFG jsonl record")
            id2pair[str(fid)] = (code, cfg)

    keys = set(id2pair.keys())
    if not (len(test_ids) == len(id2pair) and all(str(fid) in keys for fid in test_ids)):
        raise RuntimeError(
            f"ID alignment failed between artifacts and CFG jsonl: "
            f"test_ids={len(test_ids)}, jsonl_ids={len(id2pair)}"
        )

    print(f"[Align] strict ID mapping matched: {len(id2pair)} items.")
    codes = [id2pair[str(fid)][0] for fid in test_ids]
    cfgs = [id2pair[str(fid)][1] for fid in test_ids]
    return codes, cfgs

# =========================================================
# Parse structured fields from model json
# =========================================================
def fill_structured_fields_from_model(js: Dict[str, Any], rec: Dict[str, Any]) -> Dict[str, Any]:
    rec["llm_final_label_raw"] = parse_label_strict(js)
    rec["llm_confidence_raw"] = parse_confidence_strict(js)

    rec["evidence_sufficient"] = _safe_int01(js.get("evidence_sufficient"), default=None)
    rec["needs_more_context"] = _safe_int01(js.get("needs_more_context"), default=None)
    rec["context_missing_type"] = normalize_context_missing_type(js.get("context_missing_type"))

    rec["confirmed_unsafe_write"] = _safe_int01(js.get("confirmed_unsafe_write"), default=0)
    rec["confirmed_unsafe_free"] = _safe_int01(js.get("confirmed_unsafe_free"), default=0)
    rec["confirmed_source_to_sink"] = _safe_int01(js.get("confirmed_source_to_sink"), default=0)

    rec["safe_parse_pattern"] = _safe_int01(js.get("safe_parse_pattern"), default=0)
    rec["speculative_only"] = _safe_int01(js.get("speculative_only"), default=0)

    rec["has_input_validation"] = _safe_int01(js.get("has_input_validation"), default=0)
    rec["has_bounds_check"] = _safe_int01(js.get("has_bounds_check"), default=0)
    rec["uses_untrusted_input"] = _safe_int01(js.get("uses_untrusted_input"), default=0)
    rec["has_null_check"] = _safe_int01(js.get("has_null_check"), default=0)
    rec["has_error_handling_path"] = _safe_int01(js.get("has_error_handling_path"), default=0)

    rec["vuln_type"] = str(js.get("vuln_type", "")).strip()[:200]
    rec["rationale"] = str(js.get("rationale", "")).strip()[:2000]

    if rec["llm_final_label_raw"] == 1:
        if rec["evidence_sufficient"] is None:
            rec["evidence_sufficient"] = 0
        if rec["needs_more_context"] is None:
            rec["needs_more_context"] = 0
    else:
        if rec["evidence_sufficient"] is None:
            rec["evidence_sufficient"] = 1
        if rec["needs_more_context"] is None:
            rec["needs_more_context"] = 0

    return rec

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artefacts", required=True, help=".npz path exported by Stage1")
    parser.add_argument("--delta", type=float, default=None, help="gray-zone radius in probability space")
    parser.add_argument("--delta_logit", type=float, default=1.5, help="gray-zone radius in logit space")
    parser.add_argument("--gray_top_percent", type=float, default=None, help="top percent closest to threshold")
    parser.add_argument("--gray_on_pred", choices=["all", "neg_only"], default="all",
                        help="gray-zone candidate pool: all samples or only baseline_pred==0")
    parser.add_argument("--limit_calls", type=int, default=None, help="debug only first N gray samples")
    parser.add_argument("--save_dir", default="./stage2_with_cfg_features", help="output dir")
    parser.add_argument("--force_jsonl", default=None, help="override meta.test_jsonl_path")
    parser.add_argument("--no_pretty", action="store_true", help="disable code prettify")
    parser.add_argument("--code_clip", type=int, default=6000)
    parser.add_argument("--cfg_keep_nodes", type=int, default=60)
    parser.add_argument("--cfg_keep_edges", type=int, default=80)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api_model", type=str, default=API_MODEL_DEFAULT)
    parser.add_argument("--request_timeout", type=int, default=240)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--sleep_sec", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ---------------- Load stage1 artifacts ----------------
    arr = np.load(args.artefacts, allow_pickle=True)
    pred_probs = np.asarray(arr["pred_probs"]).flatten()
    test_labels = np.asarray(arr["test_labels"]).astype(int).flatten()
    test_ids = arr["test_ids"]
    thr = float(arr["valid_thr"])
    meta = arr["meta"].item() if "meta" in arr else {}
    test_jsonl_path = args.force_jsonl or meta.get("test_jsonl_path") or "test.jsonl"
    id_field = meta.get("id_field") if isinstance(meta, dict) else None

    print(f"[Load] thr={thr:.6f}, artefacts={args.artefacts}")
    print(f"[Info] test_jsonl_path={test_jsonl_path}, id_field={id_field}")
    print(f"[Info] model={args.api_model}")

    base_preds = (pred_probs > thr).astype(int)

    print("\n[Baseline only @ thr]")
    _ = evaluate_baseline(test_labels, pred_probs, thr)

    # ---------------- Select gray-zone ----------------
    gray = select_gray_indices(
        pred_probs=pred_probs,
        base_preds=base_preds,
        thr=thr,
        gray_on_pred=args.gray_on_pred,
        gray_top_percent=args.gray_top_percent,
        delta_logit=args.delta_logit,
        delta=args.delta,
    )

    candidate_count = int((base_preds == 0).sum()) if args.gray_on_pred == "neg_only" else len(base_preds)
    print(f"[Gray] candidate pool mode={args.gray_on_pred}, candidate_count={candidate_count}")

    if args.limit_calls is not None:
        gray = gray[:args.limit_calls]

    print(f"Total gray samples (within range/limit): {len(gray)}")

    # ---------------- Load code + cfg strictly aligned ----------------
    codes, cfgs = load_code_cfg_aligned(
        test_jsonl_path,
        test_ids,
        pretty=not args.no_pretty,
        id_field=id_field
    )

    if len(codes) != len(pred_probs):
        raise RuntimeError(f"Length mismatch: codes={len(codes)} vs probs={len(pred_probs)}")

    # ---------------- LLM feature generation loop ----------------
    records: List[Dict[str, Any]] = []
    stats = {
        "calls": int(len(gray)),
        "parsed_ok": 0,
        "parse_fail": 0,
        "gate_applied_count": 0,
        "evidence_gate_count": 0,
        "context_gate_count": 0,
        "safe_parse_suppress_count": 0,
        "helper_semantics_suppress_count": 0,
        "logging_only_suppress_count": 0,
        "overflow_context_suppress_count": 0,
        "raw_positive_count": 0,
        "adj_positive_count": 0,
    }

    for t, i in enumerate(gray, 1):
        code = codes[i]
        cfg_text = cfgs[i]

        code_feat = extract_code_summary(code)
        cfg_feat = extract_cfg_summary(cfg_text)

        user_block, code_used, cfg_used = assemble_user_block(
            code,
            cfg_text,
            cfg_keep_nodes=args.cfg_keep_nodes,
            cfg_keep_edges=args.cfg_keep_edges,
            code_clip=args.code_clip
        )

        rec: Dict[str, Any] = {
            "idx": int(i),
            "func_id": str(test_ids[i]),
            "label": int(test_labels[i]),
            "base_prob": float(pred_probs[i]),
            "baseline_pred": int(base_preds[i]),
            "gray": True,

            "code_used": code_used,
            "cfg_used": cfg_used,

            "llm_raw": None,
            "llm_error": None,
            "llm_ok": False,

            "llm_final_label_raw": None,
            "llm_confidence_raw": None,

            "evidence_sufficient": None,
            "needs_more_context": None,
            "context_missing_type": "none",
            "confirmed_unsafe_write": 0,
            "confirmed_unsafe_free": 0,
            "confirmed_source_to_sink": 0,
            "safe_parse_pattern": 0,
            "speculative_only": 0,

            "llm_final_label": None,
            "llm_confidence": None,
            "has_input_validation": 0,
            "has_bounds_check": 0,
            "uses_untrusted_input": 0,
            "has_null_check": 0,
            "has_error_handling_path": 0,
            "vuln_type": None,
            "rationale": None,

            "gate_applied": False,
            "gate_reason": "",
            "evidence_gate_triggered": 0,
            "context_gate_triggered": 0,
            "safe_parse_suppress_triggered": 0,
            "helper_semantics_suppress_triggered": 0,
            "logging_only_suppress_triggered": 0,
            "overflow_context_suppress_triggered": 0,
        }

        rec.update(code_feat)
        rec.update(cfg_feat)

        try:
            gen = call_llm_chat(
                user_content=user_block,
                model_name=args.api_model,
                extra_hint="Return ONLY one valid JSON object. Do not add explanation.",
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.request_timeout,
                retry=args.retry,
                sleep_sec=args.sleep_sec
            )
            rec["llm_raw"] = gen
            js = extract_json_block(gen)
        except Exception as e:
            rec["llm_error"] = str(e)
            js = None

        if js is not None:
            rec = fill_structured_fields_from_model(js, rec)
            if rec["llm_final_label_raw"] is not None and rec["llm_confidence_raw"] is not None:
                rec["llm_ok"] = True
                stats["parsed_ok"] += 1
                stats["raw_positive_count"] += int(rec["llm_final_label_raw"] == 1)
            else:
                stats["parse_fail"] += 1
        else:
            stats["parse_fail"] += 1

        rec = apply_conservative_gate(rec, code)

        if rec["gate_applied"]:
            stats["gate_applied_count"] += 1
        stats["evidence_gate_count"] += int(rec["evidence_gate_triggered"])
        stats["context_gate_count"] += int(rec["context_gate_triggered"])
        stats["safe_parse_suppress_count"] += int(rec["safe_parse_suppress_triggered"])
        stats["helper_semantics_suppress_count"] += int(rec["helper_semantics_suppress_triggered"])
        stats["logging_only_suppress_count"] += int(rec["logging_only_suppress_triggered"])
        stats["overflow_context_suppress_count"] += int(rec["overflow_context_suppress_triggered"])
        stats["adj_positive_count"] += int(rec["llm_final_label"] == 1) if rec["llm_final_label"] is not None else 0

        records.append(rec)

        if t % 20 == 0 or t == len(gray):
            print(f"[Progress] {t}/{len(gray)} gray samples processed.")

    print(
        f"[Stage2-LLM-CFG features] calls={stats['calls']} | "
        f"parsed_ok={stats['parsed_ok']} | "
        f"parse_fail={stats['parse_fail']}"
    )
    print(
        f"[Gate] raw_positive={stats['raw_positive_count']} | "
        f"adj_positive={stats['adj_positive_count']} | "
        f"gate_applied={stats['gate_applied_count']} | "
        f"evidence_gate={stats['evidence_gate_count']} | "
        f"context_gate={stats['context_gate_count']} | "
        f"safe_parse_suppress={stats['safe_parse_suppress_count']} | "
        f"helper_semantics_suppress={stats['helper_semantics_suppress_count']} | "
        f"logging_only_suppress={stats['logging_only_suppress_count']} | "
        f"overflow_context_suppress={stats['overflow_context_suppress_count']}"
    )

    # ---------------- Save outputs ----------------
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    jsonl_path = os.path.join(args.save_dir, f"gray_features_llm_with_cfg_api_{stamp}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[Saved] gray feature JSONL -> {jsonl_path}")

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.save_dir, f"gray_features_llm_with_cfg_api_{stamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Saved] gray feature CSV -> {csv_path}")

    gray_labels = test_labels[gray] if len(gray) > 0 else np.array([], dtype=int)
    gray_base = base_preds[gray] if len(gray) > 0 else np.array([], dtype=int)

    run_meta = {
        "args": vars(args),
        "thr": thr,
        "llm_model": args.api_model,
        "api_url": API_BASE_URL,
        "gray_count": int(len(gray)),
        "gray_on_pred": args.gray_on_pred,
        "candidate_count": int(candidate_count),
        "gray_pos_count": int((gray_labels == 1).sum()) if len(gray_labels) > 0 else 0,
        "gray_neg_count": int((gray_labels == 0).sum()) if len(gray_labels) > 0 else 0,
        "gray_fn_count": int(((gray_labels == 1) & (gray_base == 0)).sum()) if len(gray_labels) > 0 else 0,
        "id_field": id_field,
        "test_jsonl_path": os.path.abspath(test_jsonl_path),
        "stats": stats,
    }
    meta_path = os.path.join(args.save_dir, f"run_meta_{stamp}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    print(f"[Saved] run meta -> {meta_path}")


if __name__ == "__main__":
    main()