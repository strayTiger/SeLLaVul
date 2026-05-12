# -*- coding: utf-8 -*-
"""
build_refine_dataset.py

python build_refine_dataset.py ^
  --train_json ./reveal_train.jsonl ^
  --train_oof_npz ./artifacts/reveal_baseline_artifacts_train_oof_graphcodebert_2026-04-29_10-33-56.npz ^
  --stage2_train_gray_jsonl ./stage2_reveal_train_with_nocfg_api_negonly_top3/gray_features_llm_with_cfg_api_2026-04-29_15-08-21.jsonl ^
  --save_dir ./refine_dataset_reveal_graphcodebert_top3 ^
  --train_id_field idx ^
  --gray_id_field func_id ^
  --alpha_gold 0.85 ^
  --min_soft_conf 0.6 ^
  --min_aux_conf 0.6 ^
  --gray_weight_agree 1.20 ^
  --gray_weight_disagree 1.00 ^
  --gray_weight_no_llm 1.00
"""

from __future__ import absolute_import, division, print_function

import os
import json
import argparse
import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Safe helpers
# =========================================================
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


def safe_bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n", ""):
        return False
    return default


def normalize_id(v):
    if v is None:
        return None
    return str(v)


def get_row_id(js: Dict[str, Any], id_field: str) -> Optional[str]:
    v = js.get(id_field, None)
    if v is None:
        return None
    return normalize_id(v)


def sigmoid_prob_from_llm_label_conf(llm_label: Optional[int], llm_conf: Optional[float]) -> Optional[float]:
    """
    把 (llm_final_label, llm_confidence) 转成“预测为正类的概率”。
    仅用于 soft_label 候选构造；真正启用还要满足额外条件。
    """
    if llm_label is None or llm_conf is None:
        return None
    llm_conf = max(0.0, min(1.0, float(llm_conf)))
    if llm_label == 1:
        return llm_conf
    if llm_label == 0:
        return 1.0 - llm_conf
    return None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def preview_ids(name: str, ids: List[str], max_n: int = 5):
    print(f"[Preview] {name} first {min(max_n, len(ids))} ids: {ids[:max_n]}")


# =========================================================
# Loaders
# =========================================================
def load_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                js = json.loads(line)
                rows.append(js)
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSONL at line {line_no}: {e}")
    return rows


def load_train_raw_jsonl(path: str, id_field: str) -> Tuple[List[Dict[str, Any]], List[str], np.ndarray]:
    rows = load_jsonl_rows(path)

    ids = []
    labels = []
    seen = set()

    for i, js in enumerate(rows):
        rid = get_row_id(js, id_field)
        if rid is None:
            raise RuntimeError(f"Missing id field '{id_field}' in train_json row #{i}")

        if rid in seen:
            raise RuntimeError(f"Duplicate id detected in train_json: {rid}")
        seen.add(rid)

        if "target" not in js:
            raise RuntimeError(f"Missing target field in train_json row id={rid}")

        y = safe_int(js["target"], default=None)
        if y not in (0, 1):
            raise RuntimeError(f"Invalid target for id={rid}: {js['target']}")

        ids.append(rid)
        labels.append(y)

    return rows, ids, np.asarray(labels, dtype=np.int32)


def load_stage1_oof_npz(path: str) -> Dict[str, Any]:
    arr = np.load(path, allow_pickle=True)

    required = ["pred_probs", "test_labels", "test_ids", "valid_thr"]
    for k in required:
        if k not in arr:
            raise RuntimeError(f"Missing key '{k}' in OOF npz: {path}")

    pred_probs = np.asarray(arr["pred_probs"], dtype=float)
    labels = np.asarray(arr["test_labels"], dtype=int)
    ids = [normalize_id(x) for x in arr["test_ids"].tolist()]
    thr = float(arr["valid_thr"])

    meta = {}
    if "meta" in arr:
        try:
            meta = arr["meta"].item()
        except Exception:
            meta = {}

    if not (len(pred_probs) == len(labels) == len(ids)):
        raise RuntimeError("Mismatch lengths in OOF npz: pred_probs / test_labels / test_ids")

    seen = set()
    out = {}
    base_preds = (pred_probs > thr).astype(int)

    for rid, p, y, bp in zip(ids, pred_probs, labels, base_preds):
        if rid is None:
            raise RuntimeError("Found None id in OOF npz test_ids")
        if rid in seen:
            raise RuntimeError(f"Duplicate id detected in OOF npz: {rid}")
        seen.add(rid)

        out[rid] = {
            "base_prob": float(p),
            "label_npz": int(y),
            "baseline_pred": int(bp),
        }

    return {
        "row_map": out,
        "thr": thr,
        "meta": meta,
        "n": len(ids),
        "ids": ids,
    }


def load_stage2_gray_jsonl(path: str, id_field: str, debug_preview: int = 5) -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl_rows(path)
    out = {}

    if len(rows) == 0:
        raise RuntimeError("Stage2 gray jsonl is empty.")

    for i, js in enumerate(rows[:debug_preview]):
        print(
            f"[Stage2 Preview #{i}] {id_field}={js.get(id_field)} | "
            f"idx={js.get('idx')} | func_id={js.get('func_id')} | gray={js.get('gray')} | llm_ok={js.get('llm_ok')}"
        )

    for js in rows:
        rid = get_row_id(js, id_field)
        if rid is None:
            raise RuntimeError(f"Missing id field '{id_field}' in stage2 gray jsonl row: {js}")

        if rid in out:
            raise RuntimeError(f"Duplicate id detected in stage2 gray jsonl: {rid}")

        out[rid] = {
            "idx": safe_int(js.get("idx"), default=None),
            "func_id": normalize_id(js.get("func_id")),
            "label": safe_int(js.get("label"), default=None),
            "base_prob": safe_float(js.get("base_prob"), default=None),
            "baseline_pred": safe_int01(js.get("baseline_pred"), default=None),
            "gray": safe_bool(js.get("gray", True), default=True),
            "llm_ok": safe_bool(js.get("llm_ok", False), default=False),
            "llm_final_label": safe_int01(js.get("llm_final_label"), default=None),
            "llm_confidence": safe_float(js.get("llm_confidence"), default=None),
            "vuln_type": js.get("vuln_type"),
            "rationale": js.get("rationale"),

            "has_input_validation": safe_int01(js.get("has_input_validation"), default=None),
            "has_bounds_check": safe_int01(js.get("has_bounds_check"), default=None),
            "uses_untrusted_input": safe_int01(js.get("uses_untrusted_input"), default=None),
            "has_null_check": safe_int01(js.get("has_null_check"), default=None),
            "has_error_handling_path": safe_int01(js.get("has_error_handling_path"), default=None),

            "token_count": safe_float(js.get("token_count"), default=None),
            "line_count": safe_float(js.get("line_count"), default=None),
            "branch_count": safe_float(js.get("branch_count"), default=None),
            "return_count": safe_float(js.get("return_count"), default=None),
            "pointer_count": safe_float(js.get("pointer_count"), default=None),
            "array_access_count": safe_float(js.get("array_access_count"), default=None),
            "dangerous_api_count": safe_float(js.get("dangerous_api_count"), default=None),

            "cfg_is_real": safe_int01(js.get("cfg_is_real"), default=None),
            "cfg_is_global": safe_int01(js.get("cfg_is_global"), default=None),
            "cfg_node_count": safe_float(js.get("cfg_node_count"), default=None),
            "cfg_edge_count": safe_float(js.get("cfg_edge_count"), default=None),
            "has_validation_path_cfg": safe_int01(js.get("has_validation_path_cfg"), default=None),
            "has_error_handling_path_cfg": safe_int01(js.get("has_error_handling_path_cfg"), default=None),
            "has_null_check_cfg": safe_int01(js.get("has_null_check_cfg"), default=None),
        }

    return out


# =========================================================
# Core builder
# =========================================================
def build_refine_records(
    train_rows: List[Dict[str, Any]],
    train_ids: List[str],
    train_labels: np.ndarray,
    stage1_oof: Dict[str, Any],
    gray_map: Dict[str, Dict[str, Any]],
    alpha_gold: float,
    min_soft_conf: float,
    min_aux_conf: float,
    gray_weight_agree: float,
    gray_weight_disagree: float,
    gray_weight_no_llm: float,
    normal_weight: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    row_map = stage1_oof["row_map"]
    thr = float(stage1_oof["thr"])

    out_rows: List[Dict[str, Any]] = []

    stats = {
        "total_n": len(train_rows),
        "gray_n": 0,
        "non_gray_n": 0,
        "gray_llm_ok_n": 0,
        "gray_llm_not_ok_n": 0,
        "gray_llm_agree_gold_n": 0,
        "gray_llm_disagree_gold_n": 0,
        "aux_mask_n": 0,
        "soft_label_used_n": 0,
    }

    aux_keys = [
        "has_input_validation",
        "has_bounds_check",
        "uses_untrusted_input",
        "has_null_check",
        "has_error_handling_path",
    ]

    for js, rid, y in zip(train_rows, train_ids, train_labels):
        if rid not in row_map:
            raise RuntimeError(f"ID={rid} exists in train_json but not in train_oof_npz")

        base_info = row_map[rid]

        if int(base_info["label_npz"]) != int(y):
            raise RuntimeError(
                f"Label mismatch for id={rid}: train_json={y}, train_oof_npz={base_info['label_npz']}"
            )

        g = gray_map.get(rid, None)

        is_gray = int(g is not None and safe_bool(g.get("gray", True), default=True))
        llm_ok = safe_bool(g["llm_ok"], default=False) if g is not None else False
        llm_label = g["llm_final_label"] if g is not None else None
        llm_conf = g["llm_confidence"] if g is not None else None
        llm_prob_pos = sigmoid_prob_from_llm_label_conf(llm_label, llm_conf)

        soft_label = float(y)
        sample_weight = float(normal_weight)
        aux_mask = 0
        use_soft_label = 0

        aux_has_input_validation = -1
        aux_has_bounds_check = -1
        aux_uses_untrusted_input = -1
        aux_has_null_check = -1
        aux_has_error_handling_path = -1

        if is_gray:
            stats["gray_n"] += 1
        else:
            stats["non_gray_n"] += 1

        if is_gray and llm_ok:
            stats["gray_llm_ok_n"] += 1

            llm_agree_gold = (llm_label == int(y))

            if llm_agree_gold:
                stats["gray_llm_agree_gold_n"] += 1
                sample_weight = float(gray_weight_agree)
            else:
                stats["gray_llm_disagree_gold_n"] += 1
                sample_weight = float(gray_weight_disagree)

            # 仅高置信且与 gold 一致时启用 soft_label
            soft_ok = (
                llm_agree_gold
                and (llm_conf is not None)
                and (llm_conf >= min_soft_conf)
                and (llm_prob_pos is not None)
            )

            if soft_ok:
                soft_label = float(alpha_gold * y + (1.0 - alpha_gold) * llm_prob_pos)
                soft_label = max(0.02, min(0.98, soft_label))
                use_soft_label = 1
                stats["soft_label_used_n"] += 1
            else:
                soft_label = float(y)
                use_soft_label = 0

            # 仅高置信且与 gold 一致时启用 aux
            aux_vals = [g.get(k, None) for k in aux_keys]
            aux_ok = (
                llm_agree_gold
                and (llm_conf is not None)
                and (llm_conf >= min_aux_conf)
                and all(v in (0, 1) for v in aux_vals)
            )

            if aux_ok:
                aux_mask = 1
                stats["aux_mask_n"] += 1

                aux_has_input_validation = int(g["has_input_validation"])
                aux_has_bounds_check = int(g["has_bounds_check"])
                aux_uses_untrusted_input = int(g["uses_untrusted_input"])
                aux_has_null_check = int(g["has_null_check"])
                aux_has_error_handling_path = int(g["has_error_handling_path"])

        elif is_gray and not llm_ok:
            stats["gray_llm_not_ok_n"] += 1
            sample_weight = float(gray_weight_no_llm)
            soft_label = float(y)
            use_soft_label = 0

        rec = dict(js)

        rec.update({
            # baseline
            "base_prob": float(base_info["base_prob"]),
            "baseline_pred": int(base_info["baseline_pred"]),
            "baseline_thr": float(thr),

            # gray / llm
            "is_gray_stage2": int(is_gray),
            "llm_ok": bool(llm_ok),
            "llm_final_label": int(llm_label) if llm_label in (0, 1) else None,
            "llm_confidence": float(llm_conf) if llm_conf is not None else None,
            "llm_prob_pos": float(llm_prob_pos) if llm_prob_pos is not None else None,
            "vuln_type": g.get("vuln_type") if g is not None else None,
            "rationale": g.get("rationale") if g is not None else None,

            # refine targets
            "hard_label": int(y),
            "soft_label": float(soft_label),
            "sample_weight": float(sample_weight),
            "use_soft_label": int(use_soft_label),

            # aux targets
            "aux_mask": int(aux_mask),
            "aux_has_input_validation": int(aux_has_input_validation),
            "aux_has_bounds_check": int(aux_has_bounds_check),
            "aux_uses_untrusted_input": int(aux_uses_untrusted_input),
            "aux_has_null_check": int(aux_has_null_check),
            "aux_has_error_handling_path": int(aux_has_error_handling_path),

            # stage2 summary features
            "stage2_token_count": g.get("token_count") if g is not None else None,
            "stage2_line_count": g.get("line_count") if g is not None else None,
            "stage2_branch_count": g.get("branch_count") if g is not None else None,
            "stage2_return_count": g.get("return_count") if g is not None else None,
            "stage2_pointer_count": g.get("pointer_count") if g is not None else None,
            "stage2_array_access_count": g.get("array_access_count") if g is not None else None,
            "stage2_dangerous_api_count": g.get("dangerous_api_count") if g is not None else None,
            "stage2_cfg_is_real": g.get("cfg_is_real") if g is not None else None,
            "stage2_cfg_node_count": g.get("cfg_node_count") if g is not None else None,
            "stage2_cfg_edge_count": g.get("cfg_edge_count") if g is not None else None,
        })

        out_rows.append(rec)

    stats["gray_rate"] = float(stats["gray_n"] / max(stats["total_n"], 1))
    stats["aux_mask_rate_all"] = float(stats["aux_mask_n"] / max(stats["total_n"], 1))
    stats["aux_mask_rate_gray"] = float(stats["aux_mask_n"] / max(stats["gray_n"], 1))
    stats["gray_llm_ok_rate"] = float(stats["gray_llm_ok_n"] / max(stats["gray_n"], 1))
    stats["gray_llm_agree_rate"] = float(stats["gray_llm_agree_gold_n"] / max(stats["gray_llm_ok_n"], 1))

    return out_rows, stats


# =========================================================
# Save outputs
# =========================================================
def save_refine_outputs(
    records: List[Dict[str, Any]],
    stats: Dict[str, Any],
    save_dir: str,
    stamp: str,
    config: Dict[str, Any],
):
    ensure_dir(save_dir)

    jsonl_path = os.path.join(save_dir, f"refine_train_{stamp}.jsonl")
    csv_path = os.path.join(save_dir, f"refine_train_{stamp}.csv")
    npz_path = os.path.join(save_dir, f"refine_targets_{stamp}.npz")
    summary_path = os.path.join(save_dir, f"refine_summary_{stamp}.json")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    ids = df["idx"].values if "idx" in df.columns else np.arange(len(df))
    hard_label = df["hard_label"].astype(float).values
    soft_label = df["soft_label"].astype(float).values
    sample_weight = df["sample_weight"].astype(float).values
    base_prob = df["base_prob"].astype(float).values
    baseline_pred = df["baseline_pred"].astype(int).values
    is_gray = df["is_gray_stage2"].astype(int).values
    aux_mask = df["aux_mask"].astype(int).values

    aux_has_input_validation = df["aux_has_input_validation"].astype(int).values
    aux_has_bounds_check = df["aux_has_bounds_check"].astype(int).values
    aux_uses_untrusted_input = df["aux_uses_untrusted_input"].astype(int).values
    aux_has_null_check = df["aux_has_null_check"].astype(int).values
    aux_has_error_handling_path = df["aux_has_error_handling_path"].astype(int).values

    np.savez_compressed(
        npz_path,
        ids=np.asarray(ids),
        hard_label=hard_label,
        soft_label=soft_label,
        sample_weight=sample_weight,
        base_prob=base_prob,
        baseline_pred=baseline_pred,
        is_gray=is_gray,
        aux_mask=aux_mask,
        aux_has_input_validation=aux_has_input_validation,
        aux_has_bounds_check=aux_has_bounds_check,
        aux_uses_untrusted_input=aux_uses_untrusted_input,
        aux_has_null_check=aux_has_null_check,
        aux_has_error_handling_path=aux_has_error_handling_path,
    )

    summary = {
        "config": config,
        "stats": stats,
        "files": {
            "jsonl": os.path.abspath(jsonl_path),
            "csv": os.path.abspath(csv_path),
            "npz": os.path.abspath(npz_path),
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {jsonl_path}")
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {npz_path}")
    print(f"[Saved] {summary_path}")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_json", required=True, help="raw train.jsonl")
    parser.add_argument("--train_oof_npz", required=True, help="Stage1 train_oof artifacts npz")
    parser.add_argument("--stage2_train_gray_jsonl", required=True, help="Stage2 train gray jsonl")
    parser.add_argument("--save_dir", default="./refine_dataset_graphcodebert")

    parser.add_argument(
        "--train_id_field",
        default="idx",
        choices=["idx", "id", "uid", "func_id", "func_name"],
        help="ID field used in train_json and train_oof alignment"
    )
    parser.add_argument(
        "--gray_id_field",
        default="func_id",
        choices=["idx", "id", "uid", "func_id", "func_name"],
        help="ID field used in stage2 gray jsonl to align with train_json; for your current Stage2 output this should be func_id"
    )

    parser.add_argument("--alpha_gold", type=float, default=0.85,
                        help="soft_label = alpha_gold * gold + (1-alpha_gold) * llm_prob")
    parser.add_argument("--min_soft_conf", type=float, default=0.80,
                        help="minimum llm_confidence to enable soft_label")
    parser.add_argument("--min_aux_conf", type=float, default=0.80,
                        help="minimum llm_confidence to enable aux_mask")
    parser.add_argument("--gray_weight_agree", type=float, default=1.10,
                        help="sample weight for gray samples where llm_label agrees with gold")
    parser.add_argument("--gray_weight_disagree", type=float, default=1.00,
                        help="sample weight for gray samples where llm_label disagrees with gold")
    parser.add_argument("--gray_weight_no_llm", type=float, default=1.00,
                        help="sample weight for gray samples with llm_ok=False")
    parser.add_argument("--normal_weight", type=float, default=1.00,
                        help="sample weight for non-gray samples")

    args = parser.parse_args()
    ensure_dir(args.save_dir)

    if not (0.0 <= args.alpha_gold <= 1.0):
        raise ValueError("--alpha_gold must be in [0,1]")
    if not (0.0 <= args.min_soft_conf <= 1.0):
        raise ValueError("--min_soft_conf must be in [0,1]")
    if not (0.0 <= args.min_aux_conf <= 1.0):
        raise ValueError("--min_aux_conf must be in [0,1]")

    print(f"[Load] train_json={args.train_json}")
    train_rows, train_ids, train_labels = load_train_raw_jsonl(args.train_json, args.train_id_field)
    print(f"[Load] train rows={len(train_rows)}")
    preview_ids("train_json", train_ids)

    print(f"[Load] train_oof_npz={args.train_oof_npz}")
    stage1_oof = load_stage1_oof_npz(args.train_oof_npz)
    print(f"[Load] OOF rows={stage1_oof['n']} | thr={stage1_oof['thr']:.6f}")
    preview_ids("train_oof_npz", stage1_oof["ids"])

    train_id_set = set(train_ids)
    oof_id_set = set(stage1_oof["ids"])

    if train_id_set != oof_id_set:
        only_in_train = list(train_id_set - oof_id_set)[:10]
        only_in_oof = list(oof_id_set - train_id_set)[:10]
        raise RuntimeError(
            "train_json IDs and train_oof_npz IDs do not match exactly.\n"
            f"Only in train_json (sample): {only_in_train}\n"
            f"Only in train_oof_npz (sample): {only_in_oof}"
        )

    print(f"[Load] stage2_train_gray_jsonl={args.stage2_train_gray_jsonl}")
    gray_map = load_stage2_gray_jsonl(args.stage2_train_gray_jsonl, args.gray_id_field)
    gray_ids = list(gray_map.keys())
    print(f"[Load] stage2 gray rows={len(gray_map)}")
    preview_ids("stage2_gray", gray_ids)

    gray_id_set = set(gray_map.keys())
    overlap = train_id_set & gray_id_set
    miss_gray = gray_id_set - train_id_set

    print(f"[Check] train ids={len(train_id_set)}")
    print(f"[Check] gray ids={len(gray_id_set)}")
    print(f"[Check] overlap ids={len(overlap)}")
    print(f"[Check] overlap rate={len(overlap) / max(len(gray_id_set), 1):.4f}")

    if len(gray_id_set) == 0:
        raise RuntimeError("Stage2 gray jsonl is empty.")

    if len(overlap) == 0:
        sample_gray = list(gray_id_set)[:10]
        sample_train = list(train_id_set)[:10]
        raise RuntimeError(
            "No overlapping IDs between train_json and stage2_train_gray_jsonl.\n"
            "Likely wrong gray_id_field or wrong stage2 file.\n"
            f"Sample train IDs: {sample_train}\n"
            f"Sample gray IDs: {sample_gray}"
        )

    if len(miss_gray) > 0:
        print(f"[WARN] {len(miss_gray)} gray ids are not found in train_json; they will be ignored.")
        print(f"[WARN] sample missing gray ids: {list(miss_gray)[:10]}")

    records, stats = build_refine_records(
        train_rows=train_rows,
        train_ids=train_ids,
        train_labels=train_labels,
        stage1_oof=stage1_oof,
        gray_map=gray_map,
        alpha_gold=args.alpha_gold,
        min_soft_conf=args.min_soft_conf,
        min_aux_conf=args.min_aux_conf,
        gray_weight_agree=args.gray_weight_agree,
        gray_weight_disagree=args.gray_weight_disagree,
        gray_weight_no_llm=args.gray_weight_no_llm,
        normal_weight=args.normal_weight,
    )

    if stats["gray_n"] == 0:
        raise RuntimeError(
            "No gray samples were merged into refine dataset. "
            "The refine dataset is invalid."
        )

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config = {
        "train_json": os.path.abspath(args.train_json),
        "train_oof_npz": os.path.abspath(args.train_oof_npz),
        "stage2_train_gray_jsonl": os.path.abspath(args.stage2_train_gray_jsonl),
        "train_id_field": args.train_id_field,
        "gray_id_field": args.gray_id_field,
        "alpha_gold": float(args.alpha_gold),
        "min_soft_conf": float(args.min_soft_conf),
        "min_aux_conf": float(args.min_aux_conf),
        "gray_weight_agree": float(args.gray_weight_agree),
        "gray_weight_disagree": float(args.gray_weight_disagree),
        "gray_weight_no_llm": float(args.gray_weight_no_llm),
        "normal_weight": float(args.normal_weight),
    }

    print("\n=== Refine Dataset Summary ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    save_refine_outputs(
        records=records,
        stats=stats,
        save_dir=args.save_dir,
        stamp=stamp,
        config=config,
    )


if __name__ == "__main__":
    main()