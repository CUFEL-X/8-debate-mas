# core/blend_rank.py
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def dedup_by_symbol_keep_best(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去重：同 symbol 保留 score 更高的那条，并合并 reason/source/extra。"""
    best: Dict[str, Dict[str, Any]] = {}

    for it in items or []:
        sym = str(it.get("symbol", "")).strip()
        if not sym:
            continue

        score = _safe_float(it.get("score"), 0.0)
        if sym not in best:
            best[sym] = deepcopy(it)
            best[sym]["score"] = score
            best[sym].setdefault("reason", "")
            best[sym].setdefault("source_skill", it.get("source_skill", "unknown"))
            best[sym].setdefault("extra", {})
            continue

        cur = best[sym]
        if score > _safe_float(cur.get("score"), 0.0):
            prev = deepcopy(cur)
            best[sym] = deepcopy(it)
            best[sym]["score"] = score
            best[sym].setdefault("extra", {})
            best[sym]["extra"].setdefault("merged_from", [])
            best[sym]["extra"]["merged_from"].append(prev)
        else:
            cur.setdefault("extra", {})
            cur["extra"].setdefault("merged_from", [])
            cur["extra"]["merged_from"].append(deepcopy(it))

        cur2 = best[sym]
        r1 = str(cur2.get("reason", "") or "").strip()
        r2 = str(it.get("reason", "") or "").strip()
        if r2 and r2 not in r1:
            cur2["reason"] = (r1 + "；" + r2).strip("；")

        cur2.setdefault("extra", {})
        cur2["extra"].setdefault("sources", [])
        s = str(it.get("source_skill") or it.get("source") or "unknown")
        if s not in cur2["extra"]["sources"]:
            cur2["extra"]["sources"].append(s)

    out = list(best.values())
    out.sort(key=lambda x: _safe_float(x.get("score"), 0.0), reverse=True)
    return out


def merge_candidates(
    candidate_lists: List[List[Dict[str, Any]]],
    *,
    source_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    多来源候选融合（fancy + 可追溯）：
    - 每条候选最好包含 source_skill
    - score 做来源加权（展示分），并在 extra.blend 留痕
    """
    source_weights = source_weights or {}

    merged: List[Dict[str, Any]] = []
    for lst in candidate_lists or []:
        for it in lst or []:
            x = deepcopy(it)
            src = str(x.get("source_skill") or x.get("source") or "unknown")
            w = float(source_weights.get(src, 1.0))

            raw = _safe_float(x.get("score"), 0.0)
            show = raw * w

            x["extra"] = x.get("extra") or {}
            x["extra"]["blend"] = {
                "source_skill": src,
                "raw_score": raw,
                "weight": w,
                "weighted_score": show,
            }
            x["score"] = show
            x["source_skill"] = src
            merged.append(x)

    return dedup_by_symbol_keep_best(merged)


def explain_merge(merged: List[Dict[str, Any]], *, top_n: int = 5) -> List[str]:
    """产出给 memo/log 用的“融合解释要点”。"""
    out: List[str] = []
    if not merged:
        return ["候选融合结果为空。"]

    out.append(f"融合后候选数量: {len(merged)}。")
    for i, it in enumerate(merged[: int(top_n)]):
        sym = it.get("symbol", "")
        sc = _safe_float(it.get("score"), 0.0)
        blend = (it.get("extra") or {}).get("blend", {})
        src = blend.get("source_skill", it.get("source_skill", "unknown"))
        raw = blend.get("raw_score", None)
        w = blend.get("weight", None)
        out.append(f"Top{i+1}: {sym} 展示分={sc:.2f} 来源={src} raw={raw} w={w}")

    return out
