# protocol/etf_debate.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Tuple

PayloadType = Literal["CANDIDATES", "OBJECTIONS", "DIFF", "DECISIONS"]

ALLOWED_TYPES: Tuple[str, ...] = ("CANDIDATES", "OBJECTIONS", "DIFF", "DECISIONS")


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        s = s[first_nl + 1 :] if first_nl != -1 else ""
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s

def _extract_last_json_object_span(s: str) -> Optional[Tuple[int, int]]:
    """反向配对：从最后一个 '}' 往回找到与之匹配的 '{'"""
    if not s:
        return None

    end = s.rfind("}")
    if end < 0:
        return None

    depth = 0
    in_str = False
    escape = False

    # 从 end 往回扫：遇到 '}' depth+1，遇到 '{' depth-1，depth 回到 0 的那个 '{' 是起点
    for i in range(end, -1, -1):
        ch = s[i]

        # --- 字符串状态机（避免 JSON 字符串里出现 { } 干扰）---
        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue

        # --- 非字符串状态下，计数括号 ---
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                return (i, end)

    return None

def _parse_last_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    if not text:
        return None, None

    s = _strip_code_fences(text)
    span = _extract_last_json_object_span(s)
    if not span:
        return None, None

    start, end = span
    chunk = s[start : end + 1]
    try:
        obj = json.loads(chunk)
        if isinstance(obj, dict):
            return obj, start
        return None, None
    except Exception:
        return None, None
    
def try_parse_payload(text: str) -> Optional[Dict[str, Any]]:
    """
    从模型输出中稳健抽取 JSON 对象（允许自然语言混杂，只抓最后一个完整 {...}）。
    返回 dict 或 None。
    """
    obj, _ = _parse_last_json_object(text) 
    return obj

def try_parse_payload_with_span(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """
    返回 (payload_obj, start_idx)
    - payload_obj: dict 或 None
    - start_idx: payload 在文本中的 '{' 起始索引；payload_obj 为 None 时也为 None
    """
    return _parse_last_json_object(text)

def validate_payload(obj: Dict[str, Any]) -> bool:
    """
    轻量校验：只校验协议最小必需字段。
    - type 必须在允许范围内
    - items 必须是 list
    """
    if not isinstance(obj, dict):
        raise TypeError("payload 必须是 dict")

    t = str(obj.get("type", "") or "").strip().upper()
    if t not in ALLOWED_TYPES:
        raise ValueError(f"payload.type 非法: {t!r}（允许: {ALLOWED_TYPES}）")

    items = obj.get("items", None)
    if not isinstance(items, list):
        raise ValueError("payload.items 必须是 list")

    # stop_suggest 非必需，但建议存在（便于图收敛）
    stop = obj.get("stop_suggest", None)
    if stop is not None and not isinstance(stop, str):
        raise ValueError("payload.stop_suggest 若存在，必须是 str")

    return True


def make_payload(
    payload_type: PayloadType,
    items: List[Dict[str, Any]],
    *,
    stop_suggest: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """（可选）小工具：构造 payload，并做最小校验。"""
    out: Dict[str, Any] = {"type": payload_type, "items": items}
    if stop_suggest:
        out["stop_suggest"] = stop_suggest
    if meta:
        out["meta"] = meta
    validate_payload(out)
    return out
