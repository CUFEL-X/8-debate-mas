from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import yaml

# 约定：mappings.yaml 和本文件同目录
_MAPPING_YAML = Path(__file__).resolve().parents[1] / "references" / "mappings.yaml"

def _load_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML 配置"""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CFG = _load_yaml(_MAPPING_YAML)

INDUSTRY_FUZZY_MAP: Dict[str, List[str]] = _CFG.get("INDUSTRY_FUZZY_MAP", {}) or {}
THEME_KEYWORDS_MAP: Dict[str, List[str]] = _CFG.get("THEME_KEYWORDS_MAP", {}) or {}

# --- Guardrail 专用 ---
# 允许你在 YAML 里只挑部分 bucket 参与结构兜底
GUARDRAIL_BUCKETS: List[str] = _CFG.get("GUARDRAIL_BUCKETS", []) or []
GUARDRAIL_NOTE: str = str(_CFG.get("GUARDRAIL_NOTE", "") or "")
