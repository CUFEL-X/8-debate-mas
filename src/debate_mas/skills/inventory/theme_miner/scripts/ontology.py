"""
Ontology Inference Engine (本地知识推理引擎)

- 负责加载 references/ontology.yaml
- 提供关键词扩展（aliases / expands_to / weight）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# 全局缓存，避免每次调用都读 IO
_CONCEPTS_CACHE: Optional[Dict[str, Any]] = None

def _ontology_yaml_path() -> Path:
    # YAML 文件：skills/inventory/theme_miner/references/ontology.yaml
    return Path(__file__).resolve().parent.parent / "references" / "ontology.yaml"

def _load_ontology() -> Dict:
    """加载并解析 yaml 文件 (Lazy Loading)"""
    global _CONCEPTS_CACHE
    if _CONCEPTS_CACHE is not None:
        return _CONCEPTS_CACHE

    yaml_path = _ontology_yaml_path()
    if not yaml_path.exists():
        print(f"⚠️ [Ontology] 警告: 找不到知识库文件 {yaml_path}")
        _CONCEPTS_CACHE = {}
        return _CONCEPTS_CACHE

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _CONCEPTS_CACHE = data.get("concepts", {}) or {}
        print(f"📚 [Ontology] 已加载 {len(_CONCEPTS_CACHE)} 个宏观概念")
        return _CONCEPTS_CACHE
    except Exception as e:
        print(f"❌ [Ontology] 解析失败: {e}")
        _CONCEPTS_CACHE = {}
        return _CONCEPTS_CACHE

def get_concept_meta(query: str) -> Dict[str, Any]:
    """
    [推理核心接口]
    输入: 用户搜索词 (如 "新质生产力发展", "AI")
    输出: 概念元数据字典，包含:
          - name: 标准概念名
          - expansions: 行业词列表
          - weight: 静态权重
          - found: 是否命中
    """
    concepts = _load_ontology()

    result = {
        "name": query,
        "expansions": [],
        "weight": 1.0, 
        "found": False
    }

    if not concepts:
        return result

    query_norm = str(query).strip()
    if not query_norm:
        return result
    
    # 1. 精确匹配 Key
    if query_norm in concepts:
        info = concepts[query_norm]
        result.update({
            "name": query_norm,
            "expansions": info.get("expands_to", []) or [],
            "weight": info.get("weight", 1.0),
            "found": True
        })
        return result

    # 2. 模糊/别名匹配
    for key, info in (concepts or {}).items():
        key_s = str(key).strip()
        if not key_s:
            continue
        info = info or {}

        match_key = (query_norm in key_s) or (key_s in query_norm)

        match_alias = False
        for alias in (info.get("aliases", []) or []):
            alias_s = str(alias).strip()
            if not alias_s:
                continue
            if (query_norm in alias_s) or (alias_s in query_norm):
                match_alias = True
                break
        
        if match_key or match_alias:
            result.update({
                "name": key_s, 
                "expansions": info.get("expands_to", []) or [],
                "weight": info.get("weight", 1.0),
                "found": True
            })
            return result

    return result