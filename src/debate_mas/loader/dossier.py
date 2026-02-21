from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd

@dataclass
class Dossier:
    """
    【第一层：统一案卷】(The Unified Dossier)
    
    这是整个系统的“数据地基”。想象它是一个用于法庭辩论的“标准证据箱”。
    LangGraph 中的所有 Agent（正方/反方/裁判）都只能看到这个箱子里的内容。
    """
    # 1. 任务指令 (Mission)
    mission: str

    # 2. 结构化证据 (Structured Evidence)
    structured_data: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # 3. 文本证据 (Textual Evidence)
    unstructured_text: List[str] = field(default_factory=list)

    # 4. 案卷元数据 (Metadata)
    meta: Dict[str, Any] = field(default_factory=dict)

    # 每张表的元信息：来源/描述/行列/列名/时间戳等
    tables_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 每段文本的元信息：来源/长度/时间戳等（与 unstructured_text 同序）
    texts_meta: List[Dict[str, Any]] = field(default_factory=list)
    table_aliases: Dict[str, List[str]] = field(default_factory=dict)
    _alias_to_canonical: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    
    
    def register_table_aliases(self, mapping: Dict[str, Any]) -> None:
        """
        注册别名映射（两种格式都支持）：
        1) {"etf_basic": ["sampled_etf_basic", "basic"], ...}
        2) {"sampled_etf_basic": "etf_basic", ...}
        """
        if not mapping:
            return

        for k, v in mapping.items():
            # 格式 1
            if isinstance(v, (list, tuple)):
                canonical = str(k).strip()
                aliases = [str(x).strip() for x in v if str(x).strip()]
                if not canonical:
                    continue
                aliases = [str(x).strip() for x in v if str(x).strip()]
                self._ensure_canonical_bucket(canonical)
                for alias in aliases:
                    self._add_alias(alias=alias, canonical=canonical)
            # 格式 2
            else:
                alias = str(k).strip()
                canonical = str(v).strip()
                if not alias or not canonical:
                    continue
                self._ensure_canonical_bucket(canonical)
                self._add_alias(alias=alias, canonical=canonical)
    
    def _ensure_canonical_bucket(self, canonical: str) -> None:
        self.table_aliases.setdefault(canonical, [])

    def _add_alias(self, alias: str, canonical: str) -> None:
        alias = str(alias).strip()
        canonical = str(canonical).strip()
        if not alias or not canonical or alias == canonical:
            return

        if alias not in self.table_aliases[canonical]:
            self.table_aliases[canonical].append(alias)
        self._alias_to_canonical[alias] = canonical

    def resolve_table_name(self, name: str) -> Optional[str]:
        """把任意名字解析成真实存在的表名"""
        if not name:
            return None
        raw = str(name).strip()
        # 1) 直接命中
        if raw in self.structured_data:
            return raw

        # 2) alias -> canonical
        canonical = self._alias_to_canonical.get(raw)
        if canonical and canonical in self.structured_data:
            return canonical

        # 3) 兜底：用户传了 xxx.csv / xxx.xlsx / xxx.xls
        base = raw
        for suf in [".csv", ".xlsx", ".xls"]:
            if base.lower().endswith(suf):
                base = base[: -len(suf)]
                break
        if base in self.structured_data:
            return base
        canonical = self._alias_to_canonical.get(base)
        if canonical and canonical in self.structured_data:
            return canonical

        return None
    
    # -------------------------------------------------------
    # “积木方法”，可方便地装填数据
    # -------------------------------------------------------
    def add_table(
        self,
        name: str,
        df: pd.DataFrame,
        description: str = "",
        source: str = "unknown",
        extra: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        [工具方法] 添加表格证据。
        """
        self.structured_data[name] = df

        try:
            shape = (int(df.shape[0]), int(df.shape[1]))
            cols = [str(c) for c in list(df.columns)]
        except Exception:
            shape, cols = None, []

        m: Dict[str, Any] = {
            "name": name,
            "source": source,
            "description": description,
            "rows": shape[0] if shape else None,
            "cols": shape[1] if shape else None,
            "columns": cols,
            "added_at": datetime.now().isoformat(timespec="seconds"),
        }
        if extra and isinstance(extra, dict):
            m.update(extra)

        self.tables_meta[name] = m
        if aliases:
            self.register_table_aliases({name: aliases})

    def add_text(
        self,
        content: str,
        source: str = "Unknown",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """[工具方法] 添加文本证据。 """
        formatted_text = f"【来源: {source}】\n{content}\n"
        self.unstructured_text.append(formatted_text)
        
        m = {
            "source": source,
            "content_length": len(content),
            "added_at": datetime.now().isoformat(timespec="seconds"),
        }
        if extra and isinstance(extra, dict):
            m.update(extra)

        self.texts_meta.append(m)

    def frozen_view(self) -> Dict[str, Any]:
        """返回只读摘要（给 core/agent 透视案卷，不暴露原始 DataFrame）。"""
        tables: List[Dict[str, Any]] = []
        for name, m in self.tables_meta.items():
            tables.append(
                {
                    "name": name,
                    "source": m.get("source"),
                    "rows": m.get("rows"),
                    "cols": m.get("cols"),
                    "columns": (m.get("columns") or [])[:20],  # 防爆
                    "description": m.get("description", ""),
                }
            )

        texts: List[Dict[str, Any]] = []
        for i, m in enumerate(self.texts_meta):
            texts.append(
                {
                    "idx": i,
                    "source": m.get("source"),
                    "length": m.get("content_length"),
                    "added_at": m.get("added_at"),
                }
            )

        return {
            "mission": self.mission,
            "meta": dict(self.meta or {}),
            "tables": tables,
            "texts": texts,
        } 

    def summary(self) -> str:
        """[可视化] 打印案卷摘要。"""
        table_list = list(self.structured_data.keys())
        text_count = len(self.unstructured_text)
        
        preview_text = "无"
        if text_count > 0:
            first_text_lines = self.unstructured_text[0].split('\n')
            preview_text = (first_text_lines[1][:30] + "...") if len(first_text_lines) > 1 else (self.unstructured_text[0][:30] + "...")

        # tables meta 简要
        tables_meta_lines: List[str] = []
        for name in table_list[:20]:
            m = self.tables_meta.get(name, {})
            tables_meta_lines.append(f"- {name}: rows={m.get('rows')}, cols={m.get('cols')}, source={m.get('source')}")

        # texts meta 简要
        texts_meta_lines: List[str] = []
        for i, m in enumerate(self.texts_meta[:10]):
            texts_meta_lines.append(f"- [{i}] source={m.get('source')}, length={m.get('length')}")

        meta_preview = ""
        if self.meta:
            # 只展示少量键，避免刷屏
            keys = list(self.meta.keys())[:10]
            meta_preview = ", ".join([f"{k}={self.meta.get(k)}" for k in keys])

        return (
            f"\n📦 ========== 案卷 (Dossier) 概览 ==========\n"
            f"🎯 任务指令: {self.mission}\n"
            f"🧾 meta: {meta_preview or '无'}\n"
            f"📊 结构化数据 (Tables): {len(table_list)} 张 -> {table_list}\n"
            f"{(''.join([x + chr(10) for x in tables_meta_lines])) if tables_meta_lines else ''}"
            f"📄 非结构化数据 (Texts): {text_count} 篇 -> (首篇预览: {preview_text})\n"
            f"{(''.join([x + chr(10) for x in texts_meta_lines])) if texts_meta_lines else ''}"
            f"===========================================\n"
        )
    """
    def _canonical_name(self, name: str) -> str:
        #把外部名字/别名 归一到 canonical name
        if not name:
            return name
        m = (self.meta.get("table_alias_map") or {})
        return m.get(name, name)
    """
    # get_table：支持 alias 查表
    def get_table(self, name: str) -> Optional[pd.DataFrame]:
        """按名字或别名取结构化表格。如果不存在返回 None。"""
        canonical = self.resolve_table_name(name)
        if not canonical:
            return None
        return self.structured_data.get(canonical)

    def list_tables(self) -> List[str]:
        """返回当前案卷里所有表名，方便调试。"""
        return list(self.structured_data.keys())


    @classmethod
    def create_empty(cls, mission: str) -> "Dossier":
        """[初始化] 快速创建一个空案卷。"""
        return cls(mission=mission)