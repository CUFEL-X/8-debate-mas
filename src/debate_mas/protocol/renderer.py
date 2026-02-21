# protocol/renderer.py
from __future__ import annotations

import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

from .schema import EtfDecision, DebateMeta, DebateLog, DecisionAction


class DebateRenderer:
    """
    【Layer 4: 协议渲染器】
    将结构化对象渲染为物理文件：
    - log.json  (机器可读，可追溯，可回放)
    - memo.md   (人类可读，便于汇报)
    - rebalance.csv (调仓指令，便于下游执行/交接)
    """

    def __init__(self, output_dir: str = "./output_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)


    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def render(
        self,
        mission: str,
        decisions: List[EtfDecision],
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        执行渲染流程，生成交付三件套
        extra_meta: 由 Core 传入的 meta（比如 ref_date / rounds / tool_trace）
        """
        base_filename = self._build_base_filename(mission)
        json_path = self._save_json_log(mission, decisions, base_filename, extra_meta)
        md_path = self._save_markdown_memo(mission, decisions, base_filename, extra_meta)
        csv_path = self._save_rebalance_csv(mission, decisions, base_filename, extra_meta)

        return {"json": json_path, "md": md_path, "csv": csv_path}

    def _build_base_filename(self, mission: str) -> str:
        """文件名统一构造：timestamp + safe_mission"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_mission = "".join([c for c in mission if c.isalnum() or c in (" ", "_")]).strip()[:20]
        return f"{timestamp}_{safe_mission}"
    
    # ---------------------------------------------------------------------
    # JSON Log (机器可读)
    # ---------------------------------------------------------------------
    def _save_json_log(
        self,
        mission: str,
        decisions: List[EtfDecision],
        filename: str,
        meta: Optional[Dict[str, Any]]
    ) -> str:
        meta_obj = self._build_meta(mission, meta)

        log = DebateLog(
            timestamp=datetime.now().isoformat(),
            meta=meta_obj,
            decisions=decisions,
            visuals=self._collect_visuals(meta_obj)
        )

        path = os.path.join(self.output_dir, f"{filename}_log.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log.model_dump(), f, ensure_ascii=False, indent=2)

        return path

    def _build_meta(self, mission: str, meta: Optional[Dict[str, Any]]) -> DebateMeta:
        """
        将 Core 传来的 extra_meta（dict）转成强类型 DebateMeta。
        Teaching Tip: Core 先只填 ref_date/rounds/stop_reason 也没问题。
        兼容两种输入：
        - 扁平 dict：{"ref_date":..., "rounds":..., ...}
        - 包一层 meta：{"meta": {...}}
        """
        meta = meta or {}
        if "meta" in meta and isinstance(meta["meta"], dict):
            meta = meta["meta"]

        return DebateMeta(
            mission=mission,
            ref_date=meta.get("ref_date"),
            rounds=int(meta.get("rounds", 0) or 0),
            stop_reason=meta.get("stop_reason"),
            tool_trace=meta.get("tool_trace", []) or [],
            dossier_meta=meta.get("dossier_meta", {}) or {},
            extras=meta.get("extras", {}) or {},
        )

    def _collect_visuals(self, meta_obj: DebateMeta) -> List[str]:
        """收集 visuals：meta.extras + tool_trace.visuals（去重保序）"""
        visuals: List[str] = []

        v0 = meta_obj.extras.get("visuals", [])
        if isinstance(v0, list):
            visuals.extend([str(x) for x in v0])
 
        for t in meta_obj.tool_trace:
            visuals.extend([str(x) for x in (t.visuals or [])])

        seen = set()
        out = []
        for x in visuals:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # ---------------------------------------------------------------------
    # Markdown Memo (人可读)
    # ---------------------------------------------------------------------
    def _save_markdown_memo(
        self,
        mission: str,
        decisions: List[EtfDecision],
        filename: str,
        meta: Optional[Dict[str, Any]]
    ) -> str:
        meta_obj = self._build_meta(mission, meta)

        # 统计摘要
        buy = [d for d in decisions if d.action == DecisionAction.BUY and d.weight > 0]
        reject = [d for d in decisions if d.action == DecisionAction.REJECT]
        total_weight = sum(d.weight for d in buy)

        lines: List[str] = []
        lines.append("# 📝 智能决策备忘录 (AI Decision Memo)")
        lines.append(f"**任务指令**: {mission}")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if meta_obj.ref_date:
            lines.append(f"**决策基准日(ref_date)**: {meta_obj.ref_date}")
        if meta_obj.rounds:
            lines.append(f"**辩论轮次**: {meta_obj.rounds}")
        if meta_obj.stop_reason:
            lines.append(f"**停止原因**: {meta_obj.stop_reason}")
        lines.append(f"**组合摘要**: BUY `{len(buy)}` 只 | 总仓位 `{total_weight*100:.1f}%` | REJECT `{len(reject)}` 只\n")

        # 核心表
        lines.append("## 1. 核心决策表")
        table_data: List[Dict[str, str]] = []
        for d in decisions:
            risk_str = "; ".join(d.risk_warnings) if d.risk_warnings else "-"
            reason_str = d.key_reasons[0] if d.key_reasons else "-"
            table_data.append({
                "代码": d.symbol,
                "操作": f"**{d.action.value}**",
                "权重(%)": f"{d.weight*100:.1f}" if d.action == DecisionAction.BUY else "-",
                "得分": f"{d.final_score:.1f}",
                "决策理由": reason_str,
                "风险备注": risk_str
            })

        if table_data:
            df = pd.DataFrame(table_data)
            lines.append(df.to_markdown(index=False))
        else:
            lines.append("*本次无有效决策产出*")

        # 逐标的详情（教学更清晰）
        lines.append("\n## 2. 逐标的决策说明")
        for d in decisions:
            icon = "🟢" if d.action == DecisionAction.BUY else ("🔴" if d.action == DecisionAction.REJECT else "🟡")
            lines.append(f"### {icon} {d.symbol} ({d.action.value})")
            lines.append(f"- **综合得分**: {d.final_score}")
            if d.action == DecisionAction.BUY:
                lines.append(f"- **建议权重**: {d.weight*100:.2f}%")
            lines.append("- **主要理由**:")
            if d.key_reasons:
                for r in d.key_reasons:
                    lines.append(f"  - {r}")
            else:
                lines.append("  - -")
            if d.risk_warnings:
                lines.append("- **⚠️ 风险警告**:")
                for w in d.risk_warnings:
                    lines.append(f"  - {w}")
            lines.append("---")

        # visuals 留痕
        visuals = self._collect_visuals(meta_obj)
        if visuals:
            lines.append("\n## 3. 可视化留痕 (Visuals)")
            for p in visuals:
                lines.append(f"- {p}")

        path = os.path.join(self.output_dir, f"{filename}_memo.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path

    # ---------------------------------------------------------------------
    # Rebalance CSV（调仓指令）
    # ---------------------------------------------------------------------
    def _save_rebalance_csv(
        self,
        mission: str,
        decisions: List[EtfDecision],
        filename: str,
        meta: Optional[Dict[str, Any]]
    ) -> str:
        meta_obj = self._build_meta(mission, meta)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rows = []
        for d in decisions:
            # 只输出可执行指令（BUY/WATCH/REJECT 都可以输出，方便交接）
            reason = d.key_reasons[0] if d.key_reasons else ""
            rows.append({
                "time": ts,
                "date": meta_obj.ref_date or "",
                "code": d.symbol,
                "action": d.action.value,
                "weight": float(d.weight),
                "reason": reason,
            })

        df = pd.DataFrame(rows, columns=["time", "date", "code", "action", "weight", "reason"])
        #df = pd.DataFrame(rows, columns=["date", "code",  "weight"])
        path = os.path.join(self.output_dir, f"{filename}_rebalance.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
