from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from debate_mas.protocol.schema import (
    DebateMeta,
    DecisionAction,
    SkillResult,
    DebateLog,
    EtfDecision,  # 这里是“示例业务对象”；如果你换了业务对象，可在此处替换 import
)
from debate_mas.protocol.renderer import DebateRenderer



def test_decision_action_enum_is_stable() -> None:
    # "BUY", "WATCH", "REJECT"可视情况改成自定义string枚举值，但枚举成员名不可改动
    assert {x.value for x in DecisionAction} >= {"BUY", "WATCH", "REJECT"}


def test_skill_result_contract() -> None:
    ok = SkillResult.ok(data={"x": 1}, insight="hi")
    assert ok.success is True
    assert ok.insight == "hi"
    assert isinstance(ok.visuals, list)
    assert ok.error_msg is None

    bad = SkillResult.fail("boom", data={"debug": True})
    assert bad.success is False
    assert bad.error_msg == "boom"
    assert isinstance(bad.visuals, list)
    assert "boom" in bad.insight


def test_debate_log_is_json_serializable() -> None:
    log = DebateLog(
        timestamp="2026-01-01T00:00:00",
        meta=DebateMeta(mission="x", rounds=1),
        decisions=[],
        visuals=[],
    )
    payload = log.model_dump()
    json.dumps(payload, ensure_ascii=False) 


def test_renderer_generates_three_artifacts(tmp_path: Path) -> None:
    renderer = DebateRenderer(output_dir=str(tmp_path))

    decisions = [
        EtfDecision(
            symbol="X1",
            action=DecisionAction.BUY,
            weight=0.2,
            final_score=12.3,
            key_reasons=["r1"],
            risk_warnings=["w1"],
        ),
        EtfDecision(
            symbol="X2",
            action=DecisionAction.WATCH,
            weight=0.0,
            final_score=9.9,
            key_reasons=[],
            risk_warnings=[],
        ),
    ]

    extra_meta: Dict[str, Any] = {
        "ref_date": "2025-10-26",
        "rounds": 2,
        "stop_reason": "ok",
        # tool_trace/extras/dossier_meta 都是“允许存在”的扩展位，不要求业务必须写
        "tool_trace": [{"tool": "dummy", "ok": True, "visuals": ["v1.png"]}],
        "extras": {"visuals": ["v0.png"]},
        "dossier_meta": {"source": "unit-test"},
    }

    paths = renderer.render("unit test mission", decisions, extra_meta=extra_meta)

    for k in ["json", "md", "csv"]:
        assert k in paths
        assert Path(paths[k]).exists(), f"missing artifact: {k}"

    with open(paths["json"], "r", encoding="utf-8") as f:
        j = json.load(f)

    assert isinstance(j, dict)
    assert "timestamp" in j
    assert "meta" in j and isinstance(j["meta"], dict)
    assert "decisions" in j and isinstance(j["decisions"], list)

    # 注意：如果你的 renderer/schema 选择不输出其中某些字段，可按业务删改这些断言
    assert j["meta"].get("mission") == "unit test mission"
    assert j["meta"].get("ref_date") == "2025-10-26"
    assert j["meta"].get("rounds") == 2
    assert j["meta"].get("stop_reason") == "ok"

    # 这是“建议项”：如果你不打算做 visuals 聚合，可把这一段删掉
    if "visuals" in j and isinstance(j["visuals"], list):
        assert "v0.png" in j["visuals"]
        assert "v1.png" in j["visuals"]

    md_text = Path(paths["md"]).read_text(encoding="utf-8")
    assert len(md_text.strip()) > 0

    # 不强绑列名：因为列集合由业务决定（date/code/weight 或 time/date/code/action/... 都允许）
    df = pd.read_csv(paths["csv"])
    assert len(df) == len(decisions)

    # ============================================================
    # B) 选改（业务相关）— 你可以根据业务自定义“更严格的断言”
    # ============================================================
    # 如果你们希望把 CSV 当成“调仓指令契约”，可以在这里固定列集合。
    #
    # 示例 1：ETF 调仓版（更严格，但会限制业务自由）
    required_cols = ["time", "date", "code", "action", "weight", "reason"]
    for c in df.columns:
        if c not in required_cols:
            print(f"unexpected column in CSV: {c}")
        assert c in required_cols, f"csv has unexpected column: {c}"
    #
    # 示例 2：极简金融决策版（只要 date/code/weight）
    # required_cols = ["date", "code", "weight"]
    # for c in required_cols:
    #     assert c in df.columns, f"csv missing column: {c}"
    #
    # 建议：Checkpoint 测试默认保持“宽松”，把严格契约留给你们自己的业务测试。
