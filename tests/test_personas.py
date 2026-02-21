# tests/test_personas.py
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from debate_mas.core.personas import (
    PromptSlots,
    build_universal_system_prompt,
    build_role_prompts_etf,
    get_auditor_slots,
    get_hunter_slots,
    get_pm_slots,
)


def _dummy_dossier_view() -> Dict[str, Any]:
    """
    personas 用的 frozen_view 假数据：
    - tables 用 list[dict]（更贴近 Dossier.frozen_view 的常见形态）
    - texts 用 list[dict]
    """
    return {
        "mission": "m",
        "meta": {"source_path": "/tmp/demo"},
        "tables": [
            {"name": "prices", "source": "a.csv", "rows": 2, "cols": 3, "columns": ["date", "close"]},
            {"name": "basic", "source": "b.csv", "rows": 1, "cols": 2, "columns": ["code", "name"]},
        ],
        "texts": [
            {"idx": 0, "source": "note.md", "length": 10, "added_at": "2026-01-01T00:00:00"},
        ],
    }


def _allowlist_by_role() -> Dict[str, List[str]]:
    """
    不强绑工具集合；只要能组成白名单即可。
    练习者可以按自己的工程改这里，断言逻辑无需修改。
    """
    return {
        "hunter": ["quantitative_sniper", "theme_miner"],
        "auditor": ["market_sentry", "forensic_detective"],
        "pm": ["allocator"],
    }


def test_promptslots_shape_is_stable() -> None:
    slots = PromptSlots(
        role_name="x",
        role_goal="y",
        role_rules=["r1"],
        tool_policy=["p1"],
        output_type="CANDIDATES",
        output_schema_hint='{"symbol":"510300"}',
        style_guide=["s1"],
        json_only=True,
    )
    assert slots.role_name == "x"
    assert slots.json_only is True
    assert isinstance(slots.role_rules, list)
    assert isinstance(slots.tool_policy, list)


@pytest.mark.parametrize("json_only", [True, False])
def test_build_universal_system_prompt_contains_required_sections(json_only: bool) -> None:
    dv = _dummy_dossier_view()
    slots = PromptSlots(
        role_name="role_x",
        role_goal="do something",
        role_rules=["rule_a"],
        tool_policy=["policy_b"],
        output_type="XTYPE",
        output_schema_hint='{"k":"v"}',
        style_guide=["style_1"],
        json_only=json_only,
    )

    prompt = build_universal_system_prompt(
        mission="MISSION_TEXT",
        dossier_view=dv,
        allowed_tools=["tool_a", "tool_b"],
        slots=slots,
        extra_context="EXTRA_CTX",
    )

    assert isinstance(prompt, str)

    assert "【任务指令】" in prompt
    assert "MISSION_TEXT" in prompt

    assert "【数据证据摘要（只读）】" in prompt
    assert "表格数量" in prompt
    assert "文本数量" in prompt

    assert "【你的角色】" in prompt
    assert "角色名" in prompt
    assert "role_x" in prompt

    assert "【工具权限（白名单）】" in prompt
    assert "tool_a" in prompt and "tool_b" in prompt

    assert "【角色规则（必须遵守）】" in prompt
    assert "【工具使用政策（必须遵守）】" in prompt
    assert "白名单" in prompt
    assert "禁止" in prompt

    assert "【输出格式】" in prompt
    if json_only:
        assert "你只输出一个 JSON 对象" in prompt
        assert "type: 固定为 XTYPE" in prompt
    else:
        assert "你必须输出两段内容" in prompt
        assert "Final JSON" in prompt
        assert "type: 固定为 XTYPE" in prompt

    assert "【表达风格】" in prompt
    assert "style_1" in prompt
    assert "【补充上下文】" in prompt
    assert "EXTRA_CTX" in prompt


def test_get_slots_functions_return_promptslots() -> None:
    hunter = get_hunter_slots()
    auditor = get_auditor_slots()
    pm = get_pm_slots()

    assert isinstance(hunter, PromptSlots)
    assert isinstance(auditor, PromptSlots)
    assert isinstance(pm, PromptSlots)

    assert hunter.role_name and hunter.role_goal and hunter.output_type
    assert auditor.role_name and auditor.role_goal and auditor.output_type
    assert pm.role_name and pm.role_goal and pm.output_type


def test_build_role_prompts_etf_returns_three_role_prompts_and_injects_allowlist() -> None:
    dv = _dummy_dossier_view()
    allowlist = _allowlist_by_role()

    prompts = build_role_prompts_etf(
        mission="M",
        dossier_view=dv,
        allowlist_by_role=allowlist,
    )

    assert isinstance(prompts, dict)

    for role in ["hunter", "auditor", "pm"]:
        assert role in prompts
        assert isinstance(prompts[role], str)
        assert len(prompts[role]) > 50
        assert "【输出格式】" in prompts[role]

    for role, tools in allowlist.items():
        for t in tools:
            assert t in prompts[role]
