import json
from pathlib import Path

import pytest

from debate_mas.skills.base import SkillContext, BaseSkill, _auto_args_schema_from_execute
from debate_mas.protocol import SkillResult
from debate_mas.skills import registry as reg



def _ctx_stub():
    return SkillContext.model_construct(
        dossier=object(),
        agent_role="hunter",
        ref_date="2025-01-01",
    )


class OkSkill(BaseSkill):
    name = "ok_skill"
    description = "desc"
    expert_mindset = "expert"

    def execute(self, ctx: SkillContext, x: int = 1) -> SkillResult:
        return SkillResult.ok(data={"x": x}, insight="ok")


class BadReturnSkill(BaseSkill):
    name = "bad_return"

    def execute(self, ctx: SkillContext, **kwargs):
        return {"not": "SkillResult"}


class CrashSkill(BaseSkill):
    name = "crash"

    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
        raise RuntimeError("boom")


def test_safe_run_converges_errors_to_skillresult_fail():
    ctx = _ctx_stub()

    out1 = BadReturnSkill().safe_run(ctx)
    assert out1.success is False
    assert "SkillResult" in (out1.error_msg or "")

    out2 = CrashSkill().safe_run(ctx)
    assert out2.success is False
    assert "boom" in (out2.error_msg or "")


def test_to_langchain_tool_returns_json_string():
    ctx = _ctx_stub()
    tool = OkSkill().to_langchain_tool(ctx)

    if hasattr(tool, "invoke"):
        raw = tool.invoke({"x": 7})
    else:
        raw = tool.run({"x": 7})

    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["data"]["x"] == 7


def test_auto_args_schema_from_execute_basic():
    class S(BaseSkill):
        def execute(self, ctx: SkillContext, x: int, y: str = "a") -> SkillResult:
            return SkillResult.ok(data={"x": x, "y": y})

    schema = _auto_args_schema_from_execute(S.execute, model_name="S")
    fields = schema.model_fields

    assert "ctx" not in fields
    assert "x" in fields and "y" in fields
    assert fields["x"].is_required() is True
    assert fields["y"].is_required() is False



def _write_skill_pkg(inventory: Path, *, folder: str, skill_name: str, broken: bool = False):
    skill_dir = inventory / folder
    (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)

    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {skill_name}",
                f"chinese_name: {skill_name}_CN",
                "description: demo desc",
                "---",
                "EXPERT PROMPT HERE",
                "",
            ]
        ),
        encoding="utf-8",
    )

    handler = skill_dir / "scripts" / "handler.py"
    if broken:
        handler.write_text("x = 1\n", encoding="utf-8") 
    else:
        handler.write_text(
            "\n".join(
                [
                    "from debate_mas.skills.base import BaseSkill, SkillContext",
                    "from debate_mas.protocol import SkillResult",
                    "",
                    "class SkillHandler(BaseSkill):",
                    "    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:",
                    "        return SkillResult.ok(data={'ping': 1}, insight='ok')",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def test_registry_loads_good_skill_and_degrades_bad_skill(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    inventory = tmp_path / "inventory"
    inventory.mkdir(parents=True, exist_ok=True)
    _write_skill_pkg(inventory, folder="good_skill", skill_name="good_skill", broken=False)
    _write_skill_pkg(inventory, folder="bad_skill", skill_name="bad_skill", broken=True)

    monkeypatch.setattr(reg, "__file__", str(tmp_path / "registry.py"), raising=False)

    reg._SKILL_CACHE.clear()
    reg.SkillRegistry.load_all_skills(force_reload=True)

    assert "good_skill" in reg._SKILL_CACHE
    s = reg._SKILL_CACHE["good_skill"]
    assert s.name == "good_skill"
    assert s.chinese_name == "good_skill_CN"
    assert s.description == "demo desc"
    assert "EXPERT PROMPT HERE" in (s.expert_mindset or "")



def test_get_skill_autoloads_when_empty(monkeypatch: pytest.MonkeyPatch):
    reg._SKILL_CACHE.clear()
    called = {"n": 0}

    def fake_load_all_skills(force_reload: bool = False):
        called["n"] += 1
        sk = OkSkill()
        sk.name = "x"
        reg._SKILL_CACHE["x"] = sk

    monkeypatch.setattr(reg.SkillRegistry, "load_all_skills", staticmethod(fake_load_all_skills), raising=True)

    s = reg.SkillRegistry.get_skill("x")
    assert called["n"] == 1
    assert s.name == "x"