# skills/registry.py
"""
Layer 3 - 技能注册中心 (The Skill Store)

职责：
- 扫描 skills/inventory/*
- 解析 SKILL.md（YAML frontmatter + prompt）
- 动态加载 scripts/handler.py，实例化 SkillHandler
- 将元信息注入 instance（name/chinese_name/description/expert_mindset）
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import importlib.util
import re
import sys
import yaml

from .base import BaseSkill

_SKILL_CACHE: Dict[str, BaseSkill] = {}

class SkillRegistry:
    #TOOL_RETURN_MODE: str = "dict"
    @staticmethod
    def load_all_skills(force_reload: bool = False) -> None:
        """加载 inventory 下所有技能"""
        if force_reload:
            _SKILL_CACHE.clear()

        current_dir = Path(__file__).parent
        inventory_dir = current_dir / "inventory"
        if not inventory_dir.exists():
            return

        for skill_dir in sorted(inventory_dir.iterdir(), key=lambda p: p.name):
            if (not skill_dir.is_dir()) or skill_dir.name.startswith("__"):
                continue
            try:
                SkillRegistry._load_package(skill_dir)
            except Exception as e:
                print(f"⚠️ [Registry] 加载 '{skill_dir.name}' 失败: {e}")

    @staticmethod
    def _parse_skill_md(content: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """
        解析 SKILL.md（兼容 \n / \r\n）
        约定：
        ---
        <yaml>
        ---
        <prompt>
        """
        # 允许开头有 BOM/空行
        text = content.lstrip("\ufeff").lstrip()

        # 找到首个 frontmatter
        if not text.startswith("---"):
            return None

        # 用正则找到第二个 --- 分隔
        m = re.search(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)$", text, re.DOTALL)
        if not m:
            return None

        yaml_text = m.group(1)
        prompt_text = m.group(2).strip()

        meta = yaml.safe_load(yaml_text) or {}
        if not isinstance(meta, dict):
            return None

        return meta, prompt_text

    @staticmethod
    def _load_package(skill_dir: Path):
        """加载单个 skill 文件夹"""
        md_path = skill_dir / "SKILL.md"
        if not md_path.exists():
            return

        content = md_path.read_text(encoding="utf-8")
        parsed = SkillRegistry._parse_skill_md(content)
        if not parsed:
            print(f"⚠️ [Registry] {skill_dir.name}/SKILL.md 格式错误（缺 frontmatter）")
            return

        meta, prompt_text = parsed
        skill_name = meta.get("name")
        if not skill_name:
            print(f"⚠️ [Registry] {skill_dir.name}/SKILL.md 缺少 name 字段，已跳过")
            return

        py_path = skill_dir / "scripts" / "handler.py"
        if not py_path.exists():
            print(f"⚠️ [Registry] {skill_dir.name} 缺少 scripts/handler.py")
            return

        pkg_name = skill_dir.name
        module_name = f"debate_mas.skills.inventory.{pkg_name}.scripts.handler"

        if str(skill_name) != str(pkg_name):
            print(f"⚠️ [Registry] 提醒：folder='{pkg_name}' 与 SKILL.md name='{skill_name}' 不一致（建议统一）")

        try:
            spec = importlib.util.spec_from_file_location(module_name, py_path)
            if not (spec and spec.loader):
                print(f"⚠️ [Registry] 无法创建 spec: {py_path}")
                return

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "SkillHandler"):
                print(f"⚠️ [Registry] {skill_dir.name} 未找到 SkillHandler 类")
                return

            handler_cls = getattr(module, "SkillHandler")
            instance = handler_cls()

            # 注入元信息（对外 tool 名以 SKILL.md 为准）
            instance.name = str(skill_name)
            instance.chinese_name = str(meta.get("chinese_name", skill_name))
            instance.description = str(meta.get("description", "") or "")
            instance.expert_mindset = prompt_text

            _SKILL_CACHE[str(skill_name)] = instance

        except Exception as e:
            print(f"⚠️ [Registry] 加载 '{skill_name}' 失败: {e} | path={py_path}")

    @staticmethod
    def get_skill(name: str) -> BaseSkill:
        if not _SKILL_CACHE:
            SkillRegistry.load_all_skills()
        skill = _SKILL_CACHE.get(name)
        if not skill:
            raise ValueError(f"❌ 找不到技能 '{name}'")
        return skill