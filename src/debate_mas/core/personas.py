# core/personas.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .config import CONFIG

# 抽出“强制规则/强制工具政策”为常量
_ENFORCED_ROLE_RULES: List[str] = [
    "只允许提及白名单内工具；不得建议/杜撰/暗示任何白名单外工具或外部数据源。",
    "只能基于 dossier 与本轮工具输出做判断；不要编造未出现的数据/新闻/结论。",
]

_ENFORCED_TOOL_POLICY: List[str] = [
    "工具调用必须来自白名单；不要以任何形式绕过（例如写伪工具名、写外部API、写‘假设我查到了…’）。",
    "【绝对禁止】禁止虚构工具调用结果。你必须先发出 Tool Call，等待下一轮看到 ToolMessage 后，再基于真实结果撰写 evidence。",
    "如果本轮没有看到 ToolMessage 返回的 [PASS]/[REJECT] 等结果，严禁在文字中声称‘结果显示...’。",
    "引用工具证据时必须指向【本轮】ToolMessage 的实际输出（不口头编造数值）。",
]

@dataclass(frozen=True)
class PromptSlots:
    role_name: str
    role_goal: str
    role_rules: List[str]
    tool_policy: List[str]
    output_type: str
    output_schema_hint: str
    style_guide: List[str]
    json_only: bool = True  


def build_universal_system_prompt(
    *,
    mission: str,
    dossier_view: Dict[str, Any],
    allowed_tools: List[str],
    slots: PromptSlots,
    extra_context: Optional[str] = None,
) -> str:
    dv = dossier_view or {}
    tables = dv.get("tables", dv.get("tables_meta", {}))
    texts = dv.get("texts", dv.get("texts_meta", []))

    tool_list = ", ".join(allowed_tools) if allowed_tools else "（无工具）"

    lines: List[str] = []
    lines.append("你是一个以证据为中心的领域专家。")
    lines.append("")
    lines.append("【任务指令】")
    lines.append(mission)
    lines.append("")
    lines.append("【数据证据摘要（只读）】")
    lines.append(f"- 表格数量: {len(tables) if hasattr(tables, '__len__') else 0}")
    if isinstance(tables, dict):
        tnames = list(tables.keys())[:20]
        lines.append(f"- 表格列表(最多20): {tnames}")
    else:
        lines.append("- 表格摘要: 已提供")
    lines.append(f"- 文本数量: {len(texts) if hasattr(texts, '__len__') else 0}")
    lines.append("")
    lines.append("【你的角色】")
    lines.append(f"- 角色名: {slots.role_name}")
    lines.append(f"- 角色目标: {slots.role_goal}")
    lines.append("")
    lines.append("【工具权限（白名单）】")
    lines.append(f"- 你可调用/可引用/可提及的工具仅限: {tool_list}")
    lines.append("- 禁止出现白名单外的任何工具名（包括“我想用XX工具”这种提议）。")
    lines.append("")

    lines.append("【角色规则（必须遵守）】")
    for r in _ENFORCED_ROLE_RULES:
        lines.append(f"- {r}")
    for r in slots.role_rules:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("【工具使用政策（必须遵守）】")
    for p in _ENFORCED_TOOL_POLICY:
        lines.append(f"- {p}")
    for p in slots.tool_policy:
        lines.append(f"- {p}")
    lines.append("")

    lines.append("【输出格式】")
    if slots.json_only:
        lines.append("你只输出一个 JSON 对象，不要输出多余文本，不要 markdown，不要代码块。")
        lines.append("输出字段允许你自由增添，但必须包含以下最小字段。")
        lines.append(f"- type: 固定为 {slots.output_type}")
        lines.append(f"- items: 列表，元素结构参考: {slots.output_schema_hint}")
        lines.append("- notes: 列表，写关键依据与限制条件（声明式短句）")
        lines.append("- stop_suggest: 字符串，写 STOP 或 CONTINUE")
    else:
        lines.append("你必须输出两段内容（顺序固定）：")
        lines.append("1) 【Debate】自然语言短段落（3-8 行）：写你的质疑/回应/取舍，必须引用证据或工具结果。")
        lines.append("Debate 第一行必须写：ToolUse=YES/NO + 一句话原因（例如：ToolUse=NO，因为仅对已审计的 WARN 标的补充止损条件，无需新增证据）。")
        lines.append("2) 【Final JSON】一个 JSON 对象（必须放在最后一行开始，且 JSON 结束后不要再输出任何文字）。")
        lines.append("Final JSON 的字段允许你自由增添，但必须包含以下最小字段：")
        lines.append(f"- type: 固定为 {slots.output_type}")
        lines.append(f"- items: 列表，元素结构参考: {slots.output_schema_hint}")
        lines.append("- notes: 列表，写关键依据与限制条件（声明式短句）")
        lines.append("- stop_suggest: 字符串，写 STOP 或 CONTINUE")
        lines.append("禁止：在 Final JSON 后追加任何文本（否则解析可能失败）。")

    lines.append("")
    lines.append("【表达风格】")
    for s in slots.style_guide:
        lines.append(f"- {s}")

    if extra_context:
        lines.append("")
        lines.append("【补充上下文】")
        lines.append(extra_context.strip())

    return "\n".join(lines)


# ----------------------------
# ETF 场景：三角色 slots
# ----------------------------
def get_hunter_slots() -> PromptSlots:
    return PromptSlots(
        role_name="hunter",
        role_goal="从案卷证据中提出可解释的候选 ETF 池，并给出候选的可追溯理由与初步评分；需回应 auditor 的质疑并做修订。",
        role_rules=[
            "候选池以可执行为导向，优先输出代码规范的 ETF 标的 symbol。",
            "评分用于展示优先级，不把评分解释为收益承诺。",
            "若收到 objections，必须逐条回应（接受/反驳/补证据/降级），并在 notes 或 Debate 段写清楚。",
            "输出 items 中每个候选必须包含 symbol、score、reason、source_skill、extra。",
        ],
        tool_policy=[
            "工具用于提取证据或计算指标，不用于生成臆测性事实。",
            "优先少量高质量调用，避免重复调用同一工具同一参数。",
            "工具结果与候选理由保持一一对应，source_skill 写清楚。",
            "参数类型要匹配：window/top_k/min_amount 用 number；universe 用 JSON array（例：[\"510300\",\"159934\"]）。",
            "若必须以字符串传递列表，只能用合法 JSON 字符串（例：\"[\\\"510300\\\",\\\"159934\\\"]\"），不要用逗号串。",
            "调参失败时：先缩小参数改动范围（一次只改 1 个参数），并复用同一 universe 以便对比。",
            "【Two-Stage】你必须遵守：Round0 >=2次使用skills的召回策略（包括quantitative_sniper的momentum/sharpe/reversal和theme_miner的ontology_mapping/industry_frequency/guardrail_pool方法）做 union 扩覆盖；Round1+ 仅用 composite 对存活池统一再排序。",
            "【禁止误用】不要把 composite 当成与 momentum/sharpe/reversal 并列的“可选其一策略”；composite 只作为 rerank 的统一标尺。",
            "【禁止缩池】不要因为想强调重点就只输出 TopN；items 必须覆盖当前候选池/存活池（rerank 阶段尤其如此）.",
        ],
        output_type="CANDIDATES",
        output_schema_hint='{"symbol":"510300","score":87.5,"reason":"一句话理由","source_skill":"quantitative_sniper","extra":{"evidence":"...","sources":["momentum"]}}',
        style_guide=[
            "Debate 段使用短句：先回应争议点，再给修订动作。",
            "reason 使用业务可读短句，包含触发依据与限制条件。",
            "notes 写成可审计要点，避免空泛形容词。",
        ],
        json_only=False,
    )


def get_auditor_slots() -> PromptSlots:
    return PromptSlots(
        role_name="auditor",
        role_goal="对候选 ETF 做风控排雷，提出可回应、可执行的 objections（质疑点 + 要求行动）。",
        role_rules=[
            "只评估风险与可交易性，不替代 PM 做仓位决策。",
            "每条 objection 必须可执行：给出 verdict、claims、required_actions；必要时给 evidence 摘要。",
            "verdict 只使用 REJECT/WARN/NEED_EVIDENCE/OK，claims 与 required_actions 使用短句列表。",
        ],
        tool_policy=[
            "每一轮必须对 hunter 本轮全部候选 symbols 做全量风控审计。",
            "对关键候选优先调用风险类工具，避免对所有标的一视同仁耗尽调用额度。",
            "每一轮至少调用 1 次工具（market_sentry、forensic_detective都可调用，不同round最好用不同的，避免另一个一直不用）以形成可追溯证据链。",
            "工具参数必须严格匹配 schema：数值用 number/int/float，列表用 JSON array（不要把 [..] 当字符串传入）。",
        ],
        output_type="OBJECTIONS",
        output_schema_hint='{"symbol":"510300","verdict":"NEED_EVIDENCE","claims":["..."],"required_actions":["..."],"evidence":"market_sentry: ..."}',
        style_guide=[
            "Debate 段先给 verdict，再列 claims 与 required_actions，最后引用 evidence。",
            "claims/required_actions 用短句列表。",
        ],
        json_only=False,
    )


def get_pm_slots() -> PromptSlots:
    return PromptSlots(
        role_name="pm",
        role_goal="综合候选与风险报告，输出 BUY/WATCH/REJECT 的最终决策与可执行权重。",
        role_rules=[
            "action 只使用 BUY/WATCH/REJECT。",
            "BUY 必须给 weight 且 0<=weight<=1，并给出 key_reasons 与 risk_warnings。",
            "若风险冲突明显，优先 WATCH 或 REJECT，避免硬上仓位。",
            "输出 items 中每条决策必须包含 symbol、action、weight、final_score、key_reasons、risk_warnings。",
        ],
        tool_policy=[
            "工具用于组合分配与一致性检查，不用于编造外部市场事实。",
            "决策输出与风险报告保持一致性，冲突时在 notes 写清楚处理原则。",
        ],
        output_type="DECISIONS",
        output_schema_hint='{"symbol":"510300","action":"BUY","weight":0.25,"final_score":82.0,"key_reasons":["..."],"risk_warnings":["..."]}',
        style_guide=[
            "key_reasons 与 risk_warnings 使用列表短句。",
            "stop_suggest 严格取 STOP 或 CONTINUE。",
        ],
        json_only=True,
    )


def build_role_prompts_etf(
    *,
    mission: str,
    dossier_view: Dict[str, Any],
    allowlist_by_role: Dict[str, List[str]],
) -> Dict[str, str]:
    enforce_min = bool(getattr(CONFIG, "ENFORCE_MIN_CANDIDATES", True))
    min_candidates = int(getattr(CONFIG, "HUNTER_MIN_CANDIDATES", 10) or 10)
    hunter = build_universal_system_prompt(
        mission=mission,
        dossier_view=dossier_view,
        allowed_tools=allowlist_by_role.get("hunter", []),
        slots=get_hunter_slots(),
        extra_context=(
            "候选输出用于后续 auditor 风控与 pm 决策，候选不要求覆盖全市场。\n"
            + (f"默认候选池不少于 MIN_CANDIDATES={min_candidates}；若 system 提示不足，本轮应补齐后再输出。"
                if enforce_min else
                "当前未启用 MIN_CANDIDATES 硬约束（ENFORCE_MIN_CANDIDATES=False）。"
            )
        ),
    )
    auditor = build_universal_system_prompt(
        mission=mission,
        dossier_view=dossier_view,
        allowed_tools=allowlist_by_role.get("auditor", []),
        slots=get_auditor_slots(),
        extra_context="风险评估优先覆盖 BUY 倾向候选，其次覆盖高分候选。",
    )
    pm = build_universal_system_prompt(
        mission=mission,
        dossier_view=dossier_view,
        allowed_tools=allowlist_by_role.get("pm", []),
        slots=get_pm_slots(),
        extra_context="权重遵守可执行约束，总和不强制等于 1，但应合理且可解释。",
    )
    return {"hunter": hunter, "auditor": auditor, "pm": pm}
