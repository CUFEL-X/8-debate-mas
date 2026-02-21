# 训练手册：Debate MAS 三段式实战练习（Build → Transfer）

> 本训练册采用 **Outcome → Contract → TODO Map → Run → Pass → Transfer** 的关卡式结构。  
> 我们以 **ETF 投决**作为示例背景，带你“填空式”搭建一个最小可用的 **Debate MAS**；每关末尾提供 **迁移提示模板**，帮助你把同一套框架迁移到任意业务场景（无需真的实现另一套代码）。

---

## 开始之前（读完 60 秒就够）

### 你会得到什么
- ✅ 能跑：从命令行跑出一套“可审计”的决策产物（memo / csv / log / transcript）
- ✅ 能改：知道改哪些文件会影响“流程/停机/输出”，不会乱改
- ✅ 能迁移：换业务时知道“框架不动，业务替换在哪里做”

### 每关怎么做（六步闭环）
每一关都固定按这 6 步走（你会很快形成肌肉记忆）：
- 🎯 **Outcome**：这一关做完，你能得到什么结果
- 🧱 **Contract**：你必须满足的输入/输出约束
- 🗺️ **TODO Map**：这一关要改的文件/函数
- ▶️ **Run**：运行命令
- ✅ **Pass**：通过标准
- 🔁 **Transfer**：迁移提示（哪些不动 / 哪些可替换 / 可替换方向）


---

# 练习路线（进度清单）

> 完成后可以把 ⬜️ 改成 ✅ 来打卡。

| 阶段 | 关卡 | 你会完成什么 | 状态 |
|---|---|---|---|
| 阶段一：感知与跑通 | 关卡-01 | 跑通默认 Demo + 命令对比，找到产物并会读 | ⬜️ |
| 阶段二：核心闭环 | 关卡-02 | 状态账本：payload / transcript / round / stop_reason | ⬜️ |
| 阶段二：核心闭环 | 关卡-03 | 输出协议：Candidate / Objection / Decision（结构化） | ⬜️ |
| 阶段二：核心闭环 | 关卡-04 | 证据案卷：Loader 最小契约（folder → dossier） | ⬜️ |
| 阶段二：核心闭环 | 关卡-05 | 提示词工厂：Personas + 工具白名单 + 输出格式约束 | ⬜️ |
| 阶段二：核心闭环 | 关卡-06 | 流程编排：Graph 跳转 + 停机规则 | ⬜️ |
| 阶段二：核心闭环 | 关卡-07 | 引擎串联：Engine 最小循环（能跑完） | ⬜️ |
| 阶段三：Skills | 关卡-08 | 写 Skill：一个可调用、可返回结构化结果的技能 | ⬜️ |
| 阶段三：Skills | 关卡-09 | 注册与准入：registry + allowlist，系统能识别并筛选可用 skill | ⬜️ |
| 阶段三：Skills | 关卡-10 | 工具封装与守卫：skill→tools 映射 + 统一调用入口 + allowlist 拦截 | ⬜️ |

---


# 阶段一：感知与跑通（Run & Sense）

## 关卡-01｜跑通默认 Demo + 命令对比

<details>
<summary><b>Checkpoint 01 — 跑通默认 【详情】</b></summary>

### 🎯 目标收获（Outcome）
> 能跑通系统，并通过换一次任务指令完成对比：结论为什么变、风险点怎么跟着变。
> 
> 能找到并读懂本次运行产物，知道哪些是给人看的，哪些是给程序用的。

### 🧱 约束契约（Contract）
- 不修改任何 `.py` 文件，只允许修改 `.env` 或命令行参数。

### 🗺️ 任务清单（TODO Map）
**必看**
- `src/debate_mas/main.py`：入口参数与默认值  
- `src/debate_mas/protocol/renderer.py`：会生成哪些产物  
- `output_reports/*_memo.md`：决策摘要  
- `output_reports/*_rebalance.csv`：结构化指令  
- `output_reports/*_log.json`：运行日志与摘要信息  

**可选**
- `src/debate_mas/loader/dossier.py`：案卷对象是什么形状  
- `src/debate_mas/protocol/schema.py`：输出协议有哪些字段  

### ▶️ 执行命令（Run）

1) 默认 Demo：
```bash
python -m debate_mas
```

2) 换任务指令，对比一次输出差异：
```bash
python -m debate_mas --mission "分析当前黄金、有色金属ETF的投资机会"
```

3) 可选：顺手验证日期与输出目录也可控：
```bash
python -m debate_mas --date "2025-06-26" --output_dir "./output_reports_stage1"
```

**观察要点（看到即通过）**
- 终端出现 `🟦 VERBOSE_MODE=summary` 开头与 `🟦 VERBOSE END` 结尾  
- Hunter 出现 `🛠️ Round 1 | Role=hunter`，并能看到工具调用（常见：`momentum / sharpe / reversal`）  
- Auditor 至少出现一次 `🛠️ Round ? | Role=auditor`  
- 若出现 `__rerank_cutoff__`，表示候选池做了 TopN 截断  
- 结束后看到 `✅ 产物已生成`，并打印出产物路径或文件名
- 在 `output_reports/` 目录下确实生成了新文件组 

### ✅ 验收标准（Pass）
- 两次运行都生成新产物文件组，至少包含 `*_memo.md`、`*_rebalance.csv`、`*_log.json`  
- 你能指出两次 `memo.md` 的 BUY/WATCH/REJECT 或权重是否变化  
- 你能指出 Auditor 的风险点是否更贴近任务主题，而不是停留在通用风险  

### 🔁 可迁移点（Transfer）

**1. 不要动**
- 入口命令结构、三角色轮转、产物落盘与审计留痕  
- 本关不改任何 `.py` 文件与既有规则

**2. 可替换（仅通过命令行）**
- `--mission`：任务文本（同一套框架在不同主题下会得出不同结论）
- `--date`：基准日期（同一套数据在不同日期下结论会变化）
- `--output_dir`：输出目录（产物落盘位置可控）
- `--folder`：仅在“数据字段/表名契约一致”的前提下可替换数据目录（可选）

**‼️迁移时的“只改哪里”口诀**
- 本关只做“换任务文本/日期/输出目录”的迁移体验；真正的“换材料契约与输出结构”会在后续关卡展开

</details>

---

# 阶段二：核心闭环 (Core Loop)

## 关卡-02｜状态账本 State：payload / transcript / round / stop_reason

<details>
<summary><b>Checkpoint 02 — 状态账本 State 【详情】</b></summary>

> **payload**对应 state 里 **“要被读写的字段集合”**（含 mission/ref_date/dossier_view、四个产物插槽、stop_reason、tool_trace 等）
>
> **transcript**对应`messages`（对话历史）+ `history`（结构化过程历史）等

### 🎯 目标收获 Outcome
- 理解并实现“共享账本”这件事：多角色不直接通信，只通过 `state` 共享事实与产物  
- 跑通最小状态流转：初始化一轮干净状态、推进轮次、写入候选与质疑、记录停机原因  
- 明确哪些字段是框架通用的，哪些是本项目为 ETF 例子扩展出来的

### 🧱 约束契约 Contract
- 本关只改 `src/debate_mas/core/state.py`  
- 不改 graph、engine、skills 的逻辑，不引入新依赖  
- 目标不是“更聪明”，而是稳定、可追溯、好测试

### 🗺️ 任务清单（TODO Map）
**必看**
- `src/debate_mas/core/state.py`：本关主文件  
- `src/debate_mas/core/config.py`：理解 `MAX_ROUNDS` 与工具治理、跨轮控制开关的来源  
- `src/debate_mas/loader/dossier.py`：确认 `Dossier` 的形状，以及 `frozen_view()` 的用途  

**必写**
- `DebateState`：字段分组要清楚，至少让这些字段稳定存在  
  - `mission / ref_date / dossier / dossier_view`  
  - `messages / round_idx`  
  - `candidates_cur / objections_cur / diff_cur / decisions_cur`  
  - `stop_reason`  
  - `history` 作为可复盘的结构化历史容器  
- `init_state()`：必须返回第一轮干净状态，避免 engine/graph 读字段时 KeyError  
- `reset_round_runtime()` 与 `bump_round()`：保证每轮 runtime 重置可用  
- `push_candidates / push_objections / push_diff / push_decisions`：同时写入 cur 与 history  

<details>
<summary><b> 📄 Checkpoint-02：state.py 练习骨架</b></summary>

```py
# src/debate_mas/core/state.py 
from __future__ import annotations

import json
import hashlib
from typing import Any, Dict, List, Optional, Set, TypedDict

from langchain_core.messages import BaseMessage
from debate_mas.loader.dossier import Dossier

# ============================================================
# 0) 小工具：history 与“稳定性指纹”
# ============================================================
# 标记说明
# - 必写（框架通用）：跑通任何带 skills 的 Debate MAS 都必须有
# - 建议做（框架通用）：提升收敛与可测试性，但不做也能跑
# - 后续再讲（扩展位）：本关只要初始化合理，不影响最小闭环
# - ETF相关（任务扩展）：与 ETF 投决背景强绑定，迁移到别的任务可替换或删减

_HISTORY_DEFAULT: Dict[str, List[Any]] = {
    "candidates": [],
    "objections": [],
    "diffs": [],
    "decisions": [],
}


def _ensure_history(st: "DebateState") -> None:
    """
    统一 history 初始化，避免每个 push_* 都写一遍 setdefault。

    Args:
        st: DebateState，当前账本

    Returns:
        None
    """
    # TODO【必写（框架通用）】:
    # 1) 确保 st["history"] 存在
    # 2) 确保 history 包含四个 key：candidates / objections / diffs / decisions
    # 3) 每个 key 的 value 必须是“新的 list 对象”（不要复用 _HISTORY_DEFAULT 里的 list）
    raise NotImplementedError


def _stable_dumps(obj: Any) -> str:
    """
    把对象稳定序列化成字符串，用于计算指纹。

    Args:
        obj: 任意可序列化对象

    Returns:
        稳定的 JSON 字符串
    """
    # TODO【建议做（框架通用）】:
    # - 优先 json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # - 若失败（不可序列化），退化为 json.dumps(str(obj), ensure_ascii=False)
    raise NotImplementedError


def _fp(obj: Any) -> str:
    """
    生成指纹，用于判断跨轮是否“产物没变”。

    Args:
        obj: 任意对象

    Returns:
        sha1 十六进制字符串
    """
    # TODO【建议做（框架通用）】:
    # return hashlib.sha1(_stable_dumps(obj).encode("utf-8")).hexdigest()
    raise NotImplementedError


# ============================================================
# 1) 账本结构 DebateState
# ============================================================

class DebateState(TypedDict, total=False):
    # ---------- 输入 ----------
    # TODO【必写（框架通用）】
    mission: str
    ref_date: Optional[str]
    dossier: Dossier
    dossier_view: Dict[str, Any]

    # ---------- 对话与轮次 ----------
    # TODO【必写（框架通用）】
    messages: List[BaseMessage]
    round_idx: int

    # ---------- “最新版”结构化产物 ----------
    # TODO【必写（框架通用）】
    candidates_cur: List[Dict[str, Any]]
    objections_cur: List[Dict[str, Any]]
    diff_cur: Dict[str, Any]
    decisions_cur: List[Dict[str, Any]]

    # ---------- 历史容器 ----------
    # TODO【必写（框架通用）】
    history: Dict[str, List[Any]]

    # ---------- 停机与审计 ----------
    # TODO【必写（框架通用）】
    stop_reason: Optional[str]
    tool_trace: List[Dict[str, Any]]

    # ---------- 运行期辅助 ----------
    # TODO【必写（框架通用）】：本关只要能初始化、不 KeyError
    stable_rounds: int
    phase: str
    artifacts: Optional[Dict[str, str]]
    tool_cache: Dict[str, Any]

    # ---------- 兼容旧字段 ----------
    # TODO【后续再讲（扩展位）】：本关只初始化，不要求理解
    candidates: List[Dict[str, Any]]
    risk_reports: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]

    # ---------- 角色建议与说话人 ----------
    # TODO【后续再讲（扩展位）】：本关只初始化
    hunter_stop_suggest: str
    auditor_stop_suggest: str
    pm_stop_suggest: str
    _last_speaker_role: str

    # ---------- Tool Guard：每轮重置 ----------
    # TODO【必写（框架通用）】：reset_round_runtime 要保证这些字段每轮可用
    _round_tool_calls: Dict[str, int]
    _round_tool_calls_ok: Dict[str, int]
    _round_fingerprints: Set[str]
    _round_guard_denied: bool
    _round_missing_evidence: bool

    # ---------- 后续关卡的跨轮控制位 ----------
    # TODO【后续再讲（扩展位）】：本关只初始化
    _need_evidence: bool
    _need_evidence_symbols: List[str]
    _need_evidence_actions: List[str]

    _need_more_candidates: bool
    _need_more_candidates_min: int
    _need_more_candidates_have: int
    _need_more_candidates_missing: int
    _need_more_candidates_reason: str

    _last_stable_fp: str
    _force_hunter_tool: bool

    # ---------- ETF相关：两阶段 pipeline ----------
    # TODO【ETF相关（任务扩展）】：迁移到别的任务可替换或删减
    _hunter_pipeline_stage: str
    survivor_universe: List[str]
    _need_recall_diversity: bool
    _need_recall_diversity_reason: str
    _need_rerank_composite: bool
    _need_rerank_composite_reason: str
    _hunter_round_sniper_strategies: List[str]


# ============================================================
# 2) 初始化与每轮 runtime 重置
# ============================================================

def init_state(
    mission: str,
    dossier: Dossier,
    ref_date: Optional[str] = None,
    messages: Optional[List[BaseMessage]] = None,
) -> DebateState:
    """
    返回第一轮干净状态，避免 engine/graph 读字段时 KeyError。

    Args:
        mission: 本次任务文本
        dossier: 案卷对象
        ref_date: 基准日期，可选
        messages: 初始对话，可选

    Returns:
        st: DebateState
    """
    st: DebateState = {}

    # TODO【必写（框架通用）】核心字段初始化
    # - mission / ref_date / dossier / dossier_view
    # - messages / round_idx
    # - candidates_cur / objections_cur / diff_cur / decisions_cur
    # - history / stop_reason / tool_trace
    # - stable_rounds / phase / artifacts / tool_cache

    # TODO【必写（框架通用）】dossier_view
    # - 若 dossier 有 frozen_view()：使用它
    # - 否则：使用 {}

    # TODO【后续再讲（扩展位）】其余状态位：给合理默认值即可

    # TODO【必写（框架通用）】初始化本轮 runtime
    reset_round_runtime(st)
    return st


def reset_round_runtime(st: DebateState) -> None:
    """
    每轮 runtime 重置：工具调用次数、去重指纹、guard 状态等。

    Args:
        st: DebateState

    Returns:
        None
    """
    # TODO【必写（框架通用）】至少保证这些字段每轮可用
    # st["_round_tool_calls"] = {"hunter": 0, "auditor": 0, "pm": 0}
    # st["_round_tool_calls_ok"] = {"hunter": 0, "auditor": 0, "pm": 0}
    # st["_round_fingerprints"] = set()
    # st["_round_guard_denied"] = False
    # st["_round_missing_evidence"] = False
    raise NotImplementedError


def bump_round(st: DebateState) -> None:
    """
    轮次推进：round_idx += 1，并重置每轮 runtime。

    Args:
        st: DebateState

    Returns:
        None
    """
    # TODO【必写（框架通用）】
    # 1) st["round_idx"] += 1（注意 int 化）
    # 2) reset_round_runtime(st)
    raise NotImplementedError


# ============================================================
# 3) push 系列：把结构化产物落到账本
# ============================================================

def push_candidates(st: DebateState, items: List[Dict[str, Any]]) -> None:
    """
    写入本轮候选，并追加到 history。

    Args:
        st: DebateState
        items: 候选列表

    Returns:
        None
    """
    # TODO【必写（框架通用）】
    # 1) st["candidates_cur"] = list(items or [])
    # 2) 兼容字段：st["candidates"] = st["candidates_cur"]（若你们还在用）
    # 3) _ensure_history(st)
    # 4) st["history"]["candidates"].append({"round": round_idx, "items": st["candidates_cur"]})
    raise NotImplementedError


def push_objections(st: DebateState, items: List[Dict[str, Any]]) -> None:
    """
    写入本轮质疑点，并追加到 history。
    """
    # TODO【必写（框架通用）】同上：写 objections_cur + append history
    raise NotImplementedError


def push_diff(st: DebateState, diff_obj: Dict[str, Any]) -> None:
    """
    写入本轮 diff，并追加到 history。
    """
    # TODO【必写（框架通用）】写 diff_cur + append history
    raise NotImplementedError


def push_decisions(st: DebateState, items: List[Dict[str, Any]]) -> None:
    """
    写入本轮裁决，并追加到 history。
    """
    # TODO【必写（框架通用）】写 decisions_cur + append history（可同步兼容字段 decisions）
    raise NotImplementedError


# ============================================================
# 4) 建议增强：候选合并与收敛
# ============================================================

def push_candidates_merge(st: DebateState, incoming: List[Dict[str, Any]]) -> None:
    """
    候选合并策略：只补充/修订，不偷偷删池子。

    说明：
        - 这是“框架通用的产物合并思路”，但这里的 symbol/score 更贴近 ETF 候选
        - 迁移到别的任务时，你可以把 symbol 改成 id，把 score 改成 priority

    TODO【建议做（框架通用）】:
        1) prev_items = st.get("candidates_cur", []) or []
        2) 以 key（如 symbol）合并：incoming 覆盖同 key
        3) 按 score 由高到低排序，次序用 key 稳定化
        4) 写回 candidates_cur，并写入 history

    Args:
        st: DebateState
        incoming: 新增或修订的候选列表

    Returns:
        None
    """
    raise NotImplementedError


def bump_stable_rounds(st: DebateState, *, reset_if_changed: bool = True) -> int:
    """
    用 candidates + objections + diff 生成指纹；若不变则 stable_rounds += 1。

    TODO【建议做（框架通用）】:
        1) cur = {"candidates": ..., "objections": ..., "diff": ...}
        2) cur_fp = _fp(cur)
        3) prev_fp = st.get("_last_stable_fp", "")
        4) 若相等：stable_rounds += 1
        5) 若不等：按 reset_if_changed 重置，并更新 _last_stable_fp
        6) return stable_rounds

    Args:
        st: DebateState
        reset_if_changed: 若产物变化是否把 stable_rounds 置 0

    Returns:
        stable_rounds: 当前连续稳定轮数
    """
    raise NotImplementedError

```
</details>

### ▶️ 执行命令 Run

本关用 **pytest** 做小验收。请先在项目根目录新建测试文件，然后安装依赖并运行测试。

1) 在根目录创建文件：`tests/checkpoints/test_state_min.py` 
 
   把下面代码完整复制进去：

    <details>
    <summary><b>tests/checkpoints/test_state_min.py</b></summary>

    ```py
    from __future__ import annotations

    from typing import Any

    from debate_mas.core.state import (
        init_state,
        push_candidates,
        push_objections,
        push_diff,
        push_decisions,
        bump_round,
    )

    class FakeDossier:
        """只提供 frozen_view()，避免依赖真实 Dossier 初始化参数。"""
        def frozen_view(self) -> dict[str, Any]:
            return {"ok": True}


    def _assert_has_keys(st: dict, keys: list[str]) -> None:
        missing = [k for k in keys if k not in st]
        assert not missing, f"missing keys: {missing}"


    def test_init_state_has_core_fields() -> None:
        st = init_state("x", FakeDossier(), ref_date="2025-10-26")

        core_keys = [
            "mission",
            "ref_date",
            "dossier",
            "dossier_view",
            "messages",
            "round_idx",
            "candidates_cur",
            "objections_cur",
            "diff_cur",
            "decisions_cur",
            "history",
            "stop_reason",
            "tool_trace",
        ]
        _assert_has_keys(st, core_keys)

        assert st["round_idx"] == 0
        assert isinstance(st["messages"], list)
        assert isinstance(st["dossier_view"], dict)

        for k in ["candidates", "objections", "diffs", "decisions"]:
            assert k in st["history"], f"history missing {k}"
            assert isinstance(st["history"][k], list), f"history[{k}] must be list"


    def test_push_writes_cur_and_history() -> None:
        st = init_state("x", FakeDossier())

        push_candidates(st, [{"id": "A", "score": 1}])
        push_objections(st, [{"id": "A", "risk": "x"}])
        push_diff(st, {"changed": True})
        push_decisions(st, [{"action": "WATCH"}])

        assert len(st["candidates_cur"]) == 1
        assert len(st["objections_cur"]) == 1
        assert isinstance(st["diff_cur"], dict)
        assert len(st["decisions_cur"]) == 1

        assert len(st["history"]["candidates"]) == 1
        assert len(st["history"]["objections"]) == 1
        assert len(st["history"]["diffs"]) == 1
        assert len(st["history"]["decisions"]) == 1

        assert st["history"]["candidates"][0]["round"] == 0
        assert st["history"]["objections"][0]["round"] == 0
        assert st["history"]["diffs"][0]["round"] == 0
        assert st["history"]["decisions"][0]["round"] == 0


    def test_bump_round_resets_runtime() -> None:
        st = init_state("x", FakeDossier())

        st["_round_tool_calls"]["hunter"] = 7
        bump_round(st)

        assert st["round_idx"] == 1
        assert st["_round_tool_calls"]["hunter"] == 0
        assert st["_round_guard_denied"] is False
        assert st["_round_missing_evidence"] is False

    ```
    </details>

2) 用 uv 安装 pytest（两种方式二选一）

```bash
# 方式 A：推荐（写入项目依赖，适合有 pyproject.toml 的仓库）
uv add --dev pytest
```

```bash
# 方式 B：只在当前环境安装（不改项目依赖声明）
uv pip install pytest
```

3) 运行测试

```bash
uv run pytest -q tests/checkpoints/test_state_min.py
```

### ✅ 验收标准 Pass

- 终端输出类似下面信息（数字可能不同，但核心是 **passed**）  
  - `3 passed in ...s`  
- 过程中没有出现 `KeyError`、`AssertionError`、`ImportError`
- 如果失败，你能从报错信息定位到：
  - 缺字段：`missing keys: [...] `
  - history 结构不对：`history missing ...` 或 `history[...] must be list`
  - runtime 没重置：`_round_tool_calls` 或 `_round_guard_denied` 断言失败

### 🔁 可迁移点 Transfer

> 本关的 `state.py` 设计目标是：**框架字段稳定、业务字段可替换**。迁移到别的任务时，你不需要重写 Debate MAS，只要把“业务产物的形状”和“读取材料的方式”换掉。

**1. 框架通用 不要动**

这些是任何“多角色辩论 + skills 工具 + 可审计输出”的 Debate MAS 都离不开的骨架。迁移到别的业务时，**建议不改字段名、不改语义**。

<details>
<summary><b>state.py不需要动具体内容的地方</b></summary>

- **输入与案卷入口**
  - `mission / ref_date / dossier / dossier_view`
  - 说明：不管你做 ETF、合规、评审，都需要任务文本与材料入口。`dossier_view` 是可冻结的“摘要视图”，用于日志、提示词与报告引用。

- **对话与轮次推进**
  - `messages / round_idx`
  - `init_state() / bump_round()`
  - 说明：所有角色都要在同一条对话链和轮次上协作，轮次推进必须稳定可测。

- **结构化产物的“当前值 + 历史”**
  - `candidates_cur / objections_cur / diff_cur / decisions_cur`
  - `history`
  - `push_candidates / push_objections / push_diff / push_decisions`
  - 说明：多角色不直接通信，全靠这些字段“共享事实”。`history` 是复盘与报告的基础。

- **停机与审计**
  - `stop_reason / tool_trace`
  - 说明：停机原因用于 graph/报告解释；`tool_trace` 是“证据链”的总入口，后续关卡会把工具调用写进去并在报告引用。

- **每轮 runtime 重置**
  - `reset_round_runtime()`
  - `_round_tool_calls / _round_tool_calls_ok / _round_fingerprints / _round_guard_denied / _round_missing_evidence`
  - 说明：只要你有工具治理（次数限制、同参去重、guard 拦截），这些计数器与状态位就必须每轮可用。

</details>

**2. 业务相关 可替换或重写**

下面这些内容的“思想是通用的”，但字段名、合并 key、排序规则、pipeline 状态位，往往跟业务强绑定。迁移到别的任务时，**允许你改它们**，但建议保持“写 cur + 写 history”的模式不变。

- **候选合并策略**
  - `push_candidates_merge()` 里默认按 `symbol` 合并、按 `score` 排序，更贴近 ETF。
  - 换业务时，你可以把 “symbol/score” 换成你的主键与优先级字段。


  <details>
  <summary><b>示例 TODO：把 ETF 候选合并改成“方案评审”的提案合并</b></summary>

  ```py
  # TODO：方案评审场景（proposal_id + priority）
  def push_candidates_merge(st, incoming):
      # 1) 以 proposal_id 为 key 合并（incoming 覆盖同 id）
      # 2) 以 priority 由高到低排序
      # 3) 写回 candidates_cur，并 append history["candidates"]
      pass
  ```  
  </details>

- **收敛判断的指纹内容**
  - `bump_stable_rounds()` 默认用 `candidates + objections + diff` 生成指纹。
  - 换业务时，你可以调整指纹包含的字段，让“稳定”更符合业务定义。

  <details> 
  <summary><b>示例 TODO：把稳定判断改成“合同审阅”的条款风险稳定</b></summary>

    ```py
      # TODO：合同审阅场景（只看 risk_flags + clause_changes）
      def bump_stable_rounds(st, reset_if_changed=True):
          # cur = {"risk_flags": st["objections_cur"], "clause_changes": st["diff_cur"]}
          # 用 cur 生成指纹，若相同则 stable_rounds += 1
          pass
    ``` 
  </details> 

- **ETF pipeline 与跨轮控制位**
  - `_hunter_pipeline_stage / survivor_universe / _need_recall_diversity* / _need_rerank_composite*`
  - 说明：这些是本项目为了 ETF 的两阶段流程准备的状态位。迁移到别的业务：
    - 你可以删掉它们
    - 或替换成自己的流程状态位
    - 但要保持“默认初始化合理，不让 engine/graph KeyError”

  <details> 
  <summary><b>示例 TODO：把两阶段 pipeline 改成“医疗会诊”的流程阶段</b></summary>

    ```py
      # TODO：医疗会诊场景（triage -> consult -> final）
      # st["_pipeline_stage"] = "triage"
      # st["_need_more_tests"] = False
      # st["_need_more_tests_reason"] = ""
    ``` 
  </details> 

**‼️迁移时的“只改哪里”口诀**
  - **不动**：`init_state / bump_round / reset_round_runtime / push_* 写 cur + history / stop_reason / tool_trace`
  - **可换**：候选的主键与排序字段、稳定指纹的业务定义、业务流程阶段状态位（ETF pipeline 部分）

> 只要你遵守这条原则，Debate MAS 的核心闭环就能在不同业务里复用，而不是推倒重来。

</details>

---

## 关卡-03｜输出结构化协议 Protocol：Candidate / Objection / Decision

<details>
<summary><b>Checkpoint 03 — 结构化输出协议 Protocol 【详情】</b></summary>

> **Candidate / Objection / Decision** 分别对应三类“结构化产物”：  
> - Candidate：Hunter 的候选（写入 `state.candidates_cur`）  
> - Objection：Auditor 的质疑/风险（写入 `state.objections_cur`）  
> - Decision：PM 的最终裁决（写入 `state.decisions_cur`，并交给 Renderer 落盘）  
>
> 本关要做的是：**把这些产物的“形状”固定成协议（Schema），并确保 Renderer 能稳定消费并输出三件套。**

### 🎯 目标收获 Outcome
- 定义并稳定三类**结构化产物**的最小协议：Candidate / Objection / Decision  
- 明确 Schema 与 Renderer 分工：Schema 负责“类型与字段”，Renderer 负责“落盘与展示”  
- 用 Pydantic 把“数据形状”前置校验：让缺字段/错类型尽早失败，输出更可测试、可复盘


### 🧱 约束契约 Contract
- 本关只改：
  - `src/debate_mas/protocol/schema.py`
  - `src/debate_mas/protocol/renderer.py`
  - `src/debate_mas/protocol/__init__.py`
- 不改 graph、engine、skills 的逻辑，不引入新依赖  
- 目标不是“更聪明”，而是**协议更稳、渲染更可控、错误更早暴露**

### 🗺️ 任务清单（TODO Map）
**必看**
- `src/debate_mas/protocol/schema.py`：协议定义（Pydantic）  
- `src/debate_mas/protocol/renderer.py`：输出渲染（log.json / memo.md / rebalance.csv）  
- `src/debate_mas/protocol/__init__.py`：import 路径稳定性
- `src/debate_mas/core/state.py`：确认 `decisions_cur / tool_trace / stop_reason` 的来源与用途  

#### A) `protocol/schema.py`

**必写（框架通用）**
- `DecisionAction`：统一枚举 `BUY / WATCH / REJECT`（跨 Persona / Skills / Core / Renderer 对齐）
- `SkillResult`：Skills 的统一返回结构
  - 字段必须稳定：`success / data / insight / visuals / error_msg`
  - 提供 `ok()` / `fail()` 两个 classmethod（让 Skills 不用手写样板）
- `ToolTraceEntry`：工具留痕的最小结构（Renderer 要能落盘）
- `DebateMeta / DebateLog`：最终交付 JSON 的最小机器可读协议（log.json 的骨架）

**选改（ETF 任务扩展，但结构化思想通用）**
- `EtfCandidate`：Candidate 的最小字段（例如 `symbol/score/reason/source_skill/extra`）
- `EtfRiskReport`：Objection 的最小字段（例如 `symbol/risk_score/notes + flags`）
- `EtfDecision`：Decision 的最小字段（例如 `symbol/action/weight/final_score/key_reasons/risk_warnings`）
> 迁移到别的任务时，上面三个对象可以整体替换成别的业务对象，但“强类型 + 可校验”的思想不变。

<details>
<summary><b> 📄 Checkpoint-03：schema.py 练习骨架</b></summary>

```py
# src/debate_mas/protocol/schema.py
"""
协议定义层 (Protocol Schema)
定义全系统通用的数据交互标准：
- Layer 3 (Skills) 生产这些数据
- Layer 2 (Core)   消费/合并这些数据
- Layer 4 (Output) 展示/落盘这些数据
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# 0) 统一枚举（跨层对齐：Persona / Core / Skills / Renderer）
# =============================================================================
class DecisionAction(str, Enum):
    # TODO【必写（框架通用）】:
    # 统一枚举值，跨层对齐（Persona/Core/Skills/Renderer）
    BUY = "BUY"
    WATCH = "WATCH"
    REJECT = "REJECT"


# =============================================================================
# 1) 通用返回：SkillResult
# =============================================================================
class SkillResult(BaseModel):
    """
    [通用协议] 技能执行结果标准

    Teaching Notes:
    - success/insight: 给人看
    - data: 给机器消费（可为 List / Dict / None）
    - visuals: 图表路径留痕（可为空）
    - error_msg: 失败原因（只在 success=False 时应出现）
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # TODO【必写（框架通用）】字段必须稳定存在（Skills -> Core）
    success: bool = Field(..., description="执行是否成功")
    data: Any = Field(default=None, description="结构化返回（对象/列表/字典均可）")
    insight: str = Field(..., description="自然语言结论")
    visuals: List[str] = Field(default_factory=list, description="生成的图表路径列表")
    error_msg: Optional[str] = Field(default=None, description="错误信息（仅失败时）")

    @classmethod
    def ok(
        cls,
        data: Any = None,
        insight: str = "",
        visuals: Optional[List[str]] = None,
    ) -> "SkillResult":
        """
        快捷构造成功结果（data 可以是 List / Dict / None）

        Args:
            TODO

        Returns:
            TODO
        """
        # TODO【必写（框架通用）】:
        # - success=True
        # - error_msg=None
        # - insight 默认为空字符串
        # - visuals 默认空 list
        raise NotImplementedError

    @classmethod
    def fail(
        cls,
        error_msg: str,
        data: Any = None,
    ) -> "SkillResult":
        """
        快捷构造失败结果（允许把 debug data 带回）

        Args:
            TODO

        Returns:
            TODO
        """
        # TODO【必写（框架通用）】:
        # - success=False
        # - insight 形如 "执行失败: {error_msg}"
        # - visuals=[]
        # - error_msg=error_msg
        raise NotImplementedError


# =============================================================================
# 2) 业务中间产物（Skills 产出 / Core 消费 / Output 展示）
# =============================================================================
# TODO【选改（业务相关）】：
# - 这三类对象是 ETF 任务扩展（Candidate / Objection / Decision）
# - 迁移到别的业务时可整体替换，但“强类型 + 可校验”思想不变

class EtfCandidate(BaseModel):
    """
    Hunter 的产物：候选对象（可追溯、可排序、可解释）。

    Args:
        TODO（写你认为这个对象必须具备的字段含义，例如 symbol/score/reason/source_skill/extra）

    Returns:
        TODO（返回的是一个 EtfCandidate 实例；会被 Core 写入 state.candidates_cur，并可被 Renderer/日志消费）
    """
    # TODO【选改（业务相关）】字段定义
    symbol: str
    score: float
    reason: str
    source_skill: str
    extra: Dict[str, Any] = Field(default_factory=dict)


class EtfRiskReport(BaseModel):
    """
    Auditor 的产物：风险/质疑对象（可解释、可审计）。

    Args:
        TODO（写每个字段代表什么风险信号/标记，例如 risk_score/notes/flags 的业务含义）

    Returns:
        TODO（返回的是一个 EtfRiskReport 实例；用于跨角色共享“风险事实”，并支持落盘复盘）
    """
    # TODO【选改（业务相关）】字段定义
    symbol: str
    liquidity_flag: Optional[str] = None
    premium_flag: Optional[str] = None
    sentiment_flag: Optional[str] = None
    risk_score: float = 0.0
    notes: List[str] = Field(default_factory=list)


class EtfDecision(BaseModel):
    """
    PM 的产物：最终决策对象（可执行、可落盘、可回放）。

    Args:
        TODO（写清字段含义：symbol/action/weight/final_score/key_reasons/risk_warnings 等）

    Returns:
        TODO（返回 EtfDecision 实例；会进入 log.json / memo.md / rebalance.csv 的生成链路）
    """
    # TODO【选改（业务相关）】字段定义
    symbol: str
    action: DecisionAction
    weight: float = Field(0.0, ge=0.0, le=1.0)
    final_score: float = 0.0
    key_reasons: List[str] = Field(default_factory=list)
    risk_warnings: List[str] = Field(default_factory=list)


# =============================================================================
# 3) Layer 4 交付协议（“决策备忘录”结构）——可选但强烈建议
# =============================================================================
class ToolTraceEntry(BaseModel):
    """
    工具调用留痕（Core 可选改入，Renderer 只负责落盘）

    Args:
        TODO

    Returns:
        TODO
    """
    # TODO【必写（框架通用）】字段必须稳定（落盘 + 审计）
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    insight: str = ""
    error_msg: Optional[str] = None
    visuals: List[str] = Field(default_factory=list)


class DebateMeta(BaseModel):
    """
    交付物元信息（强烈建议 Core 填）

    Args:
        TODO

    Returns:
        TODO
    """
    # TODO【必写（框架通用）】字段定义（最小可用交付协议）
    mission: str = ""
    ref_date: Optional[str] = None
    rounds: int = 0
    stop_reason: Optional[str] = None
    tool_trace: List[ToolTraceEntry] = Field(default_factory=list)
    dossier_meta: Dict[str, Any] = Field(default_factory=dict)
    extras: Dict[str, Any] = Field(default_factory=dict)


class DebateLog(BaseModel):
    """
    最终交付 JSON 的结构化协议（机器可读、可回放）

    Args:
        TODO

    Returns:
        TODO
    """
    # TODO【必写（框架通用）】字段定义（log.json 骨架）
    timestamp: str
    meta: DebateMeta = Field(default_factory=DebateMeta)
    decisions: List[EtfDecision] = Field(default_factory=list)
    visuals: List[str] = Field(default_factory=list)

```
</details>


#### B) `protocol/renderer.py`

**必写（框架通用：Renderer 稳定性）**
- `DebateRenderer.render()`：必须生成三件套并返回路径字典
  - `log.json`：机器可读（DebateLog）
  - `memo.md`：人可读（摘要 + 表格 + 逐标的解释）
  - `rebalance.csv`：可执行/可交接（稳定列）
- `_build_meta()`：必须兼容两种输入格式（避免 Core 传参变动导致 KeyError）
  - 扁平 dict：`{"ref_date":..., "rounds":..., ...}`
  - 包一层 meta：`{"meta": {...}}`
- `_save_json_log()`：确保 `DebateLog.model_dump()` 可 JSON 序列化并落盘
- `_collect_visuals()`：收集 `meta.extras.visuals` + `tool_trace.visuals`，并去重保序

**选改（展示层/业务交付偏好）**
- `_save_markdown_memo()`：表格列/文案可以按业务调整，但要保证读取字段与 `EtfDecision` 对齐
- `_save_rebalance_csv()`：CSV 输出列可以调整，但默认应保持稳定（方便下游）
  - 你们当前写法允许通过“注释切换 columns”来增删列：
    - `df = pd.DataFrame(rows, columns=[...])`
    - `# df = pd.DataFrame(rows, columns=[...])`
  - 若修改列数/列名：需要同步下游消费方（或在 README/交接文档声明版本）
- `_build_base_filename()`：文件命名规则可按需求微调（截断长度、字符白名单等）

<details>
<summary><b> 📄 Checkpoint-03：renderer.py 练习骨架</b></summary>

```py
# src/debate_mas/protocol/renderer.py
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
        # TODO【必写（框架通用）】:
        # - 保存 output_dir
        # - 确保目录存在（os.makedirs）
        raise NotImplementedError


    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def render(
        self,
        mission: str,
        decisions: List[EtfDecision],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        执行渲染流程，生成交付三件套。
        extra_meta: 由 Core 传入的 meta（比如 ref_date / rounds / tool_trace）

        Args:
            TODO

        Returns:
            TODO
        """
        # TODO【必写（框架通用）】:
        # 1) base_filename = self._build_base_filename(mission)
        # 2) 生成 json/md/csv 三个文件并返回路径字典
        raise NotImplementedError


    def _build_base_filename(self, mission: str) -> str:
        """
        文件名统一构造：timestamp + safe_mission

        Args:
            TODO

        Returns:
            TODO
        """
        # TODO【选改（展示相关）】:
        # - timestamp 格式（如 %Y%m%d_%H%M%S）
        # - safe_mission 白名单字符与截断长度
        raise NotImplementedError


    # ---------------------------------------------------------------------
    # JSON Log (机器可读)
    # ---------------------------------------------------------------------
    def _save_json_log(
        self,
        mission: str,
        decisions: List[EtfDecision],
        filename: str,
        meta: Optional[Dict[str, Any]],
    ) -> str:
        """
        落盘 log.json（机器可读）

        Args:
            TODO

        Returns:
            TODO
        """
        # TODO【必写（框架通用）】:
        # 1) meta_obj = self._build_meta(mission, meta)
        # 2) log = DebateLog(timestamp=..., meta=meta_obj, decisions=..., visuals=...)
        # 3) json.dump(log.model_dump(), ensure_ascii=False, indent=2)
        # 4) return path
        raise NotImplementedError


    def _build_meta(self, mission: str, meta: Optional[Dict[str, Any]]) -> DebateMeta:
        """
        将 Core 传来的 extra_meta（dict）转成强类型 DebateMeta。
        兼容两种输入：
        - 扁平 dict：{"ref_date":..., "rounds":..., ...}
        - 包一层 meta：{"meta": {...}}

        Args:
            TODO

        Returns:
            TODO
        """
        # TODO【必写（框架通用）】:
        # 1) meta = meta or {}
        # 2) 若 meta 包含 {"meta": {...}}，则取内层 dict
        # 3) 构造 DebateMeta(mission=..., ref_date=..., rounds=..., stop_reason=..., tool_trace=..., dossier_meta=..., extras=...)
        raise NotImplementedError


    def _collect_visuals(self, meta_obj: DebateMeta) -> List[str]:
        """
        收集 visuals：meta.extras + tool_trace.visuals（去重保序）

        Args:
            TODO

        Returns:
            TODO
        """
        # TODO【必写（框架通用）】:
        # - 从 meta_obj.extras.get("visuals") 取 list
        # - 遍历 meta_obj.tool_trace，收集每条的 visuals
        # - 去重保序（seen set + out list）
        raise NotImplementedError


    # ---------------------------------------------------------------------
    # Markdown Memo (人可读)
    # ---------------------------------------------------------------------
    def _save_markdown_memo(
        self,
        mission: str,
        decisions: List[EtfDecision],
        filename: str,
        meta: Optional[Dict[str, Any]],
    ) -> str:
        """
        落盘 memo.md（人类可读）

        Args:
            TODO

        Returns:
            TODO
        """
        # TODO【必写（框架通用）】:
        # 1) meta_obj = self._build_meta(mission, meta)
        # 2) 生成摘要（ref_date/rounds/stop_reason + BUY/REJECT 统计）
        # 3) 生成“核心决策表”（DataFrame.to_markdown）
        # 4) 生成“逐标的决策说明”
        # 5) 追加 visuals 留痕
        # 6) 写入文件并 return path
        #
        # TODO【选改（展示相关）】:
        # - 表头字段、文案风格、逐标的段落结构
        raise NotImplementedError


    # ---------------------------------------------------------------------
    # Rebalance CSV（调仓指令）
    # ---------------------------------------------------------------------
    def _save_rebalance_csv(
        self,
        mission: str,
        decisions: List[EtfDecision],
        filename: str,
        meta: Optional[Dict[str, Any]],
    ) -> str:
        """
        落盘 rebalance.csv（调仓指令）

        Args:
            TODO

        Returns:
            TODO
        """
        # TODO【必写（框架通用）】:
        # - 默认输出列稳定（例如 time/date/code/action/weight/reason）
        # - path 写入 output_dir，编码 utf-8-sig
        #
        # TODO【选改（展示/交付相关）】:
        # - 允许通过“注释切换 columns”来增删列（教学友好）
        #   但若修改列数/列名，需要同步下游消费方或在交接文档声明版本
        raise NotImplementedError

```
</details>

#### C) `protocol/__init__.py`

**必写（框架通用：import 体验）**
- 目标：把 `schema.py` 的核心对象与 `renderer.py` 的 `DebateRenderer` **暴露到 `debate_mas.protocol` 顶层**
- 好处：
  - 外部调用更干净：`from debate_mas.protocol import SkillResult, EtfDecision, DebateRenderer`
  - 以后重构 `schema.py` 内部结构时，外部 import 路径不需要跟着改（稳定 API）

<details>
<summary><b>📄 Checkpoint-03：protocol/__init__.py</b></summary>

```py
# src/debate_mas/protocol/__init__.py
"""
Protocol Package Initialization
将 schema.py 中的核心对象暴露到 protocol 包的顶层，
方便外部使用 'from debate_mas.protocol import EtfCandidate' 这种写法。
"""
#具体暴露的对象名称根据schema.py中定义的具体写
from .schema import (
    SkillResult,
    EtfCandidate,
    EtfRiskReport,
    EtfDecision,
    DecisionAction,
)
from .renderer import DebateRenderer

__all__ = [
    "SkillResult",
    "EtfCandidate",
    "EtfRiskReport",
    "EtfDecision",
    "DecisionAction",
    "DebateRenderer",
]
```
</details>

### ▶️ 执行命令 Run

本关用 **pytest** 做小验收。

1) 在根目录创建文件：`tests/test_protocol.py`
   
   请阅读以下代码，把下面代码完整复制进去：

   <details>
   <summary><b>tests/test_protocol.py</b></summary>

   ```py
    from __future__ import annotations

    import json
    from pathlib import Path
    from typing import Any, Dict

    import pandas as pd

    from debate_mas.protocol.schema import (
        DecisionAction,
        SkillResult,
        DebateLog,
        DebateMeta,
        EtfDecision,  # 这里是“示例业务对象”；如果你换了业务对象，可在此处替换 import
    )
    from debate_mas.protocol.renderer import DebateRenderer


    def test_decision_action_enum_is_stable() -> None:
        # 若你确实要改 action 值：需要同步 Persona/Core/Renderer/下游消费方与测试断言（一般不建议）
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
        json.dumps(payload, ensure_ascii=False)  # must not raise


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

        # 注意：如果你的 renderer/schema 选择不输出其中某些字段，可根据情况删改这些断言
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
        # required_cols = ["time", "date", "code", "action", "weight", "reason"]
        # for c in required_cols:
        #     assert c in df.columns, f"csv missing column: {c}"
        #
        # 示例 2：极简金融决策版（只要 date/code/weight）
        # required_cols = ["date", "code", "weight"]
        # for c in required_cols:
        #     assert c in df.columns, f"csv missing column: {c}"


   ```
   </details>

> 说明：本关测试默认**不强绑 CSV 列名**。  
> 如果你希望把 CSV 当成“交付契约”，只需要在测试文件底部 **选改区(B)** 打开并填写你的 `required_cols`。


2) 运行测试

测试前确认`test_protocol.py`是与自己写的任务背景输出对应

```bash
uv run pytest -q tests/test_protocol.py
```

### ✅ 验收标准 Pass

- 终端输出类似下面信息（数字可能不同，但核心是 **passed**）  
  - `4 passed in ...s`  
- 过程中没有出现 `ImportError`、`AssertionError`、`JSONDecodeError`
  
- 你通过的是“框架必有”的 4 个验收点：

  1) **跨层枚举稳定**
     - `DecisionAction` 至少包含 `BUY / WATCH / REJECT`
     - 若你需要自定义枚举值，请同步修改测试中的断言集合（属于业务选改）

  2) **Skills 统一返回协议可用**
     - `SkillResult.ok()` 与 `SkillResult.fail()` 可用
     - 字段稳定存在：`success / data / insight / visuals / error_msg`

  3) **交付协议可序列化**
     - `DebateLog.model_dump()` 的结果能被 `json.dumps()` 序列化
     - 说明协议结构是“机器可读 + 可落盘”的

  4) **Renderer 能落盘三件套（不强绑业务列）**
     - `DebateRenderer.render()` 返回 `{"json": ..., "md": ..., "csv": ...}`
     - 三个文件路径都存在且可读取
     - `log.json` 至少包含：`timestamp / meta / decisions(list)`
     - `memo.md` 非空即可（文案结构允许业务自定义）
     - `rebalance.csv` 能被 `pandas.read_csv` 读入，且行数与 `decisions` 对齐  

    
- 如果失败，你能从报错快速定位到是哪一类问题
  
  <details>
  <summary><b>常见失败点</b></summary>

  - **协议缺字段 / 写错字段名**  
    - 常见报错：`ValidationError` 或 `KeyError` 或断言提示缺 key  
    - 去检查：`SkillResult / DebateMeta / DebateLog` 的字段是否与 TODO Map 一致

  - **ok/fail 没按“成功/失败”语义实现**  
    - 常见报错：`assert ok.success is True`、`assert bad.success is False` 等  
    - 去检查：`ok()` 是否 `success=True, error_msg=None`；`fail()` 是否 `success=False, error_msg=...`

  - **Renderer 没生成三件套或路径不对**  
    - 常见报错：`missing artifact: json/md/csv`  
    - 去检查：`render()` 是否真的调用 `_save_json_log/_save_markdown_memo/_save_rebalance_csv` 并返回 dict

  - **log.json 不可解析或不符合最小协议**  
    - 常见报错：`json.JSONDecodeError` 或 `assert "meta" in j`  
    - 去检查：`_save_json_log()` 是否用 `DebateLog(...).model_dump()`，并正确 `json.dump`

  - **你自定义了 CSV 列导致测试失败**  
    - 默认不会发生（本关测试不强绑列名）  
    - 若你自己打开了测试里“选改区(B)”的严格列断言，请根据你的业务列集合同步修改 `required_cols`

  </details>


### 🔁 可迁移点 Transfer

> **关卡-03** 的目标是：把“多角色 + skills”的产物，统一落到 **强类型协议**（schema）与 **可交付文件**（renderer）里。迁移到别的业务时，你不需要重写 Debate MAS，只要把“业务对象”和“展示/落盘格式”换掉。

**1. 框架通用 不要动**

这些内容是“跨 Persona / Skills / Core / Renderer”的硬接口：建议字段名、语义保持稳定。

- **统一决策枚举（跨层对齐）**
  - `DecisionAction`：`BUY / WATCH / REJECT`
  - 作用：让 Persona 的话术、Skills 的产物、Core 的合并、Renderer 的展示共用同一套 action 语义

- **Skills 统一返回结构（Skills -> Core）**
  - `SkillResult(success/data/insight/visuals/error_msg)` + `ok()/fail()`
  - 作用：让 Core/Graph 不用为每个 skill 写特殊分支，失败也能被审计与落盘

- **最小可回放交付协议（Core -> Renderer -> 文件）**
  - `ToolTraceEntry`：工具留痕最小结构（支持审计/复盘）
  - `DebateMeta / DebateLog`：log.json 的“最小机器可读骨架”

- **Renderer 的“生成三件套”职责**
  - `render()` 一次性生成：`log.json / memo.md / rebalance.csv`
  - `_build_meta()` 必须兼容：扁平 dict 和 `{"meta": {...}}` 两种传参形态
  - `_save_json_log()` 必须保证可 JSON 序列化落盘

<details>
<summary><b>迁移时：哪些属于“框架通用协议”，建议不改</b></summary>

- DecisionAction（跨层动作枚举）
- SkillResult（skills 的统一返回结构 + ok/fail 工厂方法）
- ToolTraceEntry（工具留痕最小字段：tool/args/ok/insight/error_msg/visuals）
- DebateMeta / DebateLog（log.json 的最小骨架：timestamp/meta/decisions/visuals）
- DebateRenderer.render()（三件套输出：json/md/csv）
- _build_meta() 的“兼容两种输入”能力（避免 Core 传参变动导致断裂）

</details>


**2. 业务相关 可替换**

> 这部分属于“你们当前是 ETF 的示例业务”，迁移到别的任务时可以整体替换，但建议保持“强类型 + 可校验”的思路。

- 业务对象（示例：ETF）
  - `EtfCandidate / EtfRiskReport / EtfDecision`
  - 迁移方式：整体替换成你的业务对象，例如：
    - 合同审阅：ClauseIssue / ClauseChange / ContractDecision
    - 方案评审：Proposal / ReviewRisk / ReviewDecision
    - 宏观决策：MacroSignal / RiskState / AllocationDecision
  
  <details>
  <summary><b>示例：把 EtfDecision 换成“资产配置决策”对象（业务替换）</b></summary>

  ```py
  # 示例：AllocationDecision（可替换 EtfDecision）
    class AllocationDecision(BaseModel):
        asset: str
        action: DecisionAction
        target_weight: float = Field(0.0, ge=0.0, le=1.0)
        confidence: float = 0.0
        reasons: List[str] = Field(default_factory=list)
        risk_notes: List[str] = Field(default_factory=list)
  ```
  </details>

- 展示层与交付偏好（允许强自定义）
  - `memo.md` 的结构、表头、文案：完全可以自定义
  - `rebalance.csv` 的列集合：完全可以自定义
    - 你们当前允许通过“注释切换 columns”来增删列
    - 但要记住：改列名/列数 = 改交付契约，需要同步：
      - 下游消费方（执行器/回测器/可视化工具）
      - 或在 README/交接文档里声明版本

  <details>
  <summary><b>CSV 列自定义提示（把它当作“交付契约”管理）</b></summary>

  - 默认列（示例）：time/date/code/action/weight/reason
  - 允许自定义为更简列（示例）：date/code/weight
  - 一旦自定义列集合：
    1) 同步修改下游读取逻辑
    2) 或在文档中写清楚列版本（v1/v2）
    3) 如需更严格验收，在 pytest 的“选改区(B)”固定 required_cols

  </details>

**‼️迁移时的“只改哪里”口诀**

- **不动**：`DecisionAction / SkillResult / DebateLog骨架 / Renderer三件套 + _build_meta兼容`
- **可换**：业务对象（*Decision/*Candidate/*Risk）、memo 文案结构、CSV 列集合与命名规则
- **测试策略**：关卡测试默认“宽松只验框架”；业务仓库再加“严格契约测试”

</details>

---

## 关卡-04｜证据案卷 Loader：folder → dossier

<details>
<summary><b>Checkpoint 04 — 证据案卷 Loader 【详情】</b></summary>

> 本关聚焦 **“读取指定路径的案卷”**：把一个文件夹里的 CSV/XLSX/TXT/MD/PDF/DOCX 等材料，统一装进 `Dossier`。  
> 
> **不教 SQL / ClickHouse**（数据库模式你们已有实现，练习者可自行阅读源码）。  
>
> 目标是把 Loader 的“输入/输出形状”固定成 **最小契约**，让后面的 Core/Graph/Agents 都只依赖 Dossier，而不是依赖某个业务的数据源。


### 🎯 目标收获 Outcome
- 定义并跑通 **证据入口** 的最小闭环：`folder_path → Dossier`  
- 建立“统一案卷”视角：让 Core/Agents **只读 Dossier**，不直接依赖文件格式或数据源  
- 把案卷做成可审计、可复盘的结构：
  - 表格证据：`structured_data[name] = DataFrame`  
  - 文本证据：`unstructured_text[]`  
  - 元信息留痕：`meta / tables_meta / texts_meta`  
- 支持“别名系统”（为迁移与多版本数据做准备）：
  - 文件名/表名可映射到 canonical 名称，避免数据版本命名不一致导致 KeyError


### 🧱 约束契约 Contract
- 本关只改：
  - `src/debate_mas/loader/dossier.py`
  - `src/debate_mas/loader/dual_mode_loader.py`
- 不改 graph、engine、skills 的逻辑，不引入新依赖    
- 目标不是“支持更多数据源”，而是让 **案卷入口稳定、缺依赖不崩溃、证据可冻结可审计**
  - 对可选依赖（`pypdf/docx`）必须做到：**没装也能运行，只跳过对应格式**

### 🗺️ 任务清单（TODO Map）

**必看**
- `src/debate_mas/dossier/dossier.py`：统一案卷结构（Dossier）
- `src/debate_mas/dossier/dual_mode_loader.py`：本地文件夹加载（load_from_folder）
- `src/debate_mas/core/state.py`：只要能提供 `dossier.frozen_view()`，Core/Agent 就能透视证据

#### A) `dossier.py`（案卷对象）

**必写（框架通用）**
- `Dossier`：最小字段必须稳定存在
  - `mission / structured_data / unstructured_text / meta`
  - `tables_meta / texts_meta`（用于可审计与可复盘）
- `add_table()`：写入结构化表 + 记录 tables_meta
- `add_text()`：写入文本证据 + 记录 texts_meta
- `frozen_view()`：输出只读摘要（不给 Agent 直接 DataFrame）
- `get_table()`：支持 alias/canonical 查表
- `list_tables()`：方便调试

**选改（业务偏好）**
- `table_aliases` / `_alias_to_canonical`：别名系统可扩展
- `summary()`：打印案卷概览（便于教学演示）

<details>
<summary><b> 📄 Checkpoint-04：dossier.py 练习骨架</b></summary>

```py
# src/debate_mas/dossier/dossier.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class Dossier:
    """
    【第一层：统一案卷】(The Unified Dossier)

    设计目标：
    - Core/Agents 不直接读文件、不直接连数据库
    - 只“透视” Dossier：结构化表 + 非结构化文本 + 元信息留痕
    """

    # ============================================================
    # REQ【必写-框架通用】最小字段必须稳定存在
    # ============================================================
    mission: str
    structured_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    unstructured_text: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # REQ【必写-框架通用】可审计/可复盘：记录每张表/每段文本的元信息
    tables_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    texts_meta: List[Dict[str, Any]] = field(default_factory=list)

    # OPT【选改-可迁移增强】别名系统（用于多版本数据命名不一致）
    table_aliases: Dict[str, List[str]] = field(default_factory=dict)
    _alias_to_canonical: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    # ---------------------------
    # Alias system（OPT）
    # ---------------------------
    def register_table_aliases(self, mapping: Dict[str, Any]) -> None:
        """
        注册别名映射（两种格式都支持）：
        1) {"etf_basic": ["sampled_etf_basic", "basic"], ...}
        2) {"sampled_etf_basic": "etf_basic", ...}

        Args:
            mapping: 别名映射字典

        Returns:
            None
        """
        # TODO【选改-可迁移增强】
        # - 支持两种 mapping 形态
        # - 写入 table_aliases（canonical -> aliases）
        # - 写入 _alias_to_canonical（alias -> canonical）
        raise NotImplementedError

    def resolve_table_name(self, name: str) -> Optional[str]:
        """
        把任意名字解析成真实存在的表名：
        - 先查 structured_data 是否直接命中
        - 再查 alias -> canonical
        - 再兜底处理 xxx.csv / xxx.xlsx / xxx.xls 等后缀

        Args:
            name: 用户输入的表名/别名/文件名

        Returns:
            canonical_name or None
        """
        # TODO【选改-可迁移增强】
        raise NotImplementedError

    # ---------------------------
    # Core methods（REQ）
    # ---------------------------
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
        添加表格证据（结构化）

        Args:
            name: 写入 structured_data 的表名（建议 canonical）
            df: DataFrame
            description: 表描述（可空）
            source: 来源（文件路径/数据库/接口等）
            extra: 额外元信息（可空）
            aliases: 别名列表（可空）

        Returns:
            None
        """
        # TODO【必写-框架通用】
        # 1) structured_data[name] = df
        # 2) tables_meta[name] 至少包含：
        #    - name/source/description
        #    - rows/cols/columns（列名建议 strip）
        #    - added_at（时间戳，isoformat）
        # 3) extra 合并到 tables_meta[name]
        # 4) 若 aliases 存在：register_table_aliases({name: aliases})
        raise NotImplementedError

    def add_text(
        self,
        content: str,
        source: str = "Unknown",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        添加文本证据（非结构化）

        Args:
            content: 文本内容
            source: 来源标识（文件名/URL/数据库等）
            extra: 额外元信息（可空）

        Returns:
            None
        """
        # TODO【必写-框架通用】
        # 1) unstructured_text.append(带来源的格式化文本，或原文也可)
        # 2) texts_meta.append 至少包含：
        #    - source/content_length/added_at
        # 3) extra 合并到 texts_meta[i]
        raise NotImplementedError

    def frozen_view(self) -> Dict[str, Any]:
        """
        输出只读摘要（给 Core/Agent 透视案卷，不暴露原始 DataFrame）

        Returns:
            dict with:
              - mission
              - meta（浅拷贝）
              - tables: [{name, source, rows, cols, columns[:20], description}, ...]
              - texts:  [{idx, source, length, added_at}, ...]
        """
        # TODO【必写-框架通用】
        # - columns 建议只取前 20（防爆）
        # - 不返回 DataFrame 本体
        raise NotImplementedError

    def get_table(self, name: str) -> Optional[pd.DataFrame]:
        """
        按名字或别名取结构化表格；不存在返回 None

        Args:
            name: 表名或别名

        Returns:
            DataFrame or None
        """
        # TODO【必写-框架通用】
        # - 若实现了 alias：先 resolve_table_name 再取
        # - 否则直接 structured_data.get(name)
        raise NotImplementedError

    def list_tables(self) -> List[str]:
        """
        返回案卷里所有表名，方便调试

        Returns:
            list[str]
        """
        # TODO【必写-框架通用】
        raise NotImplementedError

    @classmethod
    def create_empty(cls, mission: str) -> "Dossier":
        """
        快速创建一个空案卷

        Args:
            mission: 任务文本

        Returns:
            Dossier
        """
        # TODO【必写-框架通用】
        # - 返回 Dossier(mission=mission)
        raise NotImplementedError
```

</details>

#### B) `dual_mode_loader.py`（folder → dossier）

**必写（框架通用）**
- `load_from_folder(mission, folder_path, ...) -> Dossier`
  - 路径不存在：不抛异常，返回空 dossier（meta 里保留 source_path）
  - 支持 `.csv/.xlsx/.txt/.md` 至少四类文件
  - 对 `.pdf/.docx`：依赖缺失则跳过，不崩溃
- `table_map.json`（可选）：自动读取并注册别名映射
- `file_map`（可选）：精确文件名映射表名（教学友好）

**选改（展示/业务偏好）**
- DEFAULT_TABLE_NAME_MAP：你们项目自带的默认映射可保留，但测试不强绑
- Excel sheet 的命名规则：`base` 或 `base_sheet`（你可以自定义）

<details>
<summary><b> 📄 Checkpoint-04：dual_mode_loader.py 练习骨架</b></summary>

```py
# src/debate_mas/dossier/dual_mode_loader.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, List

import pandas as pd

from .dossier import Dossier

# --- 可选依赖：没装也不崩 ---
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


class DualModeLoader:
    """
    【双模加载器】(Dual Mode Loader)

    本关只要求实现：load_from_folder（本地文件夹模式）
    ClickHouse/SQL 模式：FUTURE（可保留代码，但不在本关教学与测试范围）
    """

    # TODO【选改-业务偏好】你们项目自带默认映射可保留；测试不强绑
    DEFAULT_TABLE_NAME_MAP: Dict[str, str] = {
        # "sampled_etf_basic": "etf_basic",
        # "govcn_2025": "govcn",
        # ...
    }

    def __init__(self) -> None:
        # TODO【必写-框架通用】允许为空
        pass

    def load_from_folder(
        self,
        mission: str,
        folder_path: str,
        file_map: Optional[Dict[str, str]] = None,
        table_name_map: Optional[Dict[str, str]] = None,
        table_name_map_path: Optional[str] = None,
        auto_load_table_map_json: bool = True,
    ) -> Dossier:
        """
        扫描指定文件夹并加载材料到 Dossier。

        REQ 支持格式（至少）：
        - .csv  -> table
        - .xlsx -> multi-sheet -> tables
        - .txt/.md -> text

        OPT 支持格式：
        - .docx/.pdf -> text（可选依赖，缺失则跳过）

        Args:
            mission: 任务文本
            folder_path: 本地资料文件夹路径
            file_map: 精确文件名 -> 表名 映射（教学友好，OPT）
            table_name_map: 别名/表名映射（runtime 覆盖，OPT）
            table_name_map_path: table_map.json 指定路径（OPT）
            auto_load_table_map_json: 是否自动尝试读取 folder/table_map.json（OPT）

        Returns:
            dossier: Dossier（即使路径不存在，也返回“空 dossier”，不抛异常）
        """
        # TODO【必写-框架通用】1) 建空 dossier + 留痕 source_path
        # dossier = Dossier.create_empty(mission)
        # dossier.meta["source_path"] = folder_path

        # TODO【必写-框架通用】2) 路径不存在：不抛异常，直接返回空 dossier
        # if not os.path.exists(folder_path): return dossier

        # TODO【选改-可迁移增强】3) 合并表名映射（优先级）：
        # DEFAULT_TABLE_NAME_MAP < table_map.json < runtime table_name_map
        # 并注册到 dossier.register_table_aliases()

        # TODO【必写-框架通用】4) 遍历文件并按后缀分发：
        # - csv/xlsx/txt/md 必须支持
        # - docx/pdf：缺依赖则跳过，不崩溃
        raise NotImplementedError

    # ---------------------------
    # Private loaders（实现细节不强绑，但建议有）
    # ---------------------------
    def _load_csv(
        self,
        dossier: Dossier,
        path: str,
        table_name: str,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        CSV 读取：尝试多编码并写入 add_table()

        Args:
            dossier: Dossier
            path: 文件路径
            table_name: 写入 structured_data 的表名
            aliases: 别名列表（通常用 base_name）

        Returns:
            None
        """
        # TODO【必写-框架通用】
        # - 至少尝试 utf-8-sig / utf-8（你也可以扩展更多编码）
        # - df.columns 建议 strip
        # - dossier.add_table(name=table_name, df=df, source=path, aliases=aliases)
        raise NotImplementedError

    def _load_excel(self, dossier: Dossier, path: str, base_name: str) -> None:
        """
        Excel 读取：每个 sheet 一张表

        Args:
            dossier: Dossier
            path: 文件路径
            base_name: 文件基础名（不含后缀）

        Returns:
            None
        """
        # TODO【必写-框架通用】
        # - pd.read_excel(sheet_name=None)
        # - 单 sheet：表名=base_name
        # - 多 sheet：表名=base_name_sheet（或你自定义规则）
        raise NotImplementedError

    def _load_txt(self, dossier: Dossier, path: str, filename: str) -> None:
        """
        txt/md：读取全文并 add_text()

        Args:
            dossier: Dossier
            path: 文件路径
            filename: 文件名（作为 source）

        Returns:
            None
        """
        # TODO【必写-框架通用】
        raise NotImplementedError

    def _load_docx(self, dossier: Dossier, path: str, filename: str) -> None:
        """
        docx：缺依赖则跳过，不崩溃

        TODO【选改-可迁移增强】
        """
        # OPT：Document is None -> return
        raise NotImplementedError

    def _load_pdf(self, dossier: Dossier, path: str, filename: str) -> None:
        """
        pdf：缺依赖则跳过，不崩溃

        TODO【选改-可迁移增强】
        """
        # OPT：PdfReader is None -> return
        raise NotImplementedError
```

</details>

### ▶️ 执行命令 Run

本关用 **pytest** 做最小验收。

1) 新建测试文件：`tests/test_dossier_loader.py`
   把下面代码完整复制进去：

   <details>
   <summary><b>tests/test_dossier_loader.py</b></summary>

   ```py
    from __future__ import annotations

    from pathlib import Path
    from typing import Any, Dict

    import pandas as pd

    from debate_mas.loader.dossier import Dossier
    from debate_mas.loader.dual_mode_loader import DualModeLoader


    def test_dossier_create_empty_contract() -> None:
        d = Dossier.create_empty(mission="x")
        assert isinstance(d, Dossier)
        assert d.mission == "x"

        # 必须存在的最小字段（框架通用）
        assert isinstance(d.structured_data, dict)
        assert isinstance(d.unstructured_text, list)
        assert isinstance(d.meta, dict)
        assert isinstance(d.tables_meta, dict)
        assert isinstance(d.texts_meta, list)


    def test_add_table_and_frozen_view_contract() -> None:
        d = Dossier.create_empty(mission="x")
        df = pd.DataFrame([{"a": 1, "b": 2}])

        d.add_table(name="t1", df=df, description="demo", source="unit-test")

        assert "t1" in d.structured_data
        assert isinstance(d.tables_meta.get("t1"), dict)

        fv = d.frozen_view()
        assert isinstance(fv, dict)
        assert fv.get("mission") == "x"
        assert "tables" in fv and isinstance(fv["tables"], list)
        assert "texts" in fv and isinstance(fv["texts"], list)

        # frozen_view 不应该暴露 DataFrame 本体
        # （只要 tables 里没有直接塞 df 就行；下面断言足够宽松）
        for t in fv["tables"]:
            assert "name" in t


    def test_add_text_contract() -> None:
        d = Dossier.create_empty(mission="x")
        d.add_text("hello world", source="note.md")

        assert len(d.unstructured_text) == 1
        assert len(d.texts_meta) == 1
        assert isinstance(d.texts_meta[0], dict)
        assert d.texts_meta[0].get("source") == "note.md"


    def test_loader_path_not_exist_returns_empty_dossier(tmp_path: Path) -> None:
        loader = DualModeLoader()
        missing = tmp_path / "not_exist_folder"

        dossier = loader.load_from_folder(
            mission="m",
            folder_path=str(missing),
        )
        assert isinstance(dossier, Dossier)
        assert dossier.mission == "m"
        # 不抛异常，并且返回空案卷
        assert isinstance(dossier.structured_data, dict)
        assert len(dossier.structured_data) == 0
        assert isinstance(dossier.unstructured_text, list)

        # 建议项：meta 里保留 source_path（你可以按需调整；如果你不打算写也可删）
        assert dossier.meta.get("source_path") == str(missing)


    def test_loader_loads_minimal_supported_files(tmp_path: Path) -> None:
        """
        必测点：至少支持 csv/xlsx/txt/md 四类。
        - csv -> 表
        - xlsx -> 表（至少 1 个 sheet）
        - txt/md -> 文本
        """
        # --- 准备测试文件 ---
        # 1) csv
        csv_path = tmp_path / "a.csv"
        pd.DataFrame([{"x": 1}, {"x": 2}]).to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 2) xlsx
        xlsx_path = tmp_path / "b.xlsx"
        with pd.ExcelWriter(xlsx_path) as w:
            pd.DataFrame([{"y": 10}]).to_excel(w, index=False, sheet_name="S1")

        # 3) txt
        txt_path = tmp_path / "c.txt"
        txt_path.write_text("hello txt", encoding="utf-8")

        # 4) md
        md_path = tmp_path / "d.md"
        md_path.write_text("# hello md", encoding="utf-8")

        loader = DualModeLoader()
        dossier = loader.load_from_folder(mission="m", folder_path=str(tmp_path))

        # 至少应有 2 张表（csv + xlsx）
        assert isinstance(dossier.structured_data, dict)
        assert len(dossier.structured_data) >= 2

        # 至少应有 2 段文本（txt + md）
        assert isinstance(dossier.unstructured_text, list)
        assert len(dossier.unstructured_text) >= 2

        # 元信息留痕应同步增长
        assert isinstance(dossier.tables_meta, dict)
        assert len(dossier.tables_meta) >= 2
        assert isinstance(dossier.texts_meta, list)
        assert len(dossier.texts_meta) >= 2


    def test_get_table_and_list_tables_are_available(tmp_path: Path) -> None:
        csv_path = tmp_path / "demo.csv"
        pd.DataFrame([{"x": 1}]).to_csv(csv_path, index=False, encoding="utf-8-sig")

        loader = DualModeLoader()
        dossier = loader.load_from_folder(mission="m", folder_path=str(tmp_path))

        names = dossier.list_tables()
        assert isinstance(names, list)
        assert len(names) >= 1

        # 不强绑具体表名，只验证 get_table 能取到某张表
        t0 = names[0]
        df0 = dossier.get_table(t0)
        assert df0 is None or isinstance(df0, pd.DataFrame)
   ```

   </details>

2) 运行测试
   
```bash
uv run pytest -q tests/test_dossier_loader.py
```

> 如果你不用 uv：也可以用 `pytest -q tests/test_dossier_loader.py`

### ✅ 验收标准 Pass

- 终端输出类似下面信息（数字可能不同，但核心是 **passed**）  
  - `6 passed in ...s`  
- 过程中没有出现 `ImportError`、`FileNotFoundError`、`KeyError`、`AssertionError`
  
- 你通过的是“框架必有”的验收点：

  1) **Dossier 最小对象可创建**
     - `Dossier.create_empty(mission)` 可用
     - 必要字段稳定存在：
       `mission / structured_data / unstructured_text / meta / tables_meta / texts_meta`

  2) **证据写入可用**
     - `add_table()` 能写入 `structured_data` 且同步写 `tables_meta`
     - `add_text()` 能写入 `unstructured_text` 且同步写 `texts_meta`

  3) **只读透视可用**
     - `frozen_view()` 返回 dict
     - 至少包含：`mission / meta / tables(list) / texts(list)`
     - 不要求输出固定字段全集，但建议 tables/texts 是可读摘要

  4) **folder → dossier 最小契约成立**
     - `load_from_folder()` 对路径不存在：不崩溃、返回空 dossier（meta 保留 source_path）
     - 至少支持 `csv/xlsx/txt/md` 四类
     - 可选依赖（`pypdf/docx`）缺失时：跳过，不报错

  5) **调试入口存在**
     - `list_tables()` 可用，便于调试
     - `get_table(name)` 可用（允许你实现 alias/canonical 或直接取）


### 🔁 可迁移点 Transfer

> **关卡-04** 的目标是：把“读取材料”与“业务推理”彻底解耦。
> 
> 迁移到任何任务（合规、投研、评审、医疗会诊…）时，你只需要替换“材料从哪来、怎么命名、怎么组织”，而 Debate MAS 的 Core/Skills/Renderer 可以保持不变。

**1. 框架通用 不要动**

- **统一证据箱：Dossier**
  - 稳定字段：`mission / structured_data / unstructured_text / meta`
  - 审计留痕：`tables_meta / texts_meta`
  - 原则：Agent 不直接读文件/连库，只看 Dossier + frozen_view()

- **只读透视：frozen_view()**
  - 原则：给 Agent 的是摘要，不是 DataFrame 本体
  - 好处：提示词更稳、日志更轻、可复盘更清晰

- **Loader 的“最小入口”**
  - `load_from_folder(mission, folder_path, ...) -> Dossier`
  - 路径不存在也不崩（返回空 dossier，让上层能继续给出“缺材料”的解释）

**2. 业务相关：允许替换/扩展**

- **别名系统**
  - 你可以用 `table_map.json / table_name_map / file_map` 统一命名
  - 如果你的业务不需要多版本兼容，可以不实现 alias；测试也不强绑

- **支持更多文件类型**
  - pdf/docx：可选依赖，没装就跳过
  - 你也可以扩展：html、pptx、图片摘要、音频转写等

- **Excel sheet 命名规则**
  - 单 sheet：`base`
  - 多 sheet：`base_sheet`（或自定义：`base__sheet`、统一小写等）
  - 只要保持“名字稳定 + 元信息留痕”即可

- **更严格的 pytest**
  - 如果你把某些表当作“必需输入契约”，可以在测试里加：
    - `assert "prices" in dossier.list_tables()`
    - `assert set(df.columns) >= {"date","close"}`
  - 如果你把文本当作“必须材料”，可以加：
    - `assert any("policy" in t.lower() for t in dossier.unstructured_text)`


**‼️迁移时的“只改哪里”口诀**

- **不动**：Dossier 的最小字段 + add_table/add_text/frozen_view + load_from_folder 的“稳定入口”
- **可换**：文件命名规则、别名映射策略、支持的文件类型、业务必需表的严格断言

</details>

---


## 关卡-05｜提示词工厂 Personas：工具白名单 + 输出格式约束

<details>
<summary><b>Checkpoint 05 — 提示词工厂 Personas 【详情】</b></summary>

> 本关把“角色提示词”从零散字符串升级为 **可复用的 Prompt Factory**：  
> 你将用一套统一模板，把 **任务指令 / 案卷摘要 / 工具白名单 / 输出格式契约** 拼成稳定的 system prompt。  
>
> 这一关不追求“更聪明”，只追求 **更稳定、可测试、可迁移**：  
> - Graph/Engine 不再手搓 prompt  
> - 角色输出更容易被解析（JSON-only / 双段输出）  
> - 工具使用不越权（白名单）  
> - 证据引用可审计


### 🎯 目标收获 Outcome
- 建立“提示词工厂”最小闭环：`mission + dossier_view + allowlist + slots -> system_prompt`
- 把角色共性规则抽成常量（强制约束）：
  - **工具白名单**：只能提及/调用白名单内工具
  - **证据优先**：只能基于 dossier + 本轮工具输出
- 把角色差异收敛到 `PromptSlots`（可配置、可扩展）
- 固定两种输出模式：
  - `json_only=True`：只输出一个 JSON 严格可解析
  - `json_only=False`：先 Debate 再 Final JSON


### 🧱 约束契约 Contract
- 本关只改 `src/debate_mas/core/personas.py`
- 不改 graph、engine、skills 的逻辑，不引入新依赖
- 目标不是“写更长的 prompt”，而是：
  - **prompt 结构稳定**
  - **工具越权被 prompt 明确禁止**
  - **输出格式对齐后续解析器需求**


### 🗺️ 任务清单（TODO Map）

**必看**
- `src/debate_mas/core/personas.py`：提示词工厂（本关，唯一需要写代码的地方）
- `src/debate_mas/core/state.py`：`dossier_view` 的来源（来自 `dossier.frozen_view()`，理解 prompt 里“证据摘要”从哪来）
- `src/debate_mas/protocol/schema.py`：最终要落地的结构类型（CANDIDATES / OBJECTIONS / DECISIONS），决定 `output_type / output_schema_hint` 怎么写


####  `personas.py`（提示词工厂）

本文件拆成两层：**通用模版层（Framework Template）** 与 **任务化个性层（Task Slots）**。  
本关练习重点是：**尽量不动通用层，只改个性层**；只有当你明确要“优化框架模版”时才动通用层。

**通用模版层**
- `build_universal_system_prompt(...) -> str`
  - 作用：把 **mission + dossier_view 摘要 + 工具白名单 + 输出格式 + 风格指南** 拼成一个稳定的 system prompt
  - 兼容两种输出模式：`json_only=True/False`
  - 这一层应该保持 **“任务无关”**：不写 ETF 专属术语、不写某个角色的细节规则
- `_ENFORCED_ROLE_RULES / _ENFORCED_TOOL_POLICY`
  - 所有角色共享的硬约束（白名单、禁止编造 tool result、证据引用规则）
  - 必须被 `build_universal_system_prompt` 注入到 prompt
- `PromptSlots`
  - 角色差异配置载体
  - 这一层不做业务决策，只提供“怎么填进 prompt”的槽位

> 初步练习目标：**不改通用模版层**，只确认它“任务无关 + 结构稳定”。

**任务化个性层（每个LLM具体的设定）**
- `get_hunter_slots() / get_auditor_slots() / get_pm_slots()`
  - 作用：把“业务角色差异”塞进 `PromptSlots`
  - 必须明确三件事：
    1) 角色要做什么（role_goal）
    2) 角色必须遵守哪些规则（role_rules / tool_policy）
    3) 输出要长什么样（output_type / output_schema_hint / json_only）
- `build_role_prompts_etf(...) -> Dict[str, str]`
  - 作用：把三套 slots + 三套 allowlist 组合成三段 prompt（hunter/auditor/pm）
  - 推荐在这里统一处理“任务级开关”（例如 MIN_CANDIDATES、是否启用约束等），写进 `extra_context`

> 初步练习核心产出：**把同一个通用模版，实例化成三种角色 prompt**。

**允许做的“进阶优化”**

- **OPT【选改-框架优化】只在明确需要时再改**
  - 优化 `build_universal_system_prompt` 的结构/排版（更短、更清晰、更利于解析）
  - 增强 dossier 摘要的兼容性（tables 是 dict/list 都能优雅展示）
  - 抽 helper 函数（如 `_render_output_section(slots)`），但不改变最终契约字段与段落要点

- **OPT【选改-业务扩展】按需增加**
  - 增加新角色 slots（analyst / execution / compliance）
  - 增加新任务组合器（例如 `build_role_prompts_credit_risk`）

<details>
<summary><b>📄 Checkpoint-05：core/personas.py 练习骨架</b></summary>

```py
# src/debate_mas/core/personas.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import CONFIG


# ============================================================
# 【框架通用】强制规则/强制工具政策：所有角色共享
# ============================================================
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


# ============================================================
# 【必写-任务化个性】ETF 三角色 slots
# ============================================================
def get_hunter_slots() -> PromptSlots:
    # TODO【必写-任务化个性】填你们的 Hunter 规则/工具政策/输出 schema hint
    raise NotImplementedError


def get_auditor_slots() -> PromptSlots:
    # TODO【必写-任务化个性】填你们的 Auditor 规则/工具政策/输出 schema hint
    raise NotImplementedError


def get_pm_slots() -> PromptSlots:
    # TODO【必写-任务化个性】填你们的 PM 规则/工具政策/输出 schema hint
    raise NotImplementedError


def build_role_prompts_etf(
    *,
    mission: str,
    dossier_view: Dict[str, Any],
    allowlist_by_role: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    一次性构造三角色 prompt，便于 Graph/Engine 调用。

    Args:
        mission: 任务指令文本
        dossier_view: Dossier.frozen_view() 产物（只读摘要）
        allowlist_by_role: {"hunter":[...], "auditor":[...], "pm":[...]}

    Returns:
        prompts: {"hunter": "...", "auditor": "...", "pm": "..."}
    """
    # TODO【必写-任务化个性】
    # - 读取 CONFIG 的 MIN_CANDIDATES 等参数（可选）
    # - 分别 build_universal_system_prompt(..., slots=get_xxx_slots())
    raise NotImplementedError
```
</details>


### ▶️ 执行命令 Run

本关用 **pytest** 做最小验收。

1) 新建测试文件：`tests/test_personas.py`
   把下面代码完整复制进去：

   <details>
   <summary><b>tests/test_personas.py</b></summary>

   ```py
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
        给 personas 用的 frozen_view 假数据：
        - tables 用 list[dict]（更贴近 Dossier.frozen_view 的输出形态）
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


    def test_build_role_prompts_etf_returns_three_role_prompts() -> None:
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

   ```

   </details>

2) 运行测试
   
```bash
uv run pytest -q tests/test_personas.py
```


### ✅ 验收标准 Pass

- 终端输出类似下面信息（数字可能不同，但核心是 **passed**）  
  - `4 passed in ...s`  
- 过程中没有出现 `ImportError`、`KeyError`、`AssertionError`
- 如果失败，你应该能从报错快速定位到三类问题：
  - **通用模版未拼装成功**：缺少固定段落（例如找不到 `【输出格式】` / `【工具权限（白名单）】`）
  - **输出模式分支不对**：`json_only=True/False` 的断言不通过（例如少了 “你必须输出两段内容” 或 “你只输出一个 JSON 对象”）
  - **角色实例化不完整**：`build_role_prompts_etf` 没返回 `hunter/auditor/pm` 三段 prompt，或白名单工具没被注入


### 🔁 可迁移点 Transfer

> 本关的 `personas.py` 设计目标是：**通用模版稳定、业务槽位可替换**。迁移到别的任务时，你不需要重写 Debate MAS，只要把“角色 slots + 工具白名单 + 输出协议”换掉。

**1. 框架通用 不要动**

这些是任何“多角色协作 + 工具白名单治理 + 结构化可解析输出”的提示词工厂都离不开的骨架。迁移到别的业务时，**建议不改段落结构、不改语义**。

<details>
<summary><b>personas.py 不需要动的地方</b></summary>

- **通用模版层**
  - `build_universal_system_prompt(mission, dossier_view, allowed_tools, slots, extra_context)`
  - 说明：负责把任务指令、证据摘要、白名单、规则、输出格式、风格拼成稳定的 system prompt。
  - 原则：保持“任务无关”（不写 ETF 专属术语，不写某个角色的细节策略）。

- **硬约束注入（治理层）**
  - `_ENFORCED_ROLE_RULES / _ENFORCED_TOOL_POLICY`
  - 说明：跨任务安全护栏，用于压制：
    - 越权提及白名单外工具
    - 虚构工具调用结果
    - 不引用真实 ToolMessage 却声称“结果显示...”

- **角色差异载体**
  - `PromptSlots`
  - 说明：只是“槽位结构”，不承担业务推理；字段稳定意味着迁移成本低。

- **输出模式契约**
  - `json_only=True/False` 两分支（只 JSON vs Debate + Final JSON）
  - 说明：保证上游 parser/renderer 能稳定吃到结构化输出。

</details>

**2. 业务相关 可替换或重写**

下面这些内容“思想是通用的”，但字段、规则、白名单、输出 schema 往往强绑定业务。迁移到别的任务时，**允许你改它们**，但建议保持“同一通用模版 + 不同 slots 实例化”的模式不变。

- 替换角色 slots
  - `get_hunter_slots / get_auditor_slots / get_pm_slots`
  - 换业务时你可以改成：
    - `analyst / reviewer / approver`
    - `triage / consult / final`
  - 只要保证每个 slots 明确三件事：
    1. 角色要做什么（`role_goal`）
    2. 角色必须遵守什么（`role_rules / tool_policy`）
    3. 输出长什么样（`output_type / output_schema_hint / json_only`）

    <details>
    <summary><b>示例 TODO：把 ETF 三角色换成“方案评审”三角色</b></summary>

    ```py
    # TODO：proposal review 场景
    def get_analyst_slots() -> PromptSlots:
        return PromptSlots(
            role_name="analyst",
            role_goal="整理证据，产出候选方案与优先级。",
            role_rules=["每个方案必须给出 evidence 摘要。"],
            tool_policy=["只可调用白名单工具取证，不可臆测外部事实。"],
            output_type="PROPOSALS",
            output_schema_hint='{"proposal_id":"p1","priority":80,"reason":"...","evidence":"..."}',
            style_guide=["短句、可审计、避免空泛。"],
            json_only=False,
        )
    ```

    </details>

- 替换组合器（把 slots + allowlist 组装成 prompt）
  - build_role_prompts_etf
  - 迁移时你通常会复制为 build_role_prompts_xxx
  - 推荐在组合器里统一处理“任务级开关”（阈值、最小产出、风险偏好），写进 extra_context
  
    <details>
    <summary><b>示例 TODO：把组合器改成 build_role_prompts_review</b></summary>

    ```py
    # TODO：把三段 prompt 一次性构造出来
    def build_role_prompts_review(mission, dossier_view, allowlist_by_role):
        analyst = build_universal_system_prompt(
            mission=mission,
            dossier_view=dossier_view,
            allowed_tools=allowlist_by_role.get("analyst", []),
            slots=get_analyst_slots(),
            extra_context="本轮至少输出 5 个候选方案。",
        )
        reviewer = ...
        approver = ...
        return {"analyst": analyst, "reviewer": reviewer, "approver": approver}
    ```

    </details>

- 对齐输出协议
  - `output_type / output_schema_hint` 必须与 `protocol/schema.py` 保持一致
  - 常见问题：
    - prompt 要求字段与 parser/renderer 期待字段不一致
    - json_only 模式与下游解析策略不匹配

**‼️迁移时的“只改哪里”口诀**
  - **不动**：`build_universal_system_prompt` 的段落结构 + `_ENFORCED_*` 治理规则 + `PromptSlots` 字段形状
  - **可换**：`get_*_slots`（角色目标/规则/输出形状/风格） + `build_role_prompts_*`（组合与任务级开关） + 各角色 allowlist
  - **再加测**：如果你的业务对字段更严格，把断言加到 pytest

</details>

---

## 关卡-06｜流程编排 Graph：Graph 跳转 + 停机规则

<details>
<summary><b>Checkpoint 06 — 流程编排 Graph 【详情】</b></summary>

> 本关把“多角色对话”升级为 **可控的流程编排**：  
> 你将用 `StateGraph` 把 **Hunter ↔ Auditor 的 attack/patch 循环** 串起来，并在一个统一裁决器里决定：  
> **继续下一轮**，还是 **收敛交给 PM 出最终决策**。  
>
> 这一关不追求“更聪明”，只追求 **更可控、可停机、可测试**：  
> - 角色输出能落进 state（cur + history）  
> - stop_reason 可解释（为什么继续 / 为什么停）  
> - 规则可插拔（MAX_ROUNDS / MIN_CANDIDATES / CONSENSUS / STABLE 等）  
> - 工具分支不死循环（tool_calls 才走 tools node）


### 🎯 目标收获 Outcome
- 理解并实现 **Graph 编排最小闭环**：`hunter -> auditor -> (next_round | pm) -> END`
- 把“停机规则”收敛到一个统一裁决器：`_should_end_debate(state) -> "next_round" | "pm"`
- 能把 LLM 输出（末尾 JSON）解析为 payload，并落入 state：
  - `CANDIDATES -> candidates_cur + history + diff_cur`
  - `OBJECTIONS -> objections_cur + history + survivor_universe + risk_reports`
  - `DECISIONS -> decisions_cur + history`
- 给“为什么继续/为什么强制”留下可追溯软 trace


### 🧱 约束契约 Contract
- 本关只改：`src/debate_mas/core/graph.py`
- 不改：engine、skills、personas、protocol 的接口契约；不引入新依赖
- 目标不是写更多节点，而是：
  - **跳转逻辑稳定**
  - **停机规则清晰**


### 🗺️ 任务清单（TODO Map）

**必看**
- `src/debate_mas/core/graph.py`：本关主文件（流程编排 + 跳转 + 停机）
- `src/debate_mas/core/state.py`：`bump_round / push_* / bump_stable_rounds / set_need_more_candidates` 等写账本接口
- `src/debate_mas/core/config.py`：`MAX_ROUNDS / EXIT_ON_CONSENSUS / ENFORCE_MIN_CANDIDATES` 等开关来源
- `src/debate_mas/protocol/etf_debate.py`：payload 解析与校验（`try_parse_payload_with_span / validate_payload`）
- `src/debate_mas/core/personas.py`：role prompt 结构（graph 会把 system prompt 置顶）


#### 必写（框架通用）

- **Graph骨架**：`build_etf_attack_patch_graph()` 负责 `hunter → auditor → (next_round|pm) → END` 的主循环编排  
- **角色节点抽象**：`RoleBlock`（`system_prompt / llm_invoke / tool_node / postprocess`）作为图节点的最小契约  
- **System置顶**：`_append_system_prompt()` 保证每轮 system prompt 在 messages 顶部且不破坏历史  
- **工具分支判定**：`_last_ai_has_tool_calls()` 只在 AIMessage 真有 tool_calls 时走 tools node（防死循环）  
- **Payload抽取**：`_extract_last_payload()` 支持 “Debate + 末尾 JSON” 与 “纯 JSON” 两种形态  
- **Stop建议读取**：`_get_stop_suggest()` 统一提取 `STOP/CONTINUE`（供裁决器使用）  
- **停机裁决器**：`_should_end_debate()` 统一写入 `stop_reason` 并决定下一跳（`next_round` / `pm`）  
- **轮次推进**：`next_round` 节点里调用 `bump_round()` 并完成每轮 runtime 重置（guard/tool计数等）  
- **三段落地**：`postprocess_hunter / postprocess_auditor / postprocess_pm` 把 payload 写入 state（cur + history）  

---

#### 必写（ETF任务相关，迁移可替换）

- **Need Evidence 抽取**：`_extract_need_evidence()` 从 OBJECTIONS 中提取 `NEED_EVIDENCE` 的 symbol/actions  
- **MIN_CANDIDATES 契约**：`_unique_candidate_count / _min_candidates_required / _min_candidates_status` 做候选池达标判定  
- **候选合并入口**：`push_candidates_merge()`（state侧）+ graph 的 hunter postprocess 合并写回  
- **Survivor Universe 计算**：`_compute_survivor_universe()` 基于 objections + risk_reports 做硬剔除得到 U1  
- **DIFF 自动计算**：`_index_by_symbol / _compute_candidates_diff()` 生成 ADD/UPDATE 等 patch（写入 diff_cur）  
- **软Trace解释**：`_append_soft_trace()` 用 tool_trace 记录“为什么继续/为什么强制/为什么 diff”（非证据但可审计）  
- **候选字段补齐**：`_normalize_candidate_items()` 缺字段自动补齐（避免下游 parser/renderer 崩）  
- **风险报告合并**：`_extract_risk_items_from_cache / _merge_risk_reports()` 把 `market_sentry/forensic_detective` 合并为 risk_reports  
- **工具缓存读取**：`tool_cache` 读取约定（graph侧只读）  
- **强制工具调用标志**：`_force_hunter_tool` 与 `_round_missing_evidence` 的联动（防“该取证却没取证”）  
- **Need More Candidates 状态位**：`set_need_more_candidates / clear_need_more_candidates`（state侧）+ next_round 里打标/清标  

---

#### 选改（拓展位）

- **Two-stage Pipeline提示拼接**：`_build_hunter_pipeline_sys_prompt()`（RECALL/RERANK）作为 Hunter 每轮额外 SystemMessage  
- **Pipeline达标判定**：`_hunter_used_sniper_strategies_this_round()` + `_need_recall_diversity/_need_rerank_composite` 的 gate  
- **Rerank截断**：RERANK 阶段 `TopN` 裁剪（防 token 爆炸）与 `__rerank_cutoff__` trace  
- **硬剔除扩展**：survivor_universe 的规则扩展（更多 flags / 更多阈值）  
- **早停策略扩展**：`EXIT_ON_CONSENSUS / stable_rounds` 之外增加新的 early-stop 条件  
- **工具节点封装**：`_make_tool_wrapper()` / `ToolNode` 的进一步封装（例如统一记录 tool_trace）  
- **更严格的payload校验**：对 `items` 的 schema 字段更强断言（配合 tests 增强）  


<details>
<summary><b>📄 Checkpoint-06：core/graph.py 练习骨架</b></summary>

```py
# src/debate_mas/core/graph.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from .config import CONFIG
from .state import DebateState

from debate_mas.protocol.etf_debate import try_parse_payload_with_span, validate_payload

# ============================================================
# 0) 类型与小约定
# ============================================================

ToolRunner = Callable[[DebateState], DebateState]


@dataclass(frozen=True)
class RoleBlock:
    """
    RoleBlock：把“角色节点”抽成统一结构，便于 graph 拼装与测试。

    Args:
        role: 角色名（如 hunter / auditor / pm）
        system_prompt: 该角色的 system prompt（每轮置顶注入）
        llm_invoke: 输入 messages，返回 AIMessage
        tool_node: 可选；工具节点 runner（LangGraph ToolNode 或自定义 wrapper）
        postprocess: 本轮产物落地到 state 的函数（写 cur + history + stop_suggest 等）

    Returns:
        None
    """
    role: str
    system_prompt: str
    llm_invoke: Callable[[List[BaseMessage]], AIMessage]
    tool_node: Optional[ToolRunner]
    postprocess: Callable[[DebateState], None]


# ============================================================
# 1) 通用：system prompt 置顶 + tool_calls 路由
# ============================================================

def _append_system_prompt(messages: List[BaseMessage], system_prompt: str) -> List[BaseMessage]:
    """
    把 system prompt 置顶，不改变原 messages 的顺序。

    TODO【必写（通用框架）】:
        - 返回: [SystemMessage(system_prompt)] + (messages or [])

    Args:
        messages: 原对话历史
        system_prompt: system prompt 文本

    Returns:
        新的 messages 列表
    """
    # TODO
    raise NotImplementedError


def _last_ai_has_tool_calls(state: DebateState) -> bool:
    """
    判断最近一条 AIMessage 是否包含 tool_calls（决定走 tools node 还是直接 postprocess）。

    TODO【必写（通用框架）】:
        - 从 state["messages"] 逆序找第一条 AIMessage
        - 兼容两种存放位置：
          1) getattr(m, "tool_calls", None)
          2) getattr(m, "additional_kwargs", {}).get("tool_calls")
        - 找到 AIMessage 后立即返回 True/False
        - 若没有 AIMessage，返回 False

    Args:
        state: DebateState

    Returns:
        是否存在 tool_calls
    """
    # TODO
    raise NotImplementedError


# ============================================================
# 2) payload 抽取：支持“辩论文字 + 末尾 JSON”
# ============================================================

def _extract_last_payload(state: DebateState, *, expected_type: str) -> Optional[Dict[str, Any]]:
    """
    从最近的 AIMessage 中抽取末尾 JSON payload（通过协议层解析 + 校验）。

    TODO【必写（通用框架）】:
        - 逆序遍历 messages，只看 AIMessage
        - obj, _span = try_parse_payload_with_span(m.content)
        - validate_payload(obj) 通过后才算成功
        - obj["type"].upper() == expected_type 才返回 obj
        - 否则继续向前找
        - 全部失败返回 None

    Args:
        state: DebateState
        expected_type: 期望类型（如 "CANDIDATES" / "OBJECTIONS" / "DECISIONS"）

    Returns:
        payload dict 或 None
    """
    # TODO
    raise NotImplementedError


def _get_stop_suggest(obj: Optional[Dict[str, Any]]) -> str:
    """
    读取 stop_suggest，统一为大写。

    TODO【必写（通用框架）】:
        - (obj or {}).get("stop_suggest", "")
        - strip + upper

    Args:
        obj: payload 或 None

    Returns:
        stop_suggest（"STOP"/"CONTINUE"/""）
    """
    # TODO
    raise NotImplementedError


# ============================================================
# 3) ETF任务：Need Evidence / MIN_CANDIDATES / Pipeline 辅助
# ============================================================

def _extract_need_evidence(objections: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """
    从 objections 中抽取 NEED_EVIDENCE 的标志位 + 证据要求集合。

    TODO【必写（ETF任务相关）】:
        - 遍历 objections:
          - verdict == "NEED_EVIDENCE" -> need=True
          - 收集 symbol（去重保序）
          - 收集 required_actions（去重保序）
        - 返回 (need, syms, actions)

    Args:
        objections: OBJECTIONS.items

    Returns:
        need: 是否需要补证据
        syms: 需要补证据的 symbol 列表（去重保序）
        actions: required_actions 列表（去重保序）
    """
    # TODO
    raise NotImplementedError


def _unique_candidate_count(items: List[Dict[str, Any]]) -> int:
    """
    统计 candidates.items 的 unique symbol 数量。

    TODO【必写（ETF任务相关）】:
        - 以 symbol 去重计数（strip 后非空）

    Args:
        items: candidates list

    Returns:
        unique symbol count
    """
    # TODO
    raise NotImplementedError


def _min_candidates_required() -> int:
    """
    从 CONFIG 读取最小候选数要求。

    TODO【必写（ETF任务相关）】:
        - 若 CONFIG.ENFORCE_MIN_CANDIDATES 为 False -> 0
        - 否则返回 int(CONFIG.HUNTER_MIN_CANDIDATES or 0)

    Args:
        None

    Returns:
        MIN_CANDIDATES
    """
    # TODO
    raise NotImplementedError


def _min_candidates_status(state: DebateState) -> Tuple[int, int, int]:
    """
    汇总 MIN_CANDIDATES 状态。

    TODO【必写（ETF任务相关）】:
        - mn = _min_candidates_required()
        - have = _unique_candidate_count(state["candidates_cur"])
        - missing = max(0, mn - have)

    Args:
        state: DebateState

    Returns:
        mn: 最小要求
        have: 当前 unique 数
        missing: 还差多少
    """
    # TODO
    raise NotImplementedError


def _hunter_used_sniper_strategies_this_round(state: DebateState) -> List[str]:
    """
    读取 hunter 本轮使用过的召回策略列表（去重保序）。

    TODO【选改（拓展位）】:
        - 直接读 state["_hunter_round_sniper_strategies"]
        - strip + 去重保序

    Args:
        state: DebateState

    Returns:
        strategies: 本轮策略列表
    """
    # TODO
    raise NotImplementedError


def _compute_survivor_universe(state: DebateState) -> List[str]:
    """
    计算存活池 survivor_universe（U1）。

    TODO【必写（ETF任务相关）】:
        - 基于 candidates_cur 的 symbol
        - 剔除 objections_cur 中 verdict == "REJECT"
        - 剔除 risk_reports 中 liquidity_flag == "illiquid"
        - 剔除 risk_score >= CONFIG.RISK_SCORE_THRESHOLD
        - 保持去重保序

    Args:
        state: DebateState

    Returns:
        survivor_universe: 存活池 symbol 列表
    """
    # TODO
    raise NotImplementedError


def _index_by_symbol(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    把 items 按 symbol 索引成 dict。

    TODO【必写（通用框架）】:
        - 忽略 symbol 为空的条目
        - 后写覆盖前写（以“最新”为准）

    Args:
        items: list[dict]

    Returns:
        symbol -> item
    """
    # TODO
    raise NotImplementedError


def _compute_candidates_diff(prev_items: List[Dict[str, Any]], cur_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算 DIFF（系统自动算）。

    TODO【必写（ETF任务相关）】:
        - 新增：cur_syms - prev_syms -> {"op":"ADD","symbol":...}
        - 同标的变更（轻量即可）：
          - score 变化 -> SCORE_UPDATE
          - reason 变化 -> REASON_UPDATE
        - 返回 {"type":"DIFF","items":[...patches...]}

    Args:
        prev_items: 上一版候选
        cur_items: 当前候选

    Returns:
        diff_obj: DIFF payload
    """
    # TODO
    raise NotImplementedError


def _append_soft_trace(
    state: DebateState,
    *,
    role: str,
    tool: str,
    insight: str,
    args: Optional[Dict[str, Any]] = None,
    ok: bool = True,
) -> None:
    """
    写入软 trace（解释“为什么继续/为什么强制/为什么 diff”）。

    TODO【必写（通用框架）】:
        - state.setdefault("tool_trace", [])
        - append 一条 dict（至少包含 kind/role/tool/args/ok/round_idx/insight 等稳定字段）

    Args:
        state: DebateState
        role: "system"/"hunter"/"auditor"/"pm"
        tool: trace 名称（如 "__diff__"）
        insight: 一句话解释
        args: 可选参数快照
        ok: 是否 ok

    Returns:
        None
    """
    # TODO
    raise NotImplementedError


def _make_tool_wrapper(tool_node: ToolRunner) -> ToolRunner:
    """
    工具节点 wrapper：统一成 (state)->state 的签名。

    TODO【必写（通用框架）】:
        - 直接 return tool_node(state)

    Args:
        tool_node: 工具 runner

    Returns:
        wrapper: ToolRunner
    """
    # TODO
    raise NotImplementedError


def _build_hunter_pipeline_sys_prompt(state: DebateState) -> Optional[str]:
    """
    Two-stage pipeline 的 hunter sys prompt 拼接（RECALL / RERANK）。

    TODO【选改（拓展位）】:
        - 若 CONFIG.HUNTER_DETERMINISTIC_PIPELINE=False -> None
        - 若 CONFIG.HUNTER_PIPELINE_MODE != "two_stage" -> None
        - 读 state["_hunter_pipeline_stage"]（默认 recall）
        - recall: 提示 multi-strategy recall（min_strats/topk_each）
        - rerank: 提示 composite rerank（universe=survivor_universe）

    Args:
        state: DebateState

    Returns:
        sys_prompt 或 None
    """
    # TODO
    raise NotImplementedError


# ============================================================
# 4) postprocess：把“本轮产物”写入 state（cur + history）
# ============================================================

def _normalize_candidate_items(
    items: List[Dict[str, Any]],
    prev_by_sym: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    轻量规范化：缺字段补齐（避免下游崩溃）。

    TODO【必写（ETF任务相关）】:
        - symbol 为空 -> 丢弃
        - 缺 score/reason/source_skill/extra 时，用 prev_by_sym[symbol] 的旧值补齐
        - extra 不是 dict 时转成 dict 包起来
        - 返回 (norm_items, autofill_count)

    Args:
        items: 原始候选 items
        prev_by_sym: 上一轮候选索引

    Returns:
        norm_items: 规范化候选
        autofill_count: 自动补齐字段次数
    """
    # TODO
    raise NotImplementedError


def postprocess_hunter(state: DebateState) -> None:
    """
    Hunter 的 postprocess：落地 CANDIDATES + 计算 DIFF + 写 stop_suggest + pipeline gate。

    TODO【必写（通用框架）】:
        - obj = _extract_last_payload(expected_type="CANDIDATES")
        - items 必须是 list，否则 return
        - push_candidates_merge(state, norm_items)
        - push_diff(state, diff_obj)
        - state["hunter_stop_suggest"] = _get_stop_suggest(obj)
        - _append_soft_trace(...) 记录关键解释

    TODO【必写（ETF任务相关）】:
        - MIN_CANDIDATES 达标时清理 need_more_candidates
        - 如 state["_force_hunter_tool"] 为 True 且本轮无有效工具调用 -> 标记 _round_missing_evidence + 强制 CONTINUE

    TODO【选改（拓展位）】:
        - two-stage pipeline：
          - recall：检查策略多样性，不达标则 _need_recall_diversity=True
          - rerank：检查 composite，不达标则 _need_rerank_composite=True
        - rerank 阶段 TopN 截断 + trace

    Args:
        state: DebateState

    Returns:
        None
    """
    # TODO
    raise NotImplementedError


def _extract_risk_items_from_cache(state: DebateState, tool_name: str) -> List[Dict[str, Any]]:
    """
    从 tool_cache 中抽取某工具的 items 列表。

    TODO【必写（ETF任务相关）】:
        - cache = state["tool_cache"].get(tool_name)
        - cache["data"]["items"] 若存在且是 list 则返回，否则 []

    Args:
        state: DebateState
        tool_name: 工具名（如 "market_sentry"）

    Returns:
        items: list[dict]
    """
    # TODO
    raise NotImplementedError


def _merge_risk_reports(*lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    合并多个风险报告列表（按 symbol 合并、风险累加、flag 取更严重、notes 去重）。

    TODO【必写（ETF任务相关）】:
        - symbol 作为主键
        - risk_score 累加并 cap 到 100
        - liquidity_flag: ok < illiquid（取更严重）
        - sentiment_flag: normal < negative（取更严重）
        - notes 去重保序
        - 最终按 risk_score desc 排序

    Args:
        *lists: 多个风险 items 列表

    Returns:
        merged: 合并后的风险报告
    """
    # TODO
    raise NotImplementedError


def postprocess_auditor(state: DebateState) -> None:
    """
    Auditor 的 postprocess：落地 OBJECTIONS + 合并风险报告 + 计算 survivor_universe + bump_stable_rounds。

    TODO【必写（通用框架）】:
        - obj = _extract_last_payload(expected_type="OBJECTIONS")
        - push_objections(state, items)
        - state["auditor_stop_suggest"] = _get_stop_suggest(obj)
        - _append_soft_trace(...) 记录输出

    TODO【必写（ETF任务相关）】:
        - 从 tool_cache 抽 market_sentry / forensic_detective 风险 items
        - state["risk_reports"] = _merge_risk_reports(...)
        - prev_u1 = survivor_universe（用于 diff REMOVE patch）
        - state["survivor_universe"] = _compute_survivor_universe(state)
        - removed -> 追加 hard REMOVE patches 到 diff_cur 并 push_diff
        - bump_stable_rounds(state)
        - need_evidence: state["_need_evidence"], symbols, actions

    Args:
        state: DebateState

    Returns:
        None
    """
    # TODO
    raise NotImplementedError


def postprocess_pm(state: DebateState) -> None:
    """
    PM 的 postprocess：落地 DECISIONS + 写 stop_suggest。

    TODO【必写（通用框架）】:
        - obj = _extract_last_payload(expected_type="DECISIONS")
        - push_decisions(state, items)
        - state["pm_stop_suggest"] = _get_stop_suggest(obj)
        - _append_soft_trace(...) 记录输出

    Args:
        state: DebateState

    Returns:
        None
    """
    # TODO
    raise NotImplementedError


# ============================================================
# 5) judge：决定下一步走向（attack/patch 的收敛规则）
# ============================================================

def _should_end_debate(state: DebateState) -> str:
    """
    收敛裁决器：决定走 next_round 还是 pm，并写 stop_reason。

    TODO【必写（通用框架）】:
        - MAX_ROUNDS：到上限 -> stop_reason="MAX_ROUNDS_DEBATE" -> "pm"
        - guard denied：_round_guard_denied -> stop_reason="GUARD_DENIED" -> "next_round"
        - 共识停机（可控开关）：EXIT_ON_CONSENSUS + hunter/auditor 都 STOP -> "pm"
        - 稳定停机（可控阈值）：stable_rounds 达标 + auditor STOP -> "pm"
        - 默认：stop_reason="CONTINUE_DEBATE" -> "next_round"

    TODO【必写（ETF任务相关）】:
        - MIN_CANDIDATES gate：不达标 -> stop_reason="MIN_CANDIDATES_NOT_MET" -> "next_round"
        - pipeline gate：
          - _need_recall_diversity -> stop_reason="PIPELINE_RECALL_DIVERSITY_NOT_MET" -> "next_round"
          - _need_rerank_composite -> stop_reason="PIPELINE_RERANK_NOT_MET" -> "next_round"

    Args:
        state: DebateState

    Returns:
        next: "next_round" or "pm"
    """
    # TODO
    raise NotImplementedError


# ============================================================
# 6) Graph：Hunter ↔ Auditor（attack/patch）→ PM
# ============================================================

def build_etf_attack_patch_graph(*, hunter: RoleBlock, auditor: RoleBlock, pm: RoleBlock) -> Any:
    """
    构建 LangGraph 主图：hunter -> auditor -> (next_round | pm) -> END。

    TODO【必写（通用框架）】:
        - g = StateGraph(DebateState)
        - add_role(rb):
          - 创建 {role}_agent / {role}_tools / {role}_postprocess 三节点
          - agent 节点：注入 system prompt、调用 llm、append AIMessage、写 phase/_last_speaker_role
          - tools 分支：_last_ai_has_tool_calls 决定 tools 或 post
          - tools -> agent 回边（允许多次工具调用）
          - agent -> postprocess 边（无工具时直达）
        - hunter_post -> auditor_agent
        - auditor_post -> conditional(_should_end_debate) -> {next_round, pm}
        - pm_post -> END
        - entry_point = hunter_agent
        - return g.compile()

    TODO【必写（ETF任务相关）】:
        - next_round 节点：
          - bump_round(state)
          - MIN_CANDIDATES 不足时 set_need_more_candidates + trace
          - 否则 clear_need_more_candidates
          - need_evidence / guard_denied / pipeline_fix -> 设置 state["_force_hunter_tool"]
          - two-stage pipeline：按 need_more / diversity 等推进 stage（recall / rerank）
          - 追加解释性 trace

    TODO【选改（拓展位）】:
        - auditor 强制工具调用提示（额外 SystemMessage）
        - hunter Two-stage sys_prompt 注入（_build_hunter_pipeline_sys_prompt）
        - 更严格的图节点命名与可视化 debug 标记

    Args:
        hunter: hunter RoleBlock
        auditor: auditor RoleBlock
        pm: pm RoleBlock

    Returns:
        compiled_graph: 可 invoke/stream 的 LangGraph 图对象
    """
    # TODO
    raise NotImplementedError
```

</details>


### ▶️ 执行命令 Run

本关用 **pytest** 做最小验收。

1) 新建测试文件：`tests/test_graph.py`
   把下面代码完整复制进去：

   <details>
   <summary><b>tests/test_graph.py</b></summary>

   ```py
    import json
    import re
    from types import SimpleNamespace

    import pytest
    from langchain_core.messages import AIMessage, HumanMessage

    from debate_mas.core import graph as g


    def _patch_config(monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
        """直接 monkeypatch 模块变量 g.CONFIG 为一个可变对象（stub）。"""
        cfg = SimpleNamespace(
            MAX_ROUNDS=3,
            EXIT_ON_CONSENSUS=True,

            ENFORCE_MIN_CANDIDATES=False,
            HUNTER_MIN_CANDIDATES=0,

            HUNTER_DETERMINISTIC_PIPELINE=True,
            HUNTER_PIPELINE_MODE="two_stage",
            HUNTER_RECALL_STRATEGIES=["momentum", "liquidity", "composite"],
            HUNTER_RECALL_MIN_STRATEGIES=2,
            HUNTER_RECALL_TOPK_PER_STRATEGY=10,
            HUNTER_RERANK_OUTPUT_TOPN=20,

            RISK_SCORE_THRESHOLD=50.0,

            ENFORCE_TOOL_ON_NEED_EVIDENCE=True,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)

        monkeypatch.setattr(g, "CONFIG", cfg, raising=True)

    def _patch_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
        """
        把协议层解析/校验 monkeypatch 成“可控且稳定”的版本，
        让本关测试聚焦在 graph 编排与状态机逻辑，而不是 JSON schema 细节。
        """
        def fake_parse(text: str):
            m = re.search(r"(\{.*\})\s*$", text, re.S)
            if not m:
                return None, -1
            try:
                return json.loads(m.group(1)), m.start(1)
            except Exception:
                return None, -1

        def fake_validate(_obj):
            return None

        monkeypatch.setattr(g, "try_parse_payload_with_span", fake_parse, raising=True)
        monkeypatch.setattr(g, "validate_payload", fake_validate, raising=True)


    def _mk_ai(payload: dict) -> AIMessage:
        return AIMessage(content=json.dumps(payload, ensure_ascii=False))

    def test_append_system_prompt_prepends_and_keeps_history_order():
        msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
        out = g._append_system_prompt(msgs, system_prompt="SYS")
        assert out[0].content == "SYS"
        assert [m.content for m in out[1:]] == ["hi", "hello"]


    def test_last_ai_has_tool_calls_detects_both_storage_styles():
        state = {"messages": [AIMessage(content="x")]}
        assert g._last_ai_has_tool_calls(state) is False

        state = {
            "messages": [
                AIMessage(
                    content="x",
                    additional_kwargs={
                        "tool_calls": [
                            {"id": "1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}
                        ]
                    },
                )
            ]
        }
        assert g._last_ai_has_tool_calls(state) is True

        m = AIMessage(content="x")
        setattr(m, "tool_calls", [{"name": "bar"}])
        state = {"messages": [m]}
        assert g._last_ai_has_tool_calls(state) is True


    def test_get_stop_suggest_uppercase_and_strip():
        assert g._get_stop_suggest({"stop_suggest": " stop "}) == "STOP"
        assert g._get_stop_suggest({"stop_suggest": "continue"}) == "CONTINUE"
        assert g._get_stop_suggest(None) == ""


    def test_extract_last_payload_supports_debate_plus_tail_json(monkeypatch: pytest.MonkeyPatch):
        _patch_protocol(monkeypatch)

        payload = {"type": "CANDIDATES", "stop_suggest": "STOP", "items": []}
        msg = AIMessage(content="some debate...\n" + json.dumps(payload, ensure_ascii=False))
        state = {"messages": [msg]}

        out = g._extract_last_payload(state, expected_type="CANDIDATES")
        assert out is not None
        assert out["type"] == "CANDIDATES"


    def test_should_end_debate_max_rounds_go_pm(monkeypatch: pytest.MonkeyPatch):
        _patch_config(monkeypatch, MAX_ROUNDS=2, ENFORCE_MIN_CANDIDATES=False)

        state = {"round_idx": 1, "stable_rounds": 0, "messages": []}
        nxt = g._should_end_debate(state)
        assert nxt == "pm"
        assert state.get("stop_reason") == "MAX_ROUNDS_DEBATE"


    def test_should_end_debate_guard_denied_forces_next_round(monkeypatch: pytest.MonkeyPatch):
        _patch_config(monkeypatch, MAX_ROUNDS=99, ENFORCE_MIN_CANDIDATES=False)

        state = {"round_idx": 0, "_round_guard_denied": True, "messages": []}
        nxt = g._should_end_debate(state)
        assert nxt == "next_round"
        assert state.get("stop_reason") == "GUARD_DENIED"


    def test_should_end_debate_min_candidates_gate(monkeypatch: pytest.MonkeyPatch):
        _patch_config(monkeypatch, MAX_ROUNDS=99, ENFORCE_MIN_CANDIDATES=True, HUNTER_MIN_CANDIDATES=3)

        state = {
            "round_idx": 0,
            "messages": [],
            "candidates_cur": [{"symbol": "510300"}, {"symbol": "510500"}],  # unique=2 < 3
            "hunter_stop_suggest": "STOP",
            "auditor_stop_suggest": "STOP",
        }
        nxt = g._should_end_debate(state)
        assert nxt == "next_round"
        assert state.get("stop_reason") == "MIN_CANDIDATES_NOT_MET"


    def test_should_end_debate_pipeline_gates(monkeypatch: pytest.MonkeyPatch):
        _patch_config(monkeypatch, MAX_ROUNDS=99, ENFORCE_MIN_CANDIDATES=False)

        s1 = {"round_idx": 0, "_need_recall_diversity": True, "messages": []}
        assert g._should_end_debate(s1) == "next_round"
        assert s1["stop_reason"] == "PIPELINE_RECALL_DIVERSITY_NOT_MET"

        s2 = {"round_idx": 0, "_need_rerank_composite": True, "messages": []}
        assert g._should_end_debate(s2) == "next_round"
        assert s2["stop_reason"] == "PIPELINE_RERANK_NOT_MET"


    def test_should_end_debate_consensus_stop(monkeypatch: pytest.MonkeyPatch):
        _patch_config(monkeypatch, MAX_ROUNDS=99, EXIT_ON_CONSENSUS=True, ENFORCE_MIN_CANDIDATES=False)

        state = {
            "round_idx": 0,
            "messages": [],
            "hunter_stop_suggest": "STOP",
            "auditor_stop_suggest": "STOP",
            "stable_rounds": 0,
        }
        nxt = g._should_end_debate(state)
        assert nxt == "pm"
        assert state.get("stop_reason") == "CONSENSUS_STOP"


    def test_should_end_debate_stable_and_auditor_stop(monkeypatch: pytest.MonkeyPatch):
        _patch_config(monkeypatch, MAX_ROUNDS=99, EXIT_ON_CONSENSUS=False, ENFORCE_MIN_CANDIDATES=False)

        state = {
            "round_idx": 0,
            "messages": [],
            "hunter_stop_suggest": "CONTINUE",
            "auditor_stop_suggest": "STOP",
            "stable_rounds": 1,
        }
        nxt = g._should_end_debate(state)
        assert nxt == "pm"
        assert state.get("stop_reason") == "STABLE_AND_AUDITOR_STOP"


    def test_should_end_debate_default_continue(monkeypatch: pytest.MonkeyPatch):
        _patch_config(monkeypatch, MAX_ROUNDS=99, EXIT_ON_CONSENSUS=True, ENFORCE_MIN_CANDIDATES=False)

        state = {
            "round_idx": 0,
            "messages": [],
            "hunter_stop_suggest": "CONTINUE",
            "auditor_stop_suggest": "CONTINUE",
            "stable_rounds": 0,
        }
        nxt = g._should_end_debate(state)
        assert nxt == "next_round"
        assert state.get("stop_reason") == "CONTINUE_DEBATE"


    def test_graph_compiles_and_runs_one_cycle(monkeypatch: pytest.MonkeyPatch):
        """
        最小端到端：能 compile + invoke，且走完 hunter->auditor->pm。
        不校验业务字段，只校验“主循环能跑通 + stop_reason 写入”。
        """
        _patch_protocol(monkeypatch)
        _patch_config(monkeypatch, MAX_ROUNDS=1, ENFORCE_MIN_CANDIDATES=False)

        hunter_rb = g.RoleBlock(
            role="hunter",
            system_prompt="HUNTER_SYS",
            llm_invoke=lambda _msgs: _mk_ai(
                {
                    "type": "CANDIDATES",
                    "stop_suggest": "STOP",
                    "items": [
                        {"symbol": "510300", "score": 80.0, "reason": "x", "source_skill": "demo", "extra": {}}
                    ],
                }
            ),
            tool_node=None,
            postprocess=g.postprocess_hunter,
        )

        auditor_rb = g.RoleBlock(
            role="auditor",
            system_prompt="AUDITOR_SYS",
            llm_invoke=lambda _msgs: _mk_ai({"type": "OBJECTIONS", "stop_suggest": "STOP", "items": []}),
            tool_node=None,
            postprocess=g.postprocess_auditor,
        )

        pm_rb = g.RoleBlock(
            role="pm",
            system_prompt="PM_SYS",
            llm_invoke=lambda _msgs: _mk_ai({"type": "DECISIONS", "stop_suggest": "STOP", "items": []}),
            tool_node=None,
            postprocess=g.postprocess_pm,
        )

        graph = g.build_etf_attack_patch_graph(hunter=hunter_rb, auditor=auditor_rb, pm=pm_rb)

        init_state = {
            "messages": [],
            "round_idx": 0,
            "stable_rounds": 0,
            "tool_trace": [],
            "tool_cache": {},
            "candidates_cur": [],
            "objections_cur": [],
            "diff_cur": {"type": "DIFF", "items": []},
            "risk_reports": [],
            "survivor_universe": [],
            "_round_tool_calls_ok": {},
            "_hunter_round_sniper_strategies": [],
        }

        out = graph.invoke(init_state)

        assert out.get("_last_speaker_role") == "pm"

        assert out.get("round_idx") == 0

        assert any(it.get("symbol") == "510300" for it in (out.get("candidates_cur") or []))

        ai_n = len([m for m in (out.get("messages") or []) if isinstance(m, AIMessage)])
        assert ai_n >= 3

        assert out.get("stop_reason") in (None, "MAX_ROUNDS_DEBATE")

   ```

   </details>

2) 运行测试
   
```bash
uv run pytest -q tests/test_graph.py
```


### ✅ 验收标准 Pass

- 终端输出类似下面信息（数字可能不同，但核心是 **passed**）
  - `12 passed in ...s`
- 过程中没有出现 `ImportError`、`FrozenInstanceError`、`KeyError`、`AssertionError`
- 如果失败，你应该能从报错快速定位到三类问题：
  - **配置不可 patch（冻结对象）**  
    - 报错形态：`dataclasses.FrozenInstanceError: cannot assign to field 'MAX_ROUNDS'`
    - 说明：测试里不应再直接 `monkeypatch.setattr(CONFIG, ...)`；需要用“配置读取函数/包装器”或在测试里 patch graph 模块内部读取口（你现在的 test_graph 已经这么做了）。
  - **Graph 主循环没按预期闭环**  
    - 报错形态：端到端断言不通过（例如 `_last_speaker_role != "pm"`、`round_idx` 意外变化）
    - 说明：多半是边没连对（`hunter_post -> auditor_agent`、`auditor_post -> (next_round|pm)`、`pm_post -> END`），或 tools 分支导致回边不正确。
  - **路由函数副作用不回写**  
    - 报错形态：端到端里 `stop_reason is None`
    - 说明：`_should_end_debate` 作为 conditional route 在部分 LangGraph 版本里不会把“对 state 的就地写入”合并回最终输出。  
      处理：端到端测试不要强依赖 `stop_reason`；`stop_reason` 的正确性由 `_should_end_debate_*` 单测覆盖。


### 🔁 可迁移点 Transfer

> 本关的 `core/graph.py` 设计目标是：**图编排（Graph）稳定、业务对象（ETF）可替换**。迁移到别的任务时，你不需要重写 Debate MAS，只要把“业务 postprocess + payload 类型 + gate 规则”替换掉。

**1. 框架通用 不要动**

这些是任何 “多角色 attack/patch → 收敛 → 最终决策” 图都离不开的骨架,迁移到别的业务时，建议保持结构不变。

<details>
<summary><b>graph.py 不需要动的地方</b></summary>

- **Graph 主循环骨架**
  - `build_*_graph()`：负责 `roleA → roleB → (next_round | final_role) → END` 的编排
  - 原则：节点命名与连边稳定，便于测试与 debug

- **角色节点抽象**
  - `RoleBlock(role/system_prompt/llm_invoke/tool_node/postprocess)`
  - 原则：把“角色差异”都塞进 RoleBlock，graph 只负责拼装

- **system 置顶与 tools 路由**
  - `_append_system_prompt()`：每轮把 system prompt 置顶
  - `_last_ai_has_tool_calls()`：严格判定才走 tools，避免死循环

- **payload 抽取 + stop_suggest 读取**
  - `_extract_last_payload()`：兼容“Debate + 末尾 JSON”/“纯 JSON”
  - `_get_stop_suggest()`：统一大小写，供裁决器使用

- **收敛裁决器接口**
  - `_should_end_debate()`：只负责“下一跳决策”，保持返回 `"next_round"` / `"pm"` 的稳定协议。
- **三段落地模式**
  - `postprocess_hunter / postprocess_auditor / postprocess_pm`：固定“落地 cur + 写 history + 写 stop_suggest”。

</details>

**2. 业务相关 可替换或重写**

下面这些内容的**思想是通用的**，但字段名、合并 key、排序规则、pipeline 状态位通常和业务强绑定。  

迁移到别的任务时，**允许你改它们**，但建议保持“写 cur + 写 history”的模式不变。

- **候选合并策略**
  - `push_candidates_merge()`（state侧）：ETF 用 `symbol` 合并、按 `score` 排序。
  - 换业务时，把 `symbol/score` 换成你的主键与优先级字段即可。

  <details>
  <summary><b>示例 TODO：把 ETF 候选合并改成“方案评审”的提案合并</b></summary>

  ```py
    # TODO：方案评审场景（proposal_id + priority）
    def push_candidates_merge(st, incoming):
        # 1) 以 proposal_id 为 key 合并（incoming 覆盖同 id）
        # 2) 以 priority 由高到低排序
        # 3) 写回 candidates_cur，并 append history["candidates"]
        pass
  ```
  </details>

- **硬剔除/存活池规则**
  - `_compute_survivor_universe()`：ETF 用 objections(REJECT) + risk_reports(illiquid/high_risk) 做硬剔除。
  - 换业务时：保留“输入来自 cur 状态、输出是一个可复用子集”的结构即可。

  <details>
  <summary><b>示例 TODO：把存活池改成“合同审阅”的可接受条款集合</b></summary>

  ```py
    # TODO：合同审阅场景（clause_id + risk_flag）
    def compute_survivors(st):
        # 1) 从 candidates_cur 取 clause_id
        # 2) 剔除 objections_cur 中 verdict=="REJECT" 的 clause_id
        # 3) 输出 survivors（去重保序）
        pass
  ```
  </details>

- **稳定轮数/收敛判断的“指纹内容”**
  - `bump_stable_rounds()` 默认用 candidates + objections + diff 做指纹（你项目里可能在 state.py）。
  - 换业务时：改“指纹包含哪些字段”，让“稳定”符合业务定义。

  <details>
  <summary><b>示例 TODO：把稳定判断改成“舆情研判”的观点一致性</b></summary>

  ```py
    # TODO：舆情研判场景（only stance + evidence_summary）
    def bump_stable_rounds(st, reset_if_changed=True):
        # 1) fingerprint 只看 st["objections_cur"] 的 stance 字段
        # 2) 若 fingerprint 相同 -> stable_rounds += 1
        # 3) 否则按 reset_if_changed 决定是否清零
        pass
  ```
  </details>

- **跨轮控制位（pipeline / 强制工具调用 / 补齐候选）**
  - ETF 的 `_hunter_pipeline_stage / _need_recall_diversity* / _need_rerank_composite* / _force_hunter_tool / _need_more_candidates` 等。
  - 迁移到别的业务：
    - 你可以删掉它们（如果不需要 pipeline）
    - 或替换成自己的阶段状态位
    - 但要保持“默认初始化合理，不让 graph KeyError”

**‼️迁移时的“只改哪里”口诀**
  - **不动**：
    - `RoleBlock` 契约
    - `messages` 驱动的主循环
    - `_extract_last_payload()` 的“末尾 JSON + validate”机制
    - 三段 postprocess 的“写 cur + 写 history + 写 stop_suggest”模式
  - **可换**：
    - “业务对象”字段：`symbol/score/reason` → 你的 `id/priority/summary`
    - `push_*` 的合并 key 与排序规则
    - `_compute_survivor_universe()` 的硬剔除逻辑
    - `DIFF` 的 patch 类型集合
    - pipeline 状态位与推进规则

</details>

---


## 关卡-07｜引擎串联 Engine：最小循环

<details>
<summary><b>Checkpoint 07 — 引擎串联 【详情】</b></summary>

> 本关把关卡-06 的 **Graph（可控流程）** 接到 **Engine（运行入口）** 上，形成一次可跑通的最小闭环：  
> **准备输入 → 初始化 state → 组装 RoleBlocks → 构建并运行 Graph → 渲染并落盘 → 返回 artifacts**  
>
> 这一关不追求“更聪明”，只追求 **更稳定、更可测、更可迁移**：  
> - 入口函数 `run()` 能跑完  
> - 三段纯组装函数把复杂度拆开（便于 stub / 测试）  
> - 最少一个落盘产物


### 🎯 目标收获 Outcome
- 理解并实现 **Engine 最小闭环**：`run()` 串联 dossier/state/prompts/tools/llm/graph/renderer
- 学会把复杂 run 拆成 **3 个纯组装函数**
- 跑通一次任务后，能返回 `artifacts: Dict[str, str]`，并至少生成一个落盘文件（如 transcript/json）


### 🧱 约束契约 Contract
- 本关只改：`src/debate_mas/core/engine.py`
- 不改：`core/graph.py` / `core/state.py` / `protocol` / `skills` 的接口契约；不引入新依赖
- 不重构目录结构；只做“最小闭环能跑完”的填空实现
- 允许用 monkeypatch 在测试中 stub 外部依赖（LLM / loader / renderer / skills 加载）


### 🗺️ 任务清单（TODO Map）

#### **必看**
- `src/debate_mas/core/engine.py`：本关主文件（run 入口 + 三段组装函数 + transcript 落盘 + renderer 输出）
- `src/debate_mas/core/graph.py`：Graph 编排闭环（engine 只负责“组装并跑”，不应重写 graph 逻辑）
- `src/debate_mas/core/state.py`：`init_state(...)` 与 state 结构约定（engine 必须按约定初始化 key，避免 KeyError）
- `src/debate_mas/core/config.py`：CONFIG 读取（role 模型名/温度/max_tokens/verbose/data_dir 等）
- `src/debate_mas/core/personas.py`：`build_role_prompts_etf(...)`（engine 负责把 prompts 接入 RoleBlock）
- `src/debate_mas/core/tools.py`：`build_role_tools_and_node(...)`（engine 负责把 tools + tool_node 接入 RoleBlock）
- `src/debate_mas/loader/dual_mode_loader.py`：`DualModeLoader.load_from_folder(...)`（案卷入口）
- `src/debate_mas/skills/registry.py`：`SkillRegistry.load_all_skills(...)`（run 开始必须先加载 skills）
- `src/debate_mas/protocol/renderer.py`：`DebateRenderer.render(...)`（最终产出 artifacts 的地方）


#### **必写（框架通用）**
- **三段组装函数结构必须稳定**
  - `_setup_dossier_and_state()`：只负责“加载 dossier + init_state”
  - `_setup_prompts_tools_llms()`：只负责“prompts/tools/llm → RoleBlock”
  - `_run_graph_and_render()`：只负责“跑图 → 渲染 → 落盘 → 返回 artifacts”
- **run() 入口必须稳定**
  - run 只做：load skills → 组装三段 → return artifacts
- **transcript 可审计输出**
  - `_serialize_messages()`：把 messages 转成 list[dict]（含 tool_calls / name / tool_call_id）
  - `_infer_role()`：最小角色推断稳定（user/assistant/tool/unknown）
- **coerce 兜底必须稳定**
  - `_coerce_decisions()`：把 pm 输出统一成 schema 列表（避免 renderer 输入不稳定）
  - `_coerce_tool_trace()`：补齐 trace 默认字段（避免 renderer KeyError）


#### **必写（ETF任务相关，迁移可替换）**
- **候选融合留痕**
  - `merge_candidates(...)`：把 graph 输出的候选改“最终融合版”
  - `explain_merge(...)`：生成 merge_notes 放进 extra_meta（便于教学/复盘）
- **renderer 的 extra_meta 打包**
  - 至少包含：`stop_reason / tool_trace / transcript / candidates_cur / objections_cur / diff_cur`
- **transcript 落盘**
  - 每次 run 后至少写出一个 `*_transcript.json`（并把路径写进 artifacts）


#### **选改（拓展位）**
- **verbose_summary（stream 增量打印）**
  - 用 `app.stream(..., stream_mode="values")` 打印增量 tool_trace/messages（教学演示用）
- **外部依赖更强健**
  - `_build_llm()` 支持更多 provider/环境变量组合（但不要改 run() 的对外签名）
- **更严格的参数校验**
  - mission 非空、output_dir 可写、folder_path 存在等（失败要给可操作的错误信息）
- **更丰富的落盘产物**
  - 除 transcript 外追加：config_snapshot、dossier 摘要、top candidates 摘要等（都放在 extra_meta.extras）

<details>
<summary><b>📄 Checkpoint-07：core/engine.py 练习骨架</b></summary>

```py
# src/debate_mas/core/engine.py
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

from debate_mas.loader.dual_mode_loader import DualModeLoader
from debate_mas.skills.registry import SkillRegistry
from debate_mas.protocol.renderer import DebateRenderer
from debate_mas.protocol.schema import EtfDecision

from .config import CONFIG
from .state import init_state, DebateState
from .personas import build_role_prompts_etf

from .graph import (
    RoleBlock,
    build_etf_attack_patch_graph,
    postprocess_hunter,
    postprocess_auditor,
    postprocess_pm,
)

from .blend_rank import merge_candidates, explain_merge
from .tools import build_role_tools_and_node

load_dotenv()

# ============================================================
# 0) LLM / coercion helpers
# ============================================================

def _build_llm(
    model_name: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 4000,
) -> ChatOpenAI:
    """
    统一 LLM 构造器：构建 ChatOpenAI 客户端。

    TODO【必写（通用框架）】:
      - 从环境变量读取 api_key / api_base（字段名按项目约定）
      - 缺失时 raise RuntimeError（报错信息要能指导排查）
      - 返回 ChatOpenAI(...)

    TODO【选改（拓展位）】:
      - 对 temperature/max_tokens 做轻量边界处理

    Args:
        model_name: 模型名
        temperature: 采样温度
        max_tokens: 最大输出 token

    Returns:
        llm: ChatOpenAI 客户端
    """
    # TODO
    raise NotImplementedError


def _coerce_decisions(decisions: List[Dict[str, Any]]) -> List[EtfDecision]:
    """
    把 PM 输出统一转成 EtfDecision 列表（schema 归一化）。

    TODO【必写（通用框架）】:
      - 入参可能是 list[dict] / list[EtfDecision] 混合
      - 统一转成 list[EtfDecision]
      - 忽略无法转换的条目（或选择 raise，二选一保持一致）

    Args:
        decisions: 原始 decisions 列表

    Returns:
        out: EtfDecision 列表
    """
    # TODO
    raise NotImplementedError


def _coerce_tool_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    tool_trace 字段补齐：保证 renderer 不因缺字段崩溃。

    TODO【必写（通用框架）】:
      - 只保留 dict 条目
      - 为关键字段 setdefault：kind/args/ok/denied/insight/error_msg/visuals/produced_n/elapsed_ms/round_idx/role
      - 返回补齐后的 trace

    Args:
        trace: 原始 trace

    Returns:
        out: 规范化 trace
    """
    # TODO
    raise NotImplementedError


# ============================================================
# 1) Transcript 序列化工具
# ============================================================

def _infer_role(m: BaseMessage) -> str:
    """
    推断消息角色：user / assistant / tool / unknown。

    TODO【必写（通用框架）】:
      - isinstance(HumanMessage) -> "user"
      - isinstance(AIMessage) -> "assistant"
      - isinstance(ToolMessage) -> "tool"
      - 兜底：读 m.type 或返回 "unknown"

    Args:
        m: BaseMessage

    Returns:
        role: str
    """
    # TODO
    raise NotImplementedError


def _serialize_messages(msgs: List[BaseMessage]) -> List[Dict[str, Any]]:
    """
    把 messages 序列化成可落盘的 transcript。

    TODO【必写（通用框架）】:
      - 输出 list[dict]，至少包含 role/content
      - 若存在 tool_calls（两种存储风格）写入 tool_calls
      - 若存在 name/tool_call_id 也写入

    Args:
        msgs: messages

    Returns:
        out: transcript list
    """
    # TODO
    raise NotImplementedError


# ============================================================
# 2) run() 的 3 个“纯组装函数”
# ============================================================

def _setup_dossier_and_state(
    *,
    mission: str,
    ref_date: Optional[str],
    folder_path: Optional[str],
    seed_user_message: Optional[str],
) -> Tuple[Any, DebateState]:
    """
    准备 dossier + 初始化 DebateState（只做组装，不跑图）。

    TODO【必写（通用框架）】:
      - 使用 DualModeLoader 从 folder_path 加载 dossier（folder_path None 则走默认）
      - seed_user_message 若存在：放入 messages（HumanMessage）
      - 调用 init_state(...) 生成 st

    TODO【必写（ETF任务相关）】:
      - folder_path 默认用 CONFIG.DATA_DIR
      - init_state 需写入 mission/ref_date/dossier/messages

    Args:
        mission: 任务描述
        ref_date: 参考日期（可选）
        folder_path: 案卷目录（可选）
        seed_user_message: 种子消息（可选）

    Returns:
        dossier: 案卷对象
        st: DebateState
    """
    # TODO
    raise NotImplementedError


def _setup_prompts_tools_llms(
    *,
    mission: str,
    dossier: Any,
    ref_date: Optional[str],
    st: DebateState,
) -> Tuple[Dict[str, str], RoleBlock, RoleBlock, RoleBlock]:
    """
    准备 prompts + tools + llms，并组装成 RoleBlock（三角色）。

    TODO【必写（通用框架）】:
      - prompts = build_role_prompts_etf(...)
      - 为 hunter/auditor/pm 构建 tools + tool_node
      - 为三角色构建 llm，并 bind_tools
      - 组装 RoleBlock：role/system_prompt/llm_invoke/tool_node/postprocess

    TODO【必写（ETF任务相关）】:
      - prompts 需要传入 allowlist_by_role（来自 CONFIG）
      - postprocess 分别使用 postprocess_hunter/postprocess_auditor/postprocess_pm

    TODO【选改（拓展位）】:
      - per-role temperature / max_tokens：从 CONFIG 读取并设置默认值
      - tool_node 允许为 None（测试场景）

    Args:
        mission: 任务描述
        dossier: 案卷对象
        ref_date: 参考日期（可选）
        st: DebateState（用于 tools 构建/上下文）

    Returns:
        prompts: role -> prompt 文本
        hunter_block: RoleBlock
        auditor_block: RoleBlock
        pm_block: RoleBlock
    """
    # TODO
    raise NotImplementedError


def _run_graph_and_render(
    *,
    mission: str,
    ref_date: Optional[str],
    output_dir: str,
    st: DebateState,
    hunter_block: RoleBlock,
    auditor_block: RoleBlock,
    pm_block: RoleBlock,
    verbose_summary: bool,
) -> Dict[str, str]:
    """
    跑图 + 渲染输出：本关最小闭环执行点。

    TODO【必写（通用框架）】:
      - app = build_etf_attack_patch_graph(...)
      - verbose_summary=False：直接 invoke(st) 得 final_state
      - verbose_summary=True：可选 stream 打印（可先留空或最小实现）
      - 序列化 messages -> transcript
      - renderer.render(...) 返回 artifacts

    TODO【必写（ETF任务相关）】:
      - candidates 合并留痕（merge + explain），写回 final_state
      - extra_meta 打包（至少包含 stop_reason / transcript / cur 状态快照）
      - transcript 以 json 落盘到 output_dir，并把路径塞进 artifacts

    TODO【选改（拓展位）】:
      - verbose 增量打印：tool_trace/messages 增量摘要
      - 错误兜底：落盘失败不应中断主流程（可打印 warning）

    Args:
        mission: 任务描述
        ref_date: 参考日期（可选）
        output_dir: 输出目录
        st: 初始 DebateState
        hunter_block/auditor_block/pm_block: RoleBlock
        verbose_summary: 是否 stream 增量摘要

    Returns:
        artifacts: Dict[str, str]
    """
    # TODO
    raise NotImplementedError


# ============================================================
# 3) 一键运行入口
# ============================================================

def run(
    mission: str,
    *,
    ref_date: Optional[str] = None,
    folder_path: Optional[str] = None,
    output_dir: str = "./output_reports",
    seed_user_message: Optional[str] = None,
) -> Dict[str, str]:
    """
    一键运行入口（对外 API）。

    TODO【必写（通用框架）】:
      - SkillRegistry.load_all_skills(...)
      - dossier, st = _setup_dossier_and_state(...)
      - prompts, hunter_block, auditor_block, pm_block = _setup_prompts_tools_llms(...)
      - return _run_graph_and_render(...)

    TODO【必写（ETF任务相关）】:
      - verbose_summary 读取 CONFIG.VERBOSE
      - output_dir 默认 "./output_reports"

    TODO【选改（拓展位）】:
      - 参数轻量校验（mission 非空）
      - output_dir mkdir(exist_ok=True)

    Args:
        mission: 任务描述
        ref_date: 参考日期（可选）
        folder_path: 案卷目录（可选）
        output_dir: 输出目录
        seed_user_message: 种子消息（可选）

    Returns:
        artifacts: Dict[str, str]
    """
    # TODO
    raise NotImplementedError

```
</details>


### ▶️ 执行命令 Run

本关用 **pytest** 做最小验收。

1) 新建测试文件：`tests/test_engine.py`
   把下面代码完整复制进去：

   <details>
   <summary><b>tests/test_engine.py</b></summary>

   ```py
    import json
    from types import SimpleNamespace
    from pathlib import Path

    import pytest
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from debate_mas.core import engine as e



    def _patch_config(monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
        """
        直接 monkeypatch 模块变量 e.CONFIG 为一个可变对象（stub），避免 FrozenInstanceError。
        """
        cfg = SimpleNamespace(
            DATA_DIR="__DATA_DIR__",
            VERBOSE=False,

            HUNTER_MODEL="stub-hunter",
            AUDITOR_MODEL="stub-auditor",
            PM_MODEL="stub-pm",

            ROLE_TEMPERATURE={"hunter": 0.9, "auditor": 0.3, "pm": 0.1},
            ROLE_MAX_TOKENS={"hunter": 1200, "auditor": 800, "pm": 800},
            MAX_TOKENS_DEFAULT=1000,

            ROLE_TOOL_ALLOWLIST={"hunter": [], "auditor": [], "pm": []},

            HUNTER_BLEND={"demo": 1.0},
        )

        def _get_model_config():
            return {"stub": True}

        cfg.get_model_config = _get_model_config

        for k, v in overrides.items():
            setattr(cfg, k, v)

        monkeypatch.setattr(e, "CONFIG", cfg, raising=True)


    class DummyLLM:
        """
        最小 LLM stub：
        - bind_tools(tools) -> self
        - invoke(messages) -> AIMessage
        """
        def __init__(self, role: str):
            self.role = role
            self.bound_tools = None

        def bind_tools(self, tools):
            self.bound_tools = tools
            return self

        def invoke(self, _messages):
            return AIMessage(content=f"[{self.role}] ok")


    def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_build_llm(model_name: str, *, temperature: float = 0.2, max_tokens: int = 4000):
            role = "unknown"
            if "hunter" in model_name: role = "hunter"
            if "auditor" in model_name: role = "auditor"
            if "pm" in model_name: role = "pm"
            return DummyLLM(role=role)

        monkeypatch.setattr(e, "_build_llm", fake_build_llm, raising=True)


    def _patch_loader_and_state(monkeypatch: pytest.MonkeyPatch):
        calls = {"load": [], "init_state": []}

        class DummyLoader:
            def load_from_folder(self, *, mission: str, folder_path: str):
                calls["load"].append({"mission": mission, "folder_path": folder_path})
                return {"dossier": True, "folder": folder_path}

        def fake_init_state(*, mission, dossier, ref_date, messages):
            calls["init_state"].append(
                {"mission": mission, "dossier": dossier, "ref_date": ref_date, "messages": messages}
            )
            return {
                "mission": mission,
                "ref_date": ref_date,
                "dossier": dossier,
                "dossier_view": {"meta": "stub"},
                "messages": list(messages or []),

                "round_idx": 0,
                "stable_rounds": 0,
                "tool_trace": [],
                "tool_cache": {},

                "candidates_cur": [],
                "objections_cur": [],
                "diff_cur": {"type": "DIFF", "items": []},
                "risk_reports": [],
                "survivor_universe": [],

                "decisions": [],
            }

        monkeypatch.setattr(e, "DualModeLoader", DummyLoader, raising=True)
        monkeypatch.setattr(e, "init_state", fake_init_state, raising=True)
        return calls


    def _patch_prompts_tools(monkeypatch: pytest.MonkeyPatch):
        calls = {"prompts": [], "tools": []}

        def fake_build_prompts_etf(*, mission, dossier_view, allowlist_by_role):
            calls["prompts"].append(
                {"mission": mission, "dossier_view": dossier_view, "allowlist_by_role": allowlist_by_role}
            )
            return {"hunter": "HUNTER_SYS", "auditor": "AUDITOR_SYS", "pm": "PM_SYS"}

        def fake_build_role_tools_and_node(*, role, dossier, ref_date, state):
            calls["tools"].append({"role": role, "ref_date": ref_date})
            return ([], None, None)

        monkeypatch.setattr(e, "build_role_prompts_etf", fake_build_prompts_etf, raising=True)
        monkeypatch.setattr(e, "build_role_tools_and_node", fake_build_role_tools_and_node, raising=True)
        return calls


    def _patch_graph_and_renderer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """
        让 _run_graph_and_render 真正跑到：
        - build graph -> invoke -> final_state
        - transcript 序列化 + 落盘
        - renderer.render 返回 artifacts
        """
        class DummyGraph:
            def __init__(self, final_state):
                self._final_state = final_state

            def invoke(self, _st):
                return self._final_state

            def stream(self, _st, stream_mode="values"):
                yield self._final_state

        def fake_build_graph(*, hunter, auditor, pm):
            final_state = {
                "round_idx": 0,
                "stable_rounds": 0,
                "stop_reason": "MAX_ROUNDS_DEBATE",

                "_last_speaker_role": "pm",

                "messages": [
                    AIMessage(content='{"type":"CANDIDATES","stop_suggest":"STOP","items":[{"symbol":"510300","score":80.0,"reason":"x","source_skill":"demo","extra":{}}]}'),
                    AIMessage(content='{"type":"OBJECTIONS","stop_suggest":"STOP","items":[]}'),
                    AIMessage(content='{"type":"DECISIONS","stop_suggest":"STOP","items":[]}'),
                ],

                "tool_trace": [{"kind": "trace", "role": "system", "tool": "__test__", "ok": True, "insight": "ok"}],
                "tool_cache": {},

                "candidates_cur": [{"symbol": "510300", "score": 80.0, "reason": "x", "source_skill": "demo", "extra": {}}],
                "objections_cur": [],
                "diff_cur": {"type": "DIFF", "items": []},

                "decisions": [],  
                "dossier_view": {"meta": "stub"},
            }
            return DummyGraph(final_state)

        class FakeRenderer:
            def __init__(self, output_dir: str):
                self.output_dir = output_dir

            def render(self, *, mission: str, decisions, extra_meta):
                return {"memo": str(Path(self.output_dir) / "memo.md")}

        def fake_merge_candidates(cands_list, source_weights=None):
            if not cands_list:
                return []
            x = cands_list[0] or []
            return list(x)

        def fake_explain_merge(_merged, top_n=5):
            return "merge_notes: stub"

        monkeypatch.setattr(e, "build_etf_attack_patch_graph", fake_build_graph, raising=True)
        monkeypatch.setattr(e, "DebateRenderer", FakeRenderer, raising=True)
        monkeypatch.setattr(e, "merge_candidates", fake_merge_candidates, raising=True)
        monkeypatch.setattr(e, "explain_merge", fake_explain_merge, raising=True)


    def _patch_skill_registry(monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(e.SkillRegistry, "load_all_skills", lambda force_reload=False: None, raising=True)


    def test_infer_role_and_serialize_messages():
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
            ToolMessage(content="tool ok", tool_call_id="t1"),
        ]

        out = e._serialize_messages(msgs)

        assert out[0]["role"] == "user"
        assert out[1]["role"] == "assistant"
        assert out[2]["role"] == "tool"
        assert out[2]["tool_call_id"] == "t1"


    def test_serialize_messages_captures_tool_calls_in_both_styles():
        m1 = AIMessage(
            content="x",
            additional_kwargs={
                "tool_calls": [{"id": "1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}]
            },
        )
        m2 = AIMessage(content="y")
        setattr(m2, "tool_calls", [{"name": "bar"}])

        out = e._serialize_messages([m1, m2])
        assert "tool_calls" in out[0]
        assert "tool_calls" in out[1]


    def test_setup_dossier_and_state_injects_seed_message_and_defaults(monkeypatch: pytest.MonkeyPatch):
        _patch_config(monkeypatch, DATA_DIR="DATA_DEFAULT")
        calls = _patch_loader_and_state(monkeypatch)

        dossier, st = e._setup_dossier_and_state(
            mission="m",
            ref_date="2025-10-26",
            folder_path=None, 
            seed_user_message="seed",
        )

        assert dossier["folder"] == "DATA_DEFAULT"
        assert isinstance(st["messages"][0], HumanMessage)
        assert st["messages"][0].content == "seed"

        assert calls["load"][0]["folder_path"] == "DATA_DEFAULT"
        assert calls["init_state"][0]["mission"] == "m"
        assert calls["init_state"][0]["ref_date"] == "2025-10-26"


    def test_setup_prompts_tools_llms_builds_roleblocks(monkeypatch: pytest.MonkeyPatch):
        _patch_config(monkeypatch)
        _patch_llm(monkeypatch)
        calls = _patch_prompts_tools(monkeypatch)

        st = {"dossier_view": {"meta": "stub"}}
        prompts, hunter_rb, auditor_rb, pm_rb = e._setup_prompts_tools_llms(
            mission="m",
            dossier={"dossier": True},
            ref_date="2025-10-26",
            st=st,
        )

        assert prompts["hunter"] == "HUNTER_SYS"
        assert hunter_rb.role == "hunter"
        assert auditor_rb.role == "auditor"
        assert pm_rb.role == "pm"

        assert len(calls["prompts"]) == 1
        assert {c["role"] for c in calls["tools"]} == {"hunter", "auditor", "pm"}

        assert isinstance(hunter_rb.llm_invoke([]), AIMessage)


    def test_run_graph_and_render_creates_transcript_and_returns_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        _patch_config(monkeypatch, VERBOSE=False)
        _patch_graph_and_renderer(monkeypatch, tmp_path)

        st = {
            "messages": [],
            "tool_trace": [],
            "tool_cache": {},
            "candidates_cur": [],
            "objections_cur": [],
            "diff_cur": {"type": "DIFF", "items": []},
            "dossier_view": {"meta": "stub"},
            "round_idx": 0,
            "stable_rounds": 0,
        }

        rb = e.RoleBlock(
            role="hunter",
            system_prompt="SYS",
            llm_invoke=lambda _ms: AIMessage(content="x"),
            tool_node=None,
            postprocess=lambda _st: None,
        )

        artifacts = e._run_graph_and_render(
            mission="m",
            ref_date="2025-10-26",
            output_dir=str(tmp_path),
            st=st,
            hunter_block=rb,
            auditor_block=rb,
            pm_block=rb,
            verbose_summary=False,
        )

        assert isinstance(artifacts, dict)
        assert "memo" in artifacts

        transcript_path = artifacts.get("transcript")
        if transcript_path:
            p = Path(transcript_path)
            assert p.exists()
            data = json.loads(p.read_text(encoding="utf-8"))
            assert data["mission"] == "m"
            assert data["ref_date"] == "2025-10-26"
            assert isinstance(data["transcript"], list)
        else:
            files = list(Path(tmp_path).glob("*_transcript.json"))
            assert len(files) >= 1


    def test_run_entrypoint_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        _patch_config(monkeypatch, VERBOSE=False, DATA_DIR="DATA_DEFAULT")
        _patch_skill_registry(monkeypatch)
        _patch_llm(monkeypatch)
        _patch_loader_and_state(monkeypatch)
        _patch_prompts_tools(monkeypatch)
        _patch_graph_and_renderer(monkeypatch, tmp_path)

        artifacts = e.run(
            "m",
            ref_date="2025-10-26",
            folder_path=None,
            output_dir=str(tmp_path),
            seed_user_message="seed",
        )

        assert isinstance(artifacts, dict)
        assert "memo" in artifacts
   ```

   </details>

2) 运行测试
   
```bash
uv run pytest -q tests/test_engine.py
```


### ✅ 验收标准 Pass

- 终端输出类似下面信息（数字可能不同，但核心是 **passed**）
  - `6 passed in ...s`
- 过程中没有出现 `ImportError`、`FrozenInstanceError`、`KeyError`、`RuntimeError: 缺少环境变量...`
- 如果失败，你应该能从报错快速定位到三类问题：
  - **组装没串起来**：`run()` 没按“三段组装函数”调用，或 RoleBlock 没正确构建
  - **落盘不稳定**：transcript 没写出 / 路径没回填 artifacts
  - **依赖耦合太深**：测试无法 stub（说明 engine 里把“组装”与“业务细节/外部 I/O”缠在一起）


### 🔁 可迁移点 Transfer

> 本关的 `core/engine.py` 设计目标是：**把外部世界（dossier/tools/llm/renderer）粘起来，但不把业务写死。**
> 
> 迁移到别的任务时，你通常只需要换“数据入口/角色 prompts/工具/渲染器”，而不必重写主流程。

**1. 框架通用 不要动**

<details>
<summary><b>engine.py 不需要动的地方</b></summary>

- **run() 对外签名**
  - `run(mission, ref_date=None, folder_path=None, output_dir=..., seed_user_message=None)`
  - 原则：保持“教学友好 + 可脚本化调用”的统一入口

- **三段组装函数结构**
  - `_setup_dossier_and_state()`：只负责 dossier + init_state
  - `_setup_prompts_tools_llms()`：只负责 prompts/tools/llm → RoleBlock
  - `_run_graph_and_render()`：只负责跑图 + transcript + renderer

- **transcript 序列化与落盘**
  - `_infer_role() / _serialize_messages()`
  - 原则：可审计、可复盘（哪怕换业务也值得保留）

- **coerce 兜底**
  - `_coerce_decisions() / _coerce_tool_trace()`
  - 原则：保证 renderer 输入稳定，不要让下游因为字段缺失崩溃

</details>

**2. 业务相关 可替换或重写**

这部分属于“你们当前是 ETF 的示例业务”。  

迁移到别的任务时可以**整体替换**，但建议保留同一条原则：**强类型（Strong Types）+ 可校验（Validatable）**，让 Graph/Engine 的输入输出长期稳定、可测试。

- **候选合并策略**
  - `EtfCandidate / EtfRiskReport / EtfDecision`
  - 迁移方式：整体替换成你的业务对象，例如：
    - 合同审阅：`ClauseIssue / ClauseChange / ContractDecision`
    - 方案评审：`Proposal / ReviewRisk / ReviewDecision`
    - 宏观决策：`MacroSignal / RiskState / AllocationDecision`

  <details>
  <summary><b>示例：把 EtfDecision 换成“资产配置决策”对象</b></summary>

  ```py
    # 示例：AllocationDecision（可替换 EtfDecision）
    from pydantic import BaseModel, Field
    from typing import List, Literal

    DecisionAction = Literal["BUY", "SELL", "HOLD", "REBALANCE"]

    class AllocationDecision(BaseModel):
        asset: str
        action: DecisionAction
        target_weight: float = Field(0.0, ge=0.0, le=1.0)
        confidence: float = Field(0.0, ge=0.0, le=1.0)
        reasons: List[str] = Field(default_factory=list)
        risk_notes: List[str] = Field(default_factory=list)
  ```
  </details>

- **输入载体：dossier**
  - ETF 当前用 `DualModeLoader().load_from_folder(...)` 从文件夹加载
  - 迁移方式：替换加载器即可（DB/API/消息队列都行），但**保持输出仍是 dossier 对象**（graph/tools 只透传，不强依赖底层存储）

  <details>
  <summary><b>示例 TODO：把 dossier 来源从“文件夹”换成“数据库/接口”</b></summary>

  ```py
    # TODO：改造 _setup_dossier_and_state 的 dossier 加载
    def _setup_dossier_and_state(...):
        # 1) 用你的 DBLoader / APIClient 替代 DualModeLoader
        # 2) 保持返回仍是 dossier（graph/tools 只透传）
        # 3) init_state 的 mission/ref_date/messages 结构不变
        pass
  ```
  </details>

- **角色设定：prompts**
  - ETF 当前用 `build_role_prompts_etf(...)` 生成 hunter/auditor/pm 的 system prompts
  - 迁移方式：替换 prompts 生成器 + 对应 postprocess

  <details>
  <summary><b>示例 TODO：把三角色 prompts 改成你定义的角色</b></summary>

  ```py
    # TODO：改造 _setup_prompts_tools_llms 的 prompts 构造
    def _setup_prompts_tools_llms(...):
        # 1) 用 build_role_prompts_xxx 替换 build_role_prompts_etf
        # 2) RoleBlock 三件套不变：system_prompt / llm_invoke / postprocess
        # 3) postprocess_* 替换成你的业务落地逻辑（写 cur + history）
        pass
  ```
  </details>

- **产物输出：renderer + artifacts**
  - ETF 当前用 `DebateRenderer.render(...)` 输出 memo/CSV 等，并把 transcript/extra_meta 放进可复盘产物
  - 迁移方式：替换 Renderer（docx/json/dashboard/数据库写入均可），但建议继续输出 `artifacts: Dict[str, str]` 作为统一“交付索引”

  <details>
  <summary><b>示例 TODO：把输出从“投资报告”换成“你的产物格式”</b></summary>

  ```py
    # TODO：改造 _run_graph_and_render 的 renderer
    def _run_graph_and_render(...):
        # 1) 用你的 Renderer（例如输出 docx/json/dashboard）
        # 2) extra_meta 建议保留 transcript/stop_reason/tool_trace（复盘很有用）
        # 3) artifacts 仍返回 Dict[str, str]（路径或标识）
        pass
  ```
  </details>


**‼️迁移时的“只改哪里”口诀**
  - **不动**：
    - `DecisionAction / SkillResult / DebateLog` 骨架
    - `Renderer 三件套 + _build_meta 兼容`（至少能接受 transcript/stop_reason/tool_trace）
    - `RoleBlock` 的契约与 Engine 三段组装结构
  - **可换**：
    - “业务对象”字段：(`*Decision/*Candidate/*Risk`)
    - memo 文案结构
    - CSV 列集合与命名规则

</details>

---


# 阶段三：技能 (Skills)

## 关卡-08｜写 Skill：以quantitative_sniper为例

<details>
<summary><b>Checkpoint 08 — 写 Skill 【详情】</b></summary>

> 本关把一个 Skill 作为**可迁移、可教学、可审计**的“最小产品”交付
>   
> **数据契约（Data Contract）→ 指标公式（Metrics）→ 可运行逻辑（scripts）→ 输出模板（templates）→ 单测验收（pytest）**   



### 🎯 目标收获 Outcome
- 写出一个完整skill技能包：包含数据契约、公式说明、输出模板与可运行逻辑
- Skill 能通过统一接口被系统调用：`SkillHandler.execute(ctx, ...) -> SkillResult`
- 输出是**强类型 + 可校验**的结构化结果（items + meta 稳定）
- 用 pytest 单测验证：输入一份最小行情表 → 输出 top_k 候选列表


### 🧱 约束契约 Contract
- 本关改动范围：`src/debate_mas/skills/inventory/quantitative_sniper/**`
- 不改：`skills/base.py` / `skills/registry.py` / `protocol` / `core/tools.py` 的接口契约；不引入新依赖
- 必须返回结构化结果（至少包含 `data.items`），无数据/缺列必须 `fail` 且给可解释信息


### 🗺️ 任务清单（TODO Map）

#### 必看
- `src/debate_mas/skills/base.py`：Skill 统一接口（BaseFinanceSkill / SkillContext / apply_date_filter）
- `src/debate_mas/protocol`：`SkillResult / EtfCandidate` 等 schema（你的输出必须遵守）
- `src/debate_mas/core/tools.py`：Skill 如何被包装成 tool（你只要输出稳定即可）
- `src/debate_mas/skills/inventory/quantitative_sniper/`：本关技能包目录（你将补齐其“可交付形态”）

#### 技能包目录结构

```md
skills/inventory/quantitative_sniper/    # [技能] 量化狙击手 (筛选/排序)
├── __init__.py                          # 包标识
├── SKILL.md                             # 定义技能角色、参数Schema、Prompt
├── references/                          # skill说明文档
│   ├── data_contract.md                 # 数据依赖契约
│   ├── metrics.md                       # 数学公式定义
│   └── README.md                        # 技能使用/拓展说明
├── scripts/                             # 执行逻辑
│   ├── __init__.py
│   ├── dataloader.py                    # 数据获取、清洗
│   ├── algo.py                          # 策略计算逻辑
│   └── handler.py                       # 技能路由
└── templates/                           # 输出模板
    ├── output.json                      # 结构化输出模板
    └── output.md                        # 输出文本摘要模板
```

> 重点学习：  
> - `handler.py` 是“系统入口 + 契约执行者”  
> - `dataloader.py` 把“数据清洗/契约校验”独立出来（便于复用与单测）  
> - `algo.py` 只放纯计算（纯函数风格，便于替换策略/复用策略）  
> - `SKILL.md + references/* + templates/*` 让技能具备“可复盘 + 可迁移 + 可教学”的完整性


#### A) `scripts/dataloader.py`

数据加载与契约执行

**必写（框架通用）**
- 从 `ctx.dossier.get_table("etf_daily")` 读数据
- `ref_date` 防未来过滤（用 base 的 apply_date_filter 或传入）
- 列名标准化（lower + data->date）
- 类型清洗：code(str)/date(datetime)/close(float)/amount(float)
- 检查必需列：缺失则 fail
- universe 过滤：支持 list / EtfCandidate-like / str(json/逗号)

**选改（拓展位）**
- 返回 `(df, universe_size)`，让 handler 做 meta 注入

<details>
<summary><b>📄Checkpoint-08：scripts/dataloader.py练习骨架</b></summary>

```py
#src/debate_mas/skills/inventory/quantitative_sniper/scripts/dataloader.py
from __future__ import annotations

import json
from typing import Any, Callable, List, Optional, Tuple, Union

import pandas as pd

from debate_mas.protocol import SkillResult
from debate_mas.skills.base import SkillContext

REQ_COLS = ("code", "date", "close")


def normalize_universe(
    universe: Optional[Union[List[str], str, list[Any]]]
) -> Optional[List[str]]:
    """
    将 universe 入参统一归一化为 List[str]（或 None）。

    TODO【必写（通用框架）】:
      - universe 允许 None / list / str
      - list 内元素允许：str / dict / EtfCandidate-like（有 symbol 属性）
      - str 允许：JSON list 字符串 或 逗号分隔字符串
      - 去重 + strip + 过滤空值

    TODO【选改（拓展位）】:
      - 支持换行分隔（把 "\\n" 当作 ","）
      - 支持输入为单个数字/字符串的 JSON（如 '"510300"' / 510300）

    Args:
        universe: 用户传入的候选范围（None 表示全市场）

    Returns:
        codes: 归一化后的 ETF code 列表；为空则返回 None
    """
    # TODO
    raise NotImplementedError


def load_etf_daily(
    ctx: SkillContext,
    universe: Optional[Union[List[str], str, list[Any]]],
    *,
    apply_date_filter: Callable[[pd.DataFrame, str], pd.DataFrame],
) -> Tuple[pd.DataFrame, Optional[int]] | SkillResult:
    """
    从 dossier 读取 etf_daily，并执行数据契约与清洗，返回可用于策略计算的 df。

    TODO【必写（通用框架）】:
      1) 读数据：
         - df = ctx.dossier.get_table("etf_daily")
         - df None/empty -> SkillResult.fail(可解释原因)

      2) 防未来过滤：
         - 使用 apply_date_filter(df, ctx.ref_date)
         - 过滤后 empty -> SkillResult.fail(说明 ref_date 截止无数据)

      3) 列名标准化：
         - columns 全部 lower + strip
         - 兼容 data -> date（当 date 不存在而 data 存在时）

      4) 类型清洗：
         - code -> str
         - date -> datetime（errors="coerce"）
         - close -> numeric（errors="coerce"）
         - amount 若存在 -> numeric（errors="coerce"）
         - 清洗异常 -> SkillResult.fail(包含异常信息)

      5) 必需列检查：
         - 缺失 REQ_COLS 任一项 -> SkillResult.fail(明确缺哪些列)
         - dropna(subset=REQ_COLS) 后 empty -> SkillResult.fail(说明清洗后无有效数据)

      6) universe 过滤：
         - universe_list = normalize_universe(universe)
         - 若存在 universe_set：只保留 df["code"] ∈ universe_set
         - 过滤后 empty -> SkillResult.fail(明确“universe 有多少，但无匹配”)

    TODO【选改（拓展位）】:
      - 返回 universe_size（None 表示全市场；否则返回 len(universe_set)）
      - 对 df 做最小排序（例如按 date 升序），便于下游 groupby

    Args:
        ctx: SkillContext（必须能提供 dossier 与 ref_date）
        universe: 候选池（None=全市场；可为 list/str/EtfCandidate-like）
        apply_date_filter: 防未来过滤函数（由 Skill/base 提供或由上层注入）

    Returns:
        (df, universe_size):
          - df: 清洗后的行情表（至少包含 code/date/close；amount 可选）
          - universe_size: None 或 int（仅当传入 universe 且成功过滤时）
        或 SkillResult.fail(...):
          - 在缺表/缺列/清洗失败/universe 无匹配等情况下返回
    """
    # TODO
    raise NotImplementedError
```
</details>


#### B) `scripts/algo.py`

纯策略计算：df -> df_score

**必写（框架通用）**
- 每个策略输出 `df_score`，至少包含：
  - `symbol, score, reason, extra`
- score 统一映射到 0~100 percentile
- threshold 逻辑（quantile / psr）要返回 attrs.meta

**选改（拓展位）**
- 提供 `run_strategy(df, params)` 路由


<details>
<summary><b>📄Checkpoint-08：scripts/algo.py练习骨架</b></summary>

```py
# src/debate_mas/skills/inventory/quantitative_sniper/scripts/algo.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

# ============================================================
# TODO【必写（通用框架）】
# - 这里是“纯计算层”：不读 ctx、不做 fail（交给 handler/dataloader）
# - 每个策略函数都必须返回 df_score（或空 df）
# - df_score 至少包含：symbol, score, reason, extra
# - score 必须统一映射到 0~100（percentile）
# - threshold 逻辑（quantile / psr）需要把 meta 写入 df_score.attrs["threshold_meta"]
# ============================================================

# ============================================================
# TODO【选改（拓展位）】
# - 提供 run_strategy(df, params) 作为统一路由
# - 提供 user_defined_strategy(df, params) 作为练习入口
# ============================================================


def pct_rank_0_100(values: pd.Series, *, neutral: float = 50.0) -> pd.Series:
    """
    将横截面数值映射为 [0,100] 分位得分。

    TODO【必写（通用框架）】:
      - 转 numeric（errors="coerce"）
      - rank(pct=True) * 100
      - NaN 用 neutral 填充（全 NaN -> 全 neutral）

    Args:
        values: 一组横截面数值
        neutral: 缺失时的中性得分（默认 50）

    Returns:
        scores: 与 values 等长的分位得分（0~100）
    """
    # TODO
    raise NotImplementedError


def apply_threshold_quantile(
    df_score: pd.DataFrame,
    *,
    top_k: int,
    quantile_q: Optional[float],
    enabled: bool,
) -> pd.DataFrame:
    """
    分位阈值过滤：把 score 低于 cutoff 的样本过滤掉。

    TODO【必写（通用框架）】:
      - enabled=False 或 df_score empty -> 原样返回
      - quantile_q=None 时：用 top_k 自动推导一个 q
      - 计算 cutoff = score.quantile(q)
      - 过滤后不足 top_k：回退不做过滤（返回原 df_score）
      - 无论是否过滤，都应写 df_score.attrs["threshold_meta"]（包含 q/cutoff/passed/before/fallback）

    Args:
        df_score: 至少包含 score 列的数据框
        top_k: 期望返回数量
        quantile_q: 手动分位阈值（0~1）；None 表示自动推导
        enabled: 是否启用

    Returns:
        out: 过滤后的 df_score（并带 attrs["threshold_meta"]）
    """
    # TODO
    raise NotImplementedError


def apply_threshold_psr(
    df_score: pd.DataFrame,
    *,
    top_k: int,
    psr_confidence: float,
) -> pd.DataFrame:
    """
    PSR 阈值过滤：仅保留 psr >= confidence 的样本，不足 top_k 时放宽或回退。

    TODO【必写（通用框架）】:
      - 先按 psr_confidence 过滤
      - 若不足 top_k：放宽到一个较低阈值；仍为空则回退为不过滤
      - 写 attrs["threshold_meta"]（confidence/effective/fallback/passed/before）

    Args:
        df_score: 至少包含 psr 列与 score 列的数据框
        top_k: 期望返回数量
        psr_confidence: 置信度阈值（0~1）

    Returns:
        out: 过滤后的 df_score（并带 attrs["threshold_meta"]）
    """
    # TODO
    raise NotImplementedError


def scan_momentum(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    动量策略：根据窗口收益（或你定义的动量 raw）计算 score。

    TODO【必写（通用框架）】:
      - 从 params 读取 window/top_k/threshold_mode/quantile_q 等
      - 对每个 symbol 计算一个 raw 指标（如窗口收益）
      - raw -> pct_rank_0_100 -> score
      - 生成 reason（含 window 与关键 raw/pct 信息）
      - extra 字典必须包含 raw 与 pct（用于可解释与审计）
      - threshold_mode=="quantile" 时调用 apply_threshold_quantile
      - 返回 df_score（空则返回空 df）

    Args:
        df: 预处理后的行情表（至少包含 code/date/close）
        params: 策略参数（window/top_k/threshold_mode 等）

    Returns:
        df_score: 至少包含 symbol/score/reason/extra 的 DataFrame
    """
    # TODO
    raise NotImplementedError


def select_by_sharpe(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    夏普策略：计算 SR，并可结合 PSR 得到去噪后的排序指标。

    TODO【必写（通用框架）】:
      - 计算收益序列（pct_change）并得到 sr_hat
      - 计算 psr（需要 n/skew/kurt 等）
      - 定义排序用的 adjusted 指标（例如 sr_hat 与 psr 的组合）
      - adjusted -> pct_rank_0_100 -> score
      - threshold_mode=="psr" 时调用 apply_threshold_psr
      - 生成 reason/extra（extra 至少包含 sr/psr/adjusted/pct）

    Args:
        df: 预处理后的行情表
        params: 策略参数（window/top_k/threshold_mode/psr_confidence/psr_ref_sharpe 等）

    Returns:
        df_score: 至少包含 symbol/score/reason/extra 的 DataFrame
    """
    # TODO
    raise NotImplementedError


def scan_reversal(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    反转策略：基于均线乖离/超跌程度生成 score。

    TODO【必写（通用框架）】:
      - 计算 bias 或其他超跌指标，并转成“越超跌越高”的 raw
      - raw -> pct_rank_0_100 -> score
      - 可选：只保留 raw>0 的样本（体现“超跌才入选”）
      - threshold_mode=="quantile" 时调用 apply_threshold_quantile
      - reason/extra 需包含 bias/raw/pct 等关键变量

    Args:
        df: 预处理后的行情表
        params: 策略参数

    Returns:
        df_score: 至少包含 symbol/score/reason/extra 的 DataFrame
    """
    # TODO
    raise NotImplementedError


def scan_composite(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    三因子融合：将多个因子的分位得分加权汇总成最终 score。

    TODO【必写（通用框架）】:
      - 分别计算多个 raw（如 mom_raw / sharpe_adj / rev_raw）
      - 分别转成 pct（缺失 -> neutral=50）
      - 读取/归一化权重 composite_weights（sum=1）
      - score = Σ w_i * pct_i
      - reason 需要能解释“各因子贡献”
      - extra 需包含：raw/pct/weights/score 等
      - threshold_mode=="quantile" 时调用 apply_threshold_quantile

    Args:
        df: 预处理后的行情表
        params: 策略参数（window/top_k/composite_weights 等）

    Returns:
        df_score: 至少包含 symbol/score/reason/extra 的 DataFrame
    """
    # TODO
    raise NotImplementedError


def user_defined_strategy(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    学生练习入口：自定义一个指标并输出 df_score。

    TODO【必写（通用框架）】:
      - 必须返回 df_score，且包含 symbol/score/reason/extra
      - score 仍需映射到 0~100 percentile
      - extra 必须包含你的关键中间变量（便于讲解/评分）

    Args:
        df: 预处理后的行情表
        params: 策略参数（window/top_k 等）

    Returns:
        df_score: 至少包含 symbol/score/reason/extra 的 DataFrame
    """
    # TODO
    raise NotImplementedError


def run_strategy(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    策略路由：根据 params["strategy"] 调用对应策略函数。

    TODO【选改（拓展位）】:
      - 做一个清晰的 dispatch（if/elif 或 dict 映射）
      - unknown strategy -> raise ValueError（交给 handler 捕获并 fail）

    Args:
        df: 预处理后的行情表
        params: 策略参数（必须包含 strategy）

    Returns:
        df_score: 对应策略输出的 df_score
    """
    # TODO
    raise NotImplementedError

```
</details>


#### C) `scripts/handler.py`（Skill 系统入口：execute）
**必写（框架通用）**
- `execute(ctx, ...)` 负责：
  1) 调 dataloader 拿到 clean df + universe_size（或 fail）
  2) 组装 params（strategy/window/top_k/min_amount/...）
  3) 调 algo.run_strategy 得到 df_score
  4) 排序截断 top_k，封装为 `EtfCandidate` 列表
  5) 返回 `SkillResult.ok(data={type/items/meta}, insight=...)`
- 空结果必须返回 ok（items=[]），但 insight 要解释“为什么为空”
- 异常要转成 `SkillResult.fail("可解释信息")`（不要 silent）

**选改（拓展位）**
- 保留 `user_defined` 策略练习入口：fail 或 NotImplemented 均可，但信息要明确


<details>
<summary><b>📄Checkpoint-08：scripts/handler.py练习骨架</b></summary>

```py
# scripts/handler.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd

from debate_mas.protocol import EtfCandidate, SkillResult
from debate_mas.skills.base import BaseFinanceSkill, SkillContext

from .dataloader import load_etf_daily
from .algo import run_strategy

Strategy = Literal["momentum", "sharpe", "reversal", "composite", "user_defined"]


class SkillHandler(BaseFinanceSkill):
    """
    Skill 系统入口：负责把 “数据 -> 策略 -> EtfCandidateList” 串起来。

    TODO【必写（通用框架）】
    - execute(ctx, ...) 负责：
      1) 调 dataloader 拿到 clean df + universe_size（或 fail）
      2) 组装 params（strategy/window/top_k/min_amount/...）
      3) 调 algo.run_strategy 得到 df_score
      4) 排序截断 top_k，封装为 EtfCandidate 列表
      5) 返回 SkillResult.ok(data={type/items/meta}, insight=...)

    - 空结果必须返回 ok（items=[]），但 insight 要解释“为什么为空”
    - 异常要转成 SkillResult.fail("可解释信息")（不要 silent）

    TODO【选改（拓展位）】
    - 保留 user_defined 策略练习入口：fail 或 NotImplemented 均可，但信息要明确
    """

    def execute(
        self,
        ctx: SkillContext,
        strategy: Strategy = "momentum",
        window: int = 20,
        top_k: int = 5,
        min_amount: float = 0.0,
        universe: Optional[Union[List[str], str, list[Any]]] = None,
        liquidity_filter: Literal["amount_latest", "amihud"] = "amount_latest",
        amount_scale: float = 1000.0,
        illiq_quantile: float = 0.8,
        threshold_mode: Literal["none", "quantile", "psr"] = "none",
        quantile_q: Optional[float] = None,
        psr_confidence: float = 0.95,
        psr_ref_sharpe: float = 0.0,
        composite_weights: Optional[Union[Dict[str, float], str]] = None,
        **kwargs: Any,
    ) -> SkillResult:
        """
        Args:
            ctx:
                - ctx.dossier: 读取案卷数据（至少包含 etf_daily 表）
                - ctx.ref_date: 防未来切片使用
                - ctx.agent_role: 可写入 meta（可选）
            strategy:
                - 策略名称（需与 algo.run_strategy 支持的策略对齐）
            window/top_k/min_amount/...:
                - 策略/过滤/阈值相关参数（需要打包进 params 传给 algo）
            universe:
                - None / list / dict-like / EtfCandidate-like / str(json/逗号)（由 dataloader 负责解析）
            **kwargs:
                - 预留扩展位（可忽略或写入 meta）

        Returns:
            SkillResult:
              - ok: data.type == "EtfCandidateList"
              - fail: 包含可解释错误信息（缺表/缺列/异常等）
        """
        try:
            # 1) Data: dataloader（契约/清洗/防未来/universe）
            # TODO【必写（通用框架）】:
            # - 调用 load_etf_daily(ctx, universe, apply_date_filter=...)
            # - 处理返回值：
            #   - 若返回 SkillResult（fail），直接 return
            #   - 若返回 (df, universe_size)，继续执行
            loaded = load_etf_daily(
                ctx=ctx,
                universe=universe,
                apply_date_filter=self.apply_date_filter, 
            )
            if isinstance(loaded, SkillResult):
                return loaded
            df, universe_size = loaded  # df: clean DataFrame

            # 2) Params: 组装策略参数（传给 algo）
            # TODO【必写（通用框架）】:
            # - 必须包含 strategy/window/top_k
            # - 把 liquidity/threshold/composite 等参数一并传下去
            # - 把 ref_date/universe_size 等写入 meta（可选，但建议）
            params: Dict[str, Any] = {
                "strategy": strategy,
                "window": int(window),
                "top_k": int(top_k),
                "min_amount": float(min_amount),
                "liquidity_filter": str(liquidity_filter),
                "amount_scale": float(amount_scale),
                "illiq_quantile": float(illiq_quantile),
                "threshold_mode": str(threshold_mode),
                "quantile_q": quantile_q,
                "psr_confidence": float(psr_confidence),
                "psr_ref_sharpe": float(psr_ref_sharpe),
                "composite_weights": composite_weights,
                # meta-only（可选）
                "ref_date": ctx.ref_date,
                "universe_size": universe_size,
                "agent_role": getattr(ctx, "agent_role", None),
            }

            # 3) Algo: 纯计算 df -> df_score
            # TODO【必写（通用框架）】:
            # - 调 run_strategy(df, params)
            # - df_score 至少应包含 symbol/score/reason/extra
            df_score = run_strategy(df, params)

            # 4) Pack: df_score -> EtfCandidateList
            # TODO【必写（通用框架）】:
            # - df_score 可能为空：必须返回 ok(items=[]) + 可解释 insight
            # - 非空：
            #   - 按 score desc 排序
            #   - head(top_k) 截断
            #   - 每行 -> EtfCandidate(symbol/score/reason/source_skill/extra)
            # - threshold_meta：从 df_score.attrs["threshold_meta"] 读取并写入 meta/extra
            candidates: List[EtfCandidate] = []

            # TODO: 处理空 df_score（必须 ok）
            if df_score is None or df_score.empty:
                return SkillResult.ok(
                    data={
                        "type": "EtfCandidateList",
                        "items": [],
                        "meta": {
                            "strategy": strategy,
                            "window": int(window),
                            "top_k": int(top_k),
                            "universe_size": universe_size,
                            "liquidity_filter": str(liquidity_filter),
                            "threshold_mode": str(threshold_mode),
                            "threshold_meta": None,
                            "score_scale": "percentile_0_100",
                            "ref_date": ctx.ref_date,
                        },
                    },
                    insight="TODO: 解释为空的原因（如：样本不足/过滤过严/数据缺失/阈值回退等）。",
                )

            # TODO: 排序 + 截断
            df_top = (
                df_score.sort_values("score", ascending=False)
                .head(int(top_k))
                .copy()
            )

            # TODO: 读取阈值 meta（如 algo 写入 attrs）
            threshold_meta = getattr(df_score, "attrs", {}).get("threshold_meta")

            # TODO: 行 -> EtfCandidate（注意 extra 合并：全局 meta + 单标的 extra）
            for _, row in df_top.iterrows():
                symbol = str(row["symbol"])
                score = float(row["score"])
                reason = str(row.get("reason", ""))

                item_extra = row.get("extra", None)
                if not isinstance(item_extra, dict):
                    item_extra = {}

                merged_extra = {
                    "strategy": strategy,
                    "window": int(window),
                    "liquidity_filter": str(liquidity_filter),
                    "threshold_mode": str(threshold_mode),
                    "threshold_meta": threshold_meta,
                    "universe_size": universe_size,
                    "score_scale": "percentile_0_100",
                    **item_extra,
                }

                candidates.append(
                    EtfCandidate(
                        symbol=symbol,
                        score=score,
                        reason=reason,
                        source_skill="quantitative_sniper",  # 本关固定；迁移时再改 TODO 写成你的 skill 名称
                        extra=merged_extra,
                    )
                )

            # 5) Return: SkillResult.ok
            # TODO【必写（通用框架）】:
            # - data 必须包含 type/items/meta
            # - insight 简洁总结：范围/策略/数量/首选/阈值信息
            data = {
                "type": "EtfCandidateList",
                "items": [c.model_dump() for c in candidates],
                "meta": {
                    "strategy": strategy,
                    "window": int(window),
                    "top_k": int(len(candidates)),
                    "universe_size": universe_size,
                    "liquidity_filter": str(liquidity_filter),
                    "threshold_mode": str(threshold_mode),
                    "threshold_meta": threshold_meta,
                    "score_scale": "percentile_0_100",
                    "ref_date": ctx.ref_date,
                },
            }

            insight = "TODO: 生成一句可解释摘要（包含数量、首选标的、是否触发阈值/回退等）。"
            return SkillResult.ok(data=data, insight=insight)

        except NotImplementedError as e:
            # TODO【选改（拓展位）】: user_defined 未实现时给出明确报错
            return SkillResult.fail(f"TODO: 自定义策略未实现：{e}")

        except Exception as e:
            # TODO【必写（通用框架）】: 捕获异常并转 fail（不要 silent）
            return SkillResult.fail(f"TODO: 执行失败（可解释信息）：{e}")
```
</details>


#### D) `SKILL.md`（技能“对外说明 + 参数 Schema + 调用约束”）
**必写（框架通用）**
- 固定字段：name / role / version / outputs.type
- 说明该技能：
  - 依赖表：`etf_daily`
  - 最小必需列：`code/date/close`
  - 输出类型：`EtfCandidateList`
  - score 统一尺度：0~100 percentile

**选改（拓展位）**
- 写出策略枚举：momentum/sharpe/reversal/composite/user_defined
- 写出阈值模式：none/quantile/psr（psr 仅 sharpe）

> 这份文件会成为“把技能迁移到别的业务”的入口文档，所以字段尽量稳定。

<details>
<summary><b>📄Checkpoint-08：SKILL.md练习骨架</b></summary>

```md
---
name: TODO_skill_name
chinese_name: TODO_中文名
version: TODO_semver
role: TODO_role
group: TODO_group
tags: [TODO, TODO]
description: >
  TODO：一句话说明这是做什么的技能（面向迁移/复用场景，尽量稳定）。

outputs:
  type: EtfCandidateList

data_dependencies:
  - table: etf_daily
    required_columns: [code, date, close]
    optional_columns: [TODO_optional_cols]

schema_notes: >
  TODO：说明返回结构的稳定字段（items/meta/extra 约定），以及可追溯性设计（raw/pct/threshold_meta）。
---

# SKILL（对外说明）

> 本文档是技能迁移/复用的入口说明：描述“依赖什么数据、怎么调用、返回什么、有哪些约束”。

---

## TODO【必写（通用框架）】

### 1) 固定字段（迁移稳定锚点）
- **name**: TODO
- **role**: TODO
- **version**: TODO
- **outputs.type**: `EtfCandidateList`（固定）

### 2) 数据依赖（Data Dependencies）
- 依赖表：`etf_daily`
- 最小必需列：`code` / `date` / `close`
- TODO：说明字段语义与类型约束
  - `code`: TODO（类型/格式）
  - `date`: TODO（类型/时区/解析）
  - `close`: TODO（类型/单位）
- TODO：说明缺列行为：**必须 fail 且可解释**（举例说明）

### 3) 防未来（Ref Date Constraint）
- TODO：说明 `ref_date` 的作用（只允许使用 ref_date 及之前的数据）
- TODO：说明如果 ref_date 缺失/非法的行为（fail 或默认策略）

### 4) 输出类型（Outputs）
- 输出类型：`EtfCandidateList`
- `score` 统一尺度：**0~100 percentile**
- TODO：说明 items 的最小字段集：
  - `symbol`
  - `score`
  - `reason`
  - `source_skill`
  - `extra`
- TODO：说明 meta 的最小字段集（必须稳定）：
  - `strategy`
  - `window`
  - `top_k`
  - `universe_size`
  - `threshold_mode`
  - `threshold_meta`
  - `score_scale`
  - `ref_date`

---

## TODO【选改（拓展位）】

### 5) 策略枚举（Strategy Enum）
- TODO：列出策略枚举并简述用途（保持稳定命名）
  - `momentum`
  - `sharpe`
  - `reversal`
  - `composite`
  - `user_defined`（练习/扩展入口）

### 6) 阈值模式（Threshold Mode）
- TODO：列出阈值模式并说明与策略的兼容关系
  - `none`
  - `quantile`
  - `psr`（TODO：强调仅对 sharpe 生效；其他策略的行为说明）

---

# Action Guide（调用说明）

## 你能做什么
TODO：用 3~5 行说明能力边界（排序/筛选/解释），以及不做什么（不做交易执行/不做预测等）。

## 什么时候调用
- TODO：触发条件（按业务语义映射到 strategy）
- TODO：禁止条件（缺表、缺列、ref_date 缺失等）

## 调用约束（必须遵守）
TODO【必写（通用框架）】
1. 读取数据：从案卷读取 `etf_daily`
2. 防未来：按 `ref_date` 过滤
3. 字段标准化：列名标准化、别名兼容（如有）
4. Universe 过滤：支持多种输入形式（list/dict-like/json/逗号等）
5. 策略计算：df -> df_score
6. 阈值/截断：top_k + threshold_meta（如启用）
7. 封装输出：EtfCandidateList + 可解释 insight

---

# Inputs（参数 Schema）

## Args
- `strategy` (str):
  - TODO：默认值与可选枚举
- `window` (int):
  - TODO：窗口定义与最小要求
- `top_k` (int):
  - TODO：返回数量与边界（<=0 怎么办）
- `universe` (list | str | None):
  - TODO：支持的输入形态与解析规则（由 dataloader 负责）
- `min_amount` (float):
  - TODO：成交额阈值语义与单位
- `liquidity_filter` (str):
  - TODO：可选模式与说明
- `amount_scale` (float):
  - TODO：amount 换算规则（如存在）
- `illiq_quantile` (float):
  - TODO：分位保留规则（如存在）
- `threshold_mode` (str):
  - TODO：none/quantile/psr（psr 的限制写清楚）
- `quantile_q` (float | None):
  - TODO：范围与默认推导规则
- `psr_confidence` (float):
  - TODO：范围与意义（仅 sharpe + psr 时使用）
- `psr_ref_sharpe` (float):
  - TODO：参考值意义（仅 sharpe 时使用）
- `composite_weights` (dict | str | None):
  - TODO：权重字段与归一化规则（仅 composite 时使用）

## Returns
- 成功：`SkillResult.ok`
  - `data.type = "EtfCandidateList"`
  - `data.items = [EtfCandidate, ...]`
  - `data.meta = { ... }`
  - `insight = "TODO: 一句话摘要（可解释）"`
- 空结果：仍然 `SkillResult.ok`
  - `items = []`
  - `insight = "TODO: 解释为空原因（过滤过严/样本不足/阈值回退等）"`
- 失败：`SkillResult.fail("TODO: 可解释失败原因")`

---

# Output Examples（对照示例）

> TODO【选改（拓展位）】给一个最小可用的 EtfCandidateList 示例结构（与 templates/output.json 对齐），用于迁移对照。

```
</details>


#### E) `references/data_contract.md`（数据依赖契约）
**必写（框架通用）**
- 写清楚：表名、必需列、可选列、列类型含义
- 写清楚：缺列时的行为（必须 fail 且可解释）

**选改（拓展位）**
- 兼容字段别名：`data -> date`（你代码中已有兼容逻辑）

<details>
<summary><b>📄Checkpoint-08：references/data_contract.md练习骨架</b></summary>

```md
# 数据契约（Data Contract）

> 本文档描述本 Skill 依赖的数据表与字段契约。
> **缺列/缺表必须 fail 且可解释**（不能 silent）。

---

## TODO【必写（通用框架）】

- 写清楚：**依赖表名**
- 写清楚：**必需列**（required）
- 写清楚：**可选列**（optional）
- 写清楚：每列的 **类型** 与 **含义**
- 写清楚：当数据不满足契约时的行为
  - 缺表：应如何 fail（提示用户去哪找/怎么补）
  - 缺必需列：应如何 fail（明确列名列表）
  - 清洗后为空：应如何 fail（说明可能原因）

## TODO【选改（拓展位）】

- 写清楚：字段别名兼容策略（例如：某字段可接受别名并在代码中标准化）
- 写清楚：单位/缩放（例如 amount 的单位、是否需要换算）

---

## 表：`TODO_TABLE_NAME`

### 必需列（Required）
| 列名 | 类型 | 含义 | 缺失时行为 |
|---|---|---|---|
| TODO | TODO | TODO | fail（解释原因） |
| TODO | TODO | TODO | fail（解释原因） |

### 可选列（Optional）
| 列名 | 类型 | 含义 | 缺失时行为 |
|---|---|---|---|
| TODO | TODO | TODO | 跳过相关功能/回退默认逻辑 |
| TODO | TODO | TODO | 跳过相关功能/回退默认逻辑 |


### 说明
TODO

---
```
</details>


#### F) `references/metrics.md`（公式定义）
**必写（框架通用）**
- 百分位映射：`pct_rank -> score ∈ [0,100]`
- Momentum / Reversal / Sharpe / PSR / Composite 的定义（与 algo 对齐）

**选改（拓展位）**
- 对 PSR 做更强解释：为什么用 `sharpe_adj = SR * PSR` 做排序


<details>
<summary><b>📄Checkpoint-08：references/metrics.md练习骨架</b></summary>

```md
# 指标公式（Metrics / Formulas）

> 本文档必须与 `scripts/algo.py` 的实现对齐：同名指标、同样的 score 映射与阈值规则。

---

## TODO【必写（通用框架）】

### 0) 统一得分尺度（Percentile Score）
- 写清楚：如何从横截面数值映射到 `score ∈ [0,100]`
- 写清楚：缺失值如何处理（neutral=50）
- 写清楚：该映射在所有策略中一致使用

### 1) Momentum（动量）
- 写清楚：动量 raw 的定义（基于价格/收益）
- 写清楚：最终 score 如何由 raw 得到（percentile）
- 写清楚：窗口 window 的含义

### 2) Reversal（反转 / 超跌）
- 写清楚：超跌 raw 的定义（基于均线乖离或其他）
- 写清楚：是否需要 sign 翻转（越超跌越高）
- 写清楚：是否只保留 raw>0（体现“超跌才算信号”）
- 写清楚：score 的构造（percentile）

### 3) Sharpe（夏普）
- 写清楚：收益率序列 r_t 的定义
- 写清楚：SR 的计算方式（是否年化、常数因子等）
- 写清楚：样本长度 n 的定义（与实现一致）

### 4) PSR（概率夏普）
- 写清楚：PSR 的输入变量（sr_hat / sr_ref / n / skew / kurt）
- 写清楚：PSR 的输出范围与含义
- 写清楚：与阈值过滤（threshold_mode="psr"）如何配合

### 5) Composite（三因子融合）
- 写清楚：各因子 raw / pct 的定义
- 写清楚：权重如何归一化（sum=1）
- 写清楚：最终 score 的加权方式

### 6) Threshold（阈值规则）
- Quantile：如何计算 cutoff，如何 fallback（不足 top_k 时）
- PSR：阈值不足 top_k 时如何放宽或回退
- 要求：阈值信息写入 attrs/meta 以便审计

---

## TODO【选改（拓展位）】

- 更强解释：为什么要用 “某种 adjusted 指标” 做排序（例如将 SR 与 PSR 组合）
- 给出一段直觉解释：PSR 在排序里起到的“去噪/可信度”作用
- 给出一个最小示例：raw → pct → score 的数值例子（3~5 个样本即可）

```
</details>


#### G) `references/README.md`（开发者指南）
**必写（框架通用）**
- 解释四层结构：
  - dataloader：契约/清洗
  - algo：纯计算
  - handler：系统入口/封装 SkillResult
  - templates：展示层（report）

**选改（拓展位）**
- 给出“扩展步骤”：如何新增策略（在 algo 增加函数 + handler 路由 + 文档补齐）

<details>
<summary><b>📄Checkpoint-08：references/README.md练习骨架</b></summary>

```md
# Developer Guide（开发者指南）

> 本文档面向开发/教学使用：解释本 Skill 的结构分层、输入输出、以及如何扩展。

---

## TODO【必写（通用框架）】

### 1) 四层结构（必须写清楚各自职责）
- **dataloader（契约/清洗）**
  - TODO：负责什么
  - TODO：输入/输出（含 fail 行为）
- **algo（纯计算）**
  - TODO：负责什么（df -> df_score）
  - TODO：df_score 必含列（symbol/score/reason/extra）
  - TODO：score 统一尺度（0~100 percentile）
- **handler（系统入口）**
  - TODO：负责什么（execute 串联 dataloader + algo + 封装 SkillResult）
  - TODO：空结果 ok 的约定（items=[] + insight 解释）
  - TODO：异常转 fail 的约定
- **templates（展示层）**
  - TODO：output.md / output.json 用于教学展示与对照

### 2) 输入依赖
- TODO：依赖哪些表（表名）
- TODO：关键字段（至少哪些列）
- TODO：防未来规则（ref_date 切片）

### 3) 输出约定
- TODO：SkillResult.data.type 固定为 EtfCandidateList
- TODO：items 每一项包含哪些字段
- TODO：meta 必须包含哪些字段（strategy/window/top_k/threshold_meta 等）

---

## TODO【选改（拓展位）】

### A) 扩展步骤：如何新增一个策略
1. TODO：在 algo.py 增加策略函数（输出 df_score）
2. TODO：在 algo.py 的 run_strategy 增加路由
3. TODO：在 metrics.md 补齐公式定义
4. TODO：在 output.md 里确保展示字段对齐 meta
5. TODO：在 handler.py 里确保 meta/insight/异常处理一致

### B) 练习入口：user_defined
- TODO：说明学习者要改哪里（algo 或 handler）
- TODO：NotImplemented 时提示信息规范

### C) 调试与验收（可选）
- TODO：如何用最小数据跑通
- TODO：常见报错（缺表/缺列/窗口不足/过滤过严）与定位建议

```
</details>


#### H) `templates/output.md` 与 `templates/output.json`（输出模板）
**必写（框架通用）**
- `output.md`：给出 top_k 的展示格式
- `output.json`：给出 EtfCandidateList 的结构示例

**选改（拓展位）**
- 让 md 模板字段与 meta 对齐（strategy/window/top_k/threshold_meta）

<details>
<summary><b>📄Checkpoint-08：templates/output.md练习骨架</b></summary>

```md
<!-- templates/output.md -->
# 量化选股结果

## 参数
- Strategy: **{{ meta.strategy }}**
- RefDate: **{{ meta.ref_date }}**
- Window: **{{ meta.window }}**
- TopK: **{{ meta.top_k }}**
- Universe: **{{ meta.universe_size | default("ALL") }}**
- LiquidityFilter: **{{ meta.liquidity_filter }}**
- ThresholdMode: **{{ meta.threshold_mode }}**
- ThresholdMeta: **{{ meta.threshold_meta }}**
- ScoreScale: **{{ meta.score_scale }}**

> TODO【必写（通用框架）】
> - 展示字段必须与 meta 对齐（strategy/window/top_k/threshold_meta）
> - items 为空时也要给出“为什么为空”的解释段落

---

## Top {{ meta.top_k }} Candidates

| Rank | Symbol | Score (0~100) | Reason |
|---:|---|---:|---|
{% for item in items %}
| {{ loop.index }} | {{ item.symbol }} | {{ item.score }} | {{ item.reason }} |
{% endfor %}

---

## Notes
- TODO：解释 score 的含义（分位得分）
- TODO：提示 extra 中可查看 raw/pct/阈值信息
- TODO【选改（拓展位）】：如需展示 threshold_meta 的关键字段，可在此渲染
```
</details>

<details>
<summary><b>📄Checkpoint-08：templates/output.json练习骨架</b></summary>

```json
// templates/output.json
{
  "status": "ok",
  "insight": "TODO: 一句话总结（范围/策略/数量/首选/阈值信息）",
  "data": {
    "type": "EtfCandidateList",
    "items": [
      {
        "symbol": "TODO_SYMBOL",
        "score": 0.0,
        "reason": "TODO: 可解释理由（含窗口/核心指标/分位等）",
        "source_skill": "TODO: skill_name",
        "extra": {
          "strategy": "TODO",
          "window": 0,
          "liquidity_filter": "TODO",
          "threshold_mode": "TODO",
          "threshold_meta": { "TODO": "..." },
          "universe_size": null,
          "score_scale": "percentile_0_100",

          "TODO_raw_metric": 0.0,
          "TODO_pct_metric": 0.0
        }
      }
    ],
    "meta": {
      "strategy": "TODO",
      "window": 0,
      "top_k": 0,
      "universe_size": null,
      "ref_date": "YYYY-MM-DD",
      "liquidity_filter": "TODO",
      "threshold_mode": "TODO",
      "threshold_meta": { "TODO": "..." },
      "score_scale": "percentile_0_100"
    }
  }
}
```
</details>

### ▶️ 执行命令 Run

本关用 **pytest** 做最小验收。

1) 新建测试文件：`tests/test_quantitative_sniper.py`
   把下面代码完整复制进去：

   <details>
   <summary><b>tests/test_quantitative_sniper.py</b></summary>

   ```py
    from __future__ import annotations

    from dataclasses import dataclass
    from typing import Any, Dict, Optional

    import pandas as pd
    import numpy as np
    import pytest

    from debate_mas.skills.inventory.quantitative_sniper.scripts.handler import SkillHandler

    class FakeDossier:
        def __init__(self, tables: Optional[Dict[str, pd.DataFrame]] = None):
            self._tables = tables or {}

        def get_table(self, name: str) -> Optional[pd.DataFrame]:
            return self._tables.get(name)


    @dataclass
    class FakeSkillContext:
        dossier: FakeDossier
        ref_date: str = "2025-07-10"
        agent_role: str = "hunter"


    def make_etf_daily_df(
        *,
        days: int = 40,
        start: str = "2025-01-01",
        with_amount: bool = True,
        amount_value: float = 1e9,
    ) -> pd.DataFrame:
        dates = pd.date_range(start=start, periods=days, freq="D")
        codes = ["AAA", "BBB", "CCC"]

        rows = []
        for code in codes:
            for i, d in enumerate(dates):
                if code == "AAA":
                    close = 100 + i * 1.0
                elif code == "BBB":
                    close = 100 + np.sin(i / 3.0) * 2.0
                else:
                    close = 100 - i * 1.0

                row = {"code": code, "date": d, "close": close}
                if with_amount:
                    row["amount"] = amount_value
                rows.append(row)

        return pd.DataFrame(rows)


    def _get_attr(res: Any, name: str, default: Any = None) -> Any:
        if isinstance(res, dict):
            return res.get(name, default)
        return getattr(res, name, default)


    def assert_ok_etf_list(res: Any) -> None:
        assert res is not None

        success = _get_attr(res, "success", None)
        status = _get_attr(res, "status", None)
        assert (success is True) or (status == "ok"), f"Expected ok, got success={success}, status={status}"

        data = _get_attr(res, "data", None)
        assert isinstance(data, dict), f"Expected data dict, got {type(data)}"
        assert data.get("type") == "EtfCandidateList"
        assert isinstance(data.get("items"), list)
        assert isinstance(data.get("meta"), dict)

        insight = _get_attr(res, "insight", "")
        assert isinstance(insight, str) and len(insight) > 0


    def assert_fail(res: Any) -> None:
        assert res is not None

        success = _get_attr(res, "success", None)
        status = _get_attr(res, "status", None)
        assert (success is False) or (status == "fail"), f"Expected fail, got success={success}, status={status}"

        msg = _get_attr(res, "error_msg", None) or _get_attr(res, "message", None) or _get_attr(res, "insight", None)
        assert isinstance(msg, str) and len(msg) > 0


    def test_handler_momentum_ok_minimal() -> None:
        df = make_etf_daily_df(days=40, start="2025-01-01", with_amount=True, amount_value=1e9)
        ctx = FakeSkillContext(dossier=FakeDossier({"etf_daily": df}), ref_date="2025-07-10")

        handler = SkillHandler()

        res = handler.execute(
            ctx, 
            strategy="momentum",
            window=20,
            top_k=2,
            min_amount=1000,
            liquidity_filter="amount_latest",
            threshold_mode="none",
        )

        assert_ok_etf_list(res)

        data = _get_attr(res, "data")
        items = data["items"]
        meta = data["meta"]

        assert len(items) == 2

        for it in items:
            assert set(["symbol", "score", "reason", "extra"]).issubset(it.keys())
            assert 0.0 <= float(it["score"]) <= 100.0
            ex = it["extra"]
            assert ex.get("score_scale") == "percentile_0_100"
            assert ex.get("strategy") == "momentum"

        assert meta.get("strategy") == "momentum"
        assert meta.get("window") == 20
        assert meta.get("top_k") == len(items)


    def test_handler_empty_result_returns_ok_with_explain_insight() -> None:
        dates = pd.date_range(start="2025-01-01", periods=40, freq="D")
        rows = []
        for code in ["AAA", "BBB"]:
            for i, d in enumerate(dates):
                rows.append({"code": code, "date": d, "close": 100 + i, "amount": 1e9})
        df = pd.DataFrame(rows)

        ctx = FakeSkillContext(dossier=FakeDossier({"etf_daily": df}), ref_date="2025-07-10")
        handler = SkillHandler()

        res = handler.execute(
            ctx, 
            strategy="reversal",
            window=20,
            top_k=5,
            liquidity_filter="amount_latest",
            threshold_mode="none",
        )

        assert_ok_etf_list(res)
        data = _get_attr(res, "data")
        assert data["items"] == []

        insight = _get_attr(res, "insight")
        assert isinstance(insight, str) and len(insight) > 0


    def test_handler_missing_table_fail_explainable() -> None:
        ctx = FakeSkillContext(dossier=FakeDossier({}), ref_date="2025-07-10")
        handler = SkillHandler()

        res = handler.execute(ctx, strategy="momentum") 
        assert_fail(res)


    def test_handler_universe_filter_no_match_fail_explainable() -> None:
        df = make_etf_daily_df(days=40, start="2025-01-01", with_amount=True, amount_value=1e9)
        ctx = FakeSkillContext(dossier=FakeDossier({"etf_daily": df}), ref_date="2025-07-10")
        handler = SkillHandler()

        res = handler.execute(
            ctx, 
            strategy="momentum",
            window=20,
            top_k=5,
            universe=["NOT_EXIST_1", "NOT_EXIST_2"],
        )
        assert_fail(res)


    def test_handler_sharpe_psr_threshold_meta_pass_through() -> None:
        df = make_etf_daily_df(days=80, start="2025-01-01", with_amount=True, amount_value=1e9)
        ctx = FakeSkillContext(dossier=FakeDossier({"etf_daily": df}), ref_date="2025-07-10")
        handler = SkillHandler()

        res = handler.execute(
            ctx, 
            strategy="sharpe",
            window=20,
            top_k=3,
            liquidity_filter="amount_latest",
            threshold_mode="psr",
            psr_confidence=0.95,
            psr_ref_sharpe=0.0,
        )

        assert_ok_etf_list(res)
        meta = _get_attr(res, "data")["meta"]

        assert "threshold_meta" in meta
        tm = meta["threshold_meta"]
        if tm is not None:
            assert isinstance(tm, dict)
            assert tm.get("mode") == "psr"


    def test_handler_user_defined_must_fail_with_clear_message() -> None:
        df = make_etf_daily_df(days=40, start="2025-01-01", with_amount=True, amount_value=1e9)
        ctx = FakeSkillContext(dossier=FakeDossier({"etf_daily": df}), ref_date="2025-07-10")
        handler = SkillHandler()

        res = handler.execute(ctx, strategy="user_defined") 
        assert_fail(res)

   ```

   </details>

2) 运行测试
   
```bash
uv run pytest -q tests/test_quantitative_sniper.py
```


### ✅ 验收标准 Pass

- 终端输出类似下面信息（数字可能不同，但核心是 **passed**）
  - `6 passed in ...s`
- 过程中没有出现 `ImportError`、`KeyError`、`FrozenInstanceError`、`RuntimeError: 缺少环境变量...`
- 如果失败，你应该能从报错快速定位到三类问题：
  - **SkillResult 结构不匹配**
    - 典型现象：断言在 `status/success` 上失败
    - 处理方式：统一用 `success=True/False` 断言，必要时兼容 `status`
  - **handler 组装没串起来**
    - 典型现象：`data` 为空 / `type` 不对 / `items` 不是 `list`
    - 处理方式：检查 handler 是否走到 `_wrap_result()` 并返回 `SkillResult.ok(...)`
  - **输出不稳定导致 meta/extra 缺字段**
    - 典型现象：`meta.threshold_meta` 不存在、`extra.score_scale` 缺失
    - 处理方式：在封装层保证 meta/extra 的固定字段恒定存在


### 🔁 可迁移点 Transfer

> 本关的 `skills/<skill_name>/` 目录设计目标是：**把“技能交付”拆成稳定骨架（可复用）+ 可替换业务（可重写）。**
>
> 迁移到别的任务时，你通常不是“改一个类”这么简单，而是**保留同一套文件夹结构与契约**，把 Python 逻辑与文档解释整体替换成新业务。

**1. 框架通用 不要动**

<details>
<summary><b>目录骨架（建议保持不变）</b></summary>

- `scripts/handler.py`
  - 对外入口：`execute(ctx, ...)`
  - 负责：取数/组参/调用 algo/封装 `SkillResult`
- `scripts/dataloader.py`
  - 数据契约与清洗：从 dossier 取表、标准化列名、类型转换、过滤、返回 `(df, universe_size)`
- `scripts/algo.py`
  - 纯计算：给定 clean df + params → 返回 df_score（不做 dossier、不做 SkillResult）
- `templates/output.json`
  - 标准输出结构示例（便于对照/教学/验收）
- `templates/output.md`
  - 人类可读展示模板（教学用 top_k 表格）
- `SKILL.md`
  - 对外说明 + 参数 Schema + 调用约束（迁移入口文档）
- `references/README.md`
  - 开发者指南：四层结构与扩展步骤

> `scripts/`可加入其他可调用的工具py文件
>
> `reference/`可加入其他说明md文件

</details>

**2. 业务相关 可替换或重写**

这部分属于“你当前这个 skill 的具体任务”。  
迁移到别的 skill 时，通常需要**整体替换**（不仅仅改一个类名），但保持上面的骨架与契约。


- **数据依赖与数据契约**
  - 当前：`etf_daily` + `code/date/close/(amount)`
  - 迁移方式：换成你的业务表/字段，但仍由 dataloader 统一清洗并返回 clean df
  
  <details>
  <summary><b>示例 TODO：把 ETF 数据入口替换成“贷款审核数据”</b></summary>

  ```py
  # TODO（必写/业务替换）：
  # 1) dataloader.py: 从 dossier.get_table("loan_applications") 取表
  # 2) 标准化字段，例如：id/applicant_income/credit_score/employment_years/...
  # 3) 返回 (clean_df, population_size) 或 SkillResult.fail("可解释原因")
  ```
  </details>


- **因子策略扩展（同业务、同 Skill）**
  - 当前：`momentum / sharpe(psr) / reversal / composite(3-factor)`
  - 迁移方式：保持 `dataloader → algo → handler` 三段结构不变，只在 `algo.py` 新增因子与组合权重口径（score 仍统一 `0~100 percentile`，extra 继续保留 raw 证据）

  
  <details>
  <summary><b>示例 TODO：在 ETF 场景里新增一个因子（乖离率 Bias），并把 Composite 从三因子升级为四因子</b></summary>
  
  ```py
    # TODO（选改/同业务扩展）：
    # 目标：在不改动 handler 的通用流程前提下，
    #      1) 新增一个“乖离率 bias”因子计算
    #      2) 把 composite 从 mom/sharpe/rev 三因子 → mom/sharpe/rev/bias 四因子
    #      3) 同时给 user_defined_strategy 一个练习入口：实现“四因子版本”的返回 df_score

    # === 你需要改的文件 ===
    # - scripts/algo.py:
    #     - 新增/补全 bias 因子计算
    #     - composite_weights 支持 {"mom","sharpe","rev","bias"} 并归一化
    #     - 输出 df_score: 至少包含 symbol/score/reason/extra
    # - scripts/handler.py:
    #     - 只需要把 composite_weights 的 keys 允许 bias（如果你在 handler 做了校验）
    #     - 其他 execute 流程保持不动

    # === 数据依赖与数据契约 ===
    # - 当前：etf_daily + code/date/close/(amount)
    # - 迁移方式：不改表名；只新增 bias 的计算与 composite 的权重口径；score 仍需映射到 0~100 percentile

    # === 乖离率因子定义（建议）===
    # bias = (close_t - MA_window(close)) / MA_window(close)
    # bias_pct = pct_rank_0_100(bias) 或 pct_rank_0_100(-abs(bias))（看你要“追偏离”还是“均值回归”）

    def normalize_weights_4(w: dict | None) -> dict:
        # TODO：把 keys 扩展为 mom/sharpe/rev/bias，并做 sum=1 归一化
        # - 如果传入 None：默认等权 0.25
        # - 如果传入部分 keys：其余用默认，再归一化
        return {"mom": 0.25, "sharpe": 0.25, "rev": 0.25, "bias": 0.25}

    def scan_composite_4factors(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        # TODO：参考你现有 scan_composite，一次遍历算四个 raw：
        # - mom_raw
        # - sharpe_adj（sharpe*psr）
        # - rev_raw
        # - bias_raw（新的）
        #
        # 再算四个 pct：
        # - mom_pct / sharpe_pct / rev_pct / bias_pct
        #
        # 最终 score = w_mom*mom_pct + w_sharpe*sharpe_pct + w_rev*rev_pct + w_bias*bias_pct
        #
        # extra 必须包含：
        # - 四因子的 raw 与 pct
        # - composite_weights（含 bias）
        # - composite_score
        #
        # reason 必须可解释：至少打印四个 pct + 关键 raw（简短）
        pass

    def user_defined_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        TODO（练习入口）：
        - 用“四因子版本”实现一个你自己的打分（例如 bias 用 -abs(bias) 表示“越接近均值越好”）
        - 返回 df_score（symbol/score/reason/extra）
        - score 建议用 0~100 百分位，保持全框架一致
        """
        # TODO：写你的四因子逻辑
        # 1) 计算四个 raw
        # 2) pct_rank_0_100
        # 3) score 合成
        # 4) reason + extra
        # return df_score
        raise NotImplementedError("TODO: implement user_defined_strategy (4-factor).")
  ```
  </details>


- **技能整体重写（跨业务、保留骨架）**
  - 当前：ETF 因子排序类 Skill（`etf_daily` 驱动，输出 `EtfCandidateList`）
  - 迁移方式：保持文件夹骨架（`dataloader.py / algo.py / handler.py / templates/SKILL.md`）与“通用契约”（返回 `CandidateList` + 0~100 percentile + 可解释 reason/extra）不变，替换为你的业务表、字段、策略路由与打分逻辑（技能会重新命名、role/group/tags/description 全部同步改）

  <details>
  <summary><b>示例 TODO：把 algo 从“ETF 因子排序”迁移成“贷款准入评分”（非 ETF 业务）</b></summary>

  ```py
    # TODO（必写/业务替换）：
    # 目标：保留 skill 骨架（dataloader/algo/handler/templates/SKILL.md），
    #      但把业务替换成“贷款准入评分”。

    # === 数据依赖与数据契约 ===
    # - 当前：etf_daily + code/date/close/(amount)
    # - 迁移方式：换成 loan_applications + 你的字段
    #   但仍坚持：dataloader 统一清洗 → algo 只做计算 → handler 只做路由与封装（输出稳定、可测试）

    # === 1) dataloader.py：换数据入口与清洗契约 ===
    def load_loan_applications(ctx: SkillContext, *, apply_date_filter) -> tuple[pd.DataFrame, int] | SkillResult:
        # TODO：
        # 1) 从 dossier.get_table("loan_applications") 取表
        # 2) 标准化字段：id / applicant_income / credit_score / employment_years / debt_to_income / ...
        # 3) 类型转换：数值列 to_numeric，日期列 to_datetime
        # 4) 必需字段检查：缺失则 SkillResult.fail("可解释原因")
        # 5) 返回 (clean_df, population_size)
        pass

    # === 2) algo.py：把 run_strategy 变成评分规则/模型输出 ===
    def run_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        # TODO：根据 params["strategy"] 路由：
        # - "rule_based": 规则打分（硬门槛+加分项）
        # - "risk_score": 风险分（例如 logistic regression 输出概率）
        # - "hybrid": 融合（规则筛 + 模型排序）
        pass

    def rule_based_score(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        # TODO：输出 df_score，至少包含：
        # - symbol: 用 application_id / applicant_id
        # - score: 0~100（可用 pct_rank_0_100 映射）
        # - reason: 简短说明命中的关键规则（例如 “信用分>=700, DTI<0.35”）
        # - extra: 规则命中明细（例如 {"credit_score":720, "dti":0.31, "rule_hits":[...]}）
        pass

    # === 3) handler.py：保留 execute 骨架，只换“表名/参数/封装语义” ===
    class SkillHandler(BaseFinanceSkill):
        def execute(self, ctx: SkillContext, strategy: str = "rule_based", top_k: int = 20, **kwargs) -> SkillResult:
            # TODO（只做三件事）：
            # 1) df, population_size = load_loan_applications(...)
            # 2) params = {...} 组装
            # 3) df_score = run_strategy(df, params) → sort/head(top_k) → SkillResult.ok(...)
            pass
  ```
  </details>


**‼️迁移时的“只改哪里”口诀**
  - **不动**：`skills/<skill_name>/` 目录骨架 + `execute(ctx)->SkillResult` 契约 + `df_score(symbol/score/reason/extra)` 与 `score_scale=percentile_0_100`
  - **可换**：`dataloader` 的表名/字段清洗契约 + `algo` 的指标/策略路由 + `templates/SKILL.md/references` 的业务说明与输出字段
  - **一句话**：**骨架不改、契约不破；换数据、换算法、换文档，就能换业务**

</details>

---


## 关卡-09｜注册与准入 Registry：技能上架

<details>
<summary><b>Checkpoint 09 — 注册与准入 【详情】</b></summary>

> 本关把关卡-08 写好的 Skill 接到系统的 **注册中心 SkillRegistry** 上，完成“插件上架”的最小闭环：  
>
> **扫描 inventory → 解析 SKILL.md → 动态加载 SkillHandler → 注入元信息 → 缓存（_SKILL_CACHE）**
>
> 这一关不追求“更多技能”，只追求 **更稳定、更可测、更可降级**：  
> - 新增一个 skill 文件夹后，系统能自动识别并注册  
> - 注册过程可测试（可用 tmp_path 构造假的 inventory）  
> - 单个 skill 加载失败不影响其他技能注册


### 🎯 目标收获 Outcome
- 理解并实现 **技能上架链路**：inventory → registry → tool 列表
- 学会把“可用技能集合”做成 **可治理的准入系统**（allowlist-by-role）
- 让 SkillRegistry **可测试**：允许在测试中通过 monkeypatch registry.__file__ 指向临时目录，构造假的 inventory。


### 🧱 约束契约 Contract
- 本关只改：
  - `src/debate_mas/skills/base.py`
  - `src/debate_mas/skills/registry.py`
- 不改：
  - `BaseSkill.execute/safe_run/to_langchain_tool` 的**对外协议**
  - `protocol/SkillResult`、`loader/Dossier` 的既有契约
- 注册失败必须 **可降级**：单个 skill 加载失败不影响其他技能被注册


### 🗺️ 任务清单（TODO Map）

> 本关分两段写：先把 `base.py` 写稳，再把 `registry.py` 写通。


#### A) `skills/base.py`

**必看**
- `src/debate_mas/skills/base.py`
  - `SkillContext`：skill 调用时的上下文输入
  - `_auto_args_schema_from_execute`：从 `execute(...)` 签名生成 schema
  - `_ensure_schema_ready`：动态加载 + postponed annotations 的解析兜底
  - `BaseSkill.safe_run / to_langchain_tool`：错误兜底 + Tool 适配入口
- `src/debate_mas/protocol/SkillResult`：技能必须返回的统一结果结构
- `src/debate_mas/loader/dossier.py`：ctx.dossier 数据入口


**必写（框架通用）**
- `safe_run(...)`：确保 **任何异常都被 SkillResult.fail 收敛**
- `to_langchain_tool(...)`：
  - schema 选择优先级：显式 `args_schema` > 自动生成并缓存
  - 统一返回 **JSON 字符串**（避免 dict->str 单引号污染）
  - schema 必须 `model_rebuild`（避免 forward-ref / postponed annotations 崩溃）

**必写（ETF任务相关，迁移可替换）**
- `SkillContext` 字段必须包含：
  - `dossier / agent_role / ref_date`
- `ref_date` 语义：**不能使用 ref_date 当天及之后的数据**

**选改（拓展位）**
- 对 tool 的描述做裁剪（避免 prompt 过长）
- schema 自动生成对类型标注缺失时给出更友好的报错
- `safe_run` 打印更结构化的 debug 信息（但保持返回仍是 SkillResult）


<details>
<summary><b>📄 Checkpoint-09：skills/base.py练习骨架</b></summary>

```py
# src/debate_mas/skills/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, get_type_hints

import inspect
import json
import traceback

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_object_dtype, is_string_dtype

from pydantic import BaseModel, Field, ConfigDict, create_model

from ..protocol import SkillResult
from ..loader.dossier import Dossier


# ==========================================
# 1) Runtime Context
# ==========================================
class SkillContext(BaseModel):
    """
    [上下文环境]
    TODO【必写（ETF任务相关）】:
      - 定义 dossier / agent_role / ref_date 三个字段
      - dossier 类型为 Dossier（只读），agent_role 默认 unknown，ref_date 可选

    Args:
        None

    Returns:
        None
    """
    # TODO
    raise NotImplementedError


# ==========================================================
# 2) Pydantic schema 兜底
# ==========================================================
def _ensure_schema_ready(schema: Optional[type[BaseModel]], *, execute_fn=None) -> None:
    """
    TODO【必写（通用框架）】:
      - schema 为 None 或无 model_rebuild 时直接 return
      - 依次尝试：
        1) schema.model_rebuild(force=True)
        2) schema.model_rebuild(force=True, _types_namespace=execute_fn.__globals__)
        3) schema.model_rebuild(force=True, _types_namespace={})
      - 任何异常都吞掉（兜底函数不应崩）

    Args:
        schema: 需要被 model_rebuild 的 schema
        execute_fn: 可选，execute 函数对象，用于取 globals 作为 types namespace

    Returns:
        None
    """
    # TODO
    raise NotImplementedError


def _auto_args_schema_from_execute(execute_fn, *, model_name: str) -> type[BaseModel]:
    """
    从 execute(self, ctx: SkillContext, ...) 自动生成 args_schema。

    TODO【必写（通用框架）】:
      - inspect.signature(execute_fn) 获取参数
      - 排除 self / ctx
      - 排除 *args / **kwargs
      - 用 get_type_hints 解析类型标注（失败则回退 Any）
      - default 缺省 => 必填(...)
      - create_model(..., __config__=ConfigDict(extra="forbid"), **fields)
      - 对生成的 model 做 model_rebuild（带 execute_fn globals）

    Args:
        execute_fn: 子类实现的 execute 方法
        model_name: schema 名称前缀（用于可读性）

    Returns:
        model: Pydantic BaseModel 子类（args_schema）
    """
    # TODO
    raise NotImplementedError


# ==========================================
# 3) BaseSkill
# ==========================================
class BaseSkill(ABC):
    """
    【通用技能基类】
    """
    name: str = ""
    chinese_name: str = ""
    description: str = ""
    expert_mindset: str = ""

    args_schema: Optional[type[BaseModel]] = None

    @abstractmethod
    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
        """
        TODO【必写（通用框架）】:
          - 子类必须实现
          - 必须返回 SkillResult

        Args:
            ctx: SkillContext
            **kwargs: execute 的业务参数

        Returns:
            SkillResult
        """
        raise NotImplementedError

    def safe_run(self, ctx: SkillContext, **kwargs) -> SkillResult:
        """
        系统调用入口：捕获异常并收敛成 SkillResult。

        TODO【必写（通用框架）】:
          - 调用 self.execute(ctx, **kwargs)
          - 若返回不是 SkillResult：返回 SkillResult.fail（报错信息包含 type）
          - 捕获所有异常：traceback.print_exc + SkillResult.fail

        Args:
            ctx: SkillContext
            **kwargs: execute 参数

        Returns:
            result: SkillResult
        """
        # TODO
        raise NotImplementedError

    def _dump_result(self, result: SkillResult) -> Dict[str, Any]:
        """
        将 SkillResult 转成 dict。

        TODO【必写（通用框架）】:
          - 优先 result.model_dump()
          - 失败则手动拼 success/data/insight/visuals/error_msg

        Args:
            result: SkillResult

        Returns:
            payload: Dict[str, Any]
        """
        # TODO
        raise NotImplementedError

    def to_langchain_tool(self, ctx: SkillContext):
        """
        适配成 LangChain StructuredTool。

        TODO(【必写（通用框架）】):
          - 构造 description：description + expert_mindset（可做长度裁剪）
          - schema 选择：self.args_schema 优先，否则自动生成并缓存到 self._lc_args_schema
          - 调用 _ensure_schema_ready(schema, execute_fn=self.execute)
          - func(**kwargs)：
            - res = self.safe_run(ctx, **kwargs)
            - payload = self._dump_result(res)
            - return json.dumps(payload, ensure_ascii=False)
          - return StructuredTool(name=self.name, description=..., args_schema=schema, func=func)

        TODO(【选改（拓展位）】):
          - 对 schema 生成失败给出更友好的错误提示（但不要改变返回类型约定）

        Args:
            ctx: SkillContext

        Returns:
            tool: StructuredTool
        """
        # TODO
        raise NotImplementedError


# ==========================================
# 4) BaseFinanceSkill
# ==========================================
class BaseFinanceSkill(BaseSkill):
    """
    TODO【必写（ETF任务相关，迁移可替换）】:
      - 提供 apply_date_filter / get_entity_data / rank_by_column
      - 关键语义：ref_date 防未来函数

    Args:
        None

    Returns:
        None
    """
    # TODO（如该文件在前置关卡已实现，可保持不动）
    raise NotImplementedError

```
</details>

#### B) `skills/registry.py`

**必看**
- `src/debate_mas/skills/registry.py`
  - `load_all_skills(...)`：inventory 扫描入口
  - `_parse_skill_md(...)`：解析 `SKILL.md` 的 frontmatter + prompt
  - `_load_package(...)`：动态 import + 实例化 SkillHandler + 注入 meta
  - `_SKILL_CACHE`：注册缓存
- `skills/inventory/<skill_name>/SKILL.md`：name/chinese_name/description + expert prompt
- `skills/inventory/<skill_name>/scripts/handler.py`：必须暴露 SkillHandler


#### **必写（框架通用）**
- `load_all_skills()`：扫描 `inventory/`，逐个包加载；单个失败不影响整体（可降级）
- `_parse_skill_md()`：稳健解析 frontmatter（兼容 BOM / \n / \r\n），返回 `(meta, prompt_text)` 或 `None`
- `_load_package()`：动态 import handler；找不到 `SkillHandler` / 缺文件要给清晰提示并跳过
- `_SKILL_CACHE`：以 `skill_name` 为 key 缓存实例；重复 key 的策略要一致（覆盖/跳过二选一）

#### **必写（ETF任务相关，迁移可替换）**
- 元信息注入：把 `SKILL.md` 的 `name/chinese_name/description` 与 prompt 注入到 instance（供 LLM/tool 描述使用）
- 名称一致性提示：folder 名与 `SKILL.md.name` 不一致时给 warning（不阻断加载）

#### **选改（拓展位）**
- `force_reload=True`：清空缓存并重新扫描（便于开发热加载）
- `get_skill()`：首次访问若缓存为空自动触发加载；找不到 skill 时抛出可读错误
- 可选暴露 `list_skills()`：返回已注册技能名列表 


<details>
<summary><b>📄 src/debate_mas/skills/registry.py练习骨架</b></summary>

```py
# src/debate_mas/skills/registry.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any, Optional

# TODO: import 动态加载所需模块（importlib.util / sys）
# TODO: import 解析 frontmatter 所需模块（re / yaml）

from .base import BaseSkill

_SKILL_CACHE: Dict[str, BaseSkill] = {}


class SkillRegistry:
    @staticmethod
    def load_all_skills(force_reload: bool = False) -> None:
        """
        加载 inventory 下所有技能（扫描入口）。

        TODO【必写（通用框架）】:
          - force_reload=True 时清空 _SKILL_CACHE
          - 扫描 skills/inventory/ 下的子目录（跳过非目录与 __*）
          - 逐个调用 _load_package(...)，单个失败要可降级（不中断整体）

        TODO【选改（拓展位）】:
          - 记录加载数量/失败数量（print 或返回统计二选一）

        Args:
            force_reload: 是否强制清空缓存并重新加载

        Returns:
            None
        """
        # TODO
        raise NotImplementedError

    @staticmethod
    def _parse_skill_md(content: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """
        解析 SKILL.md：frontmatter(YAML) + prompt(正文)。

        TODO【必写（通用框架）】:
          - 兼容 BOM/空行；兼容 \n 与 \r\n
          - 成功返回 (meta_dict, prompt_text)，失败返回 None
          - meta 必须是 dict，否则视为失败

        Args:
            content: SKILL.md 全文字符串

        Returns:
            parsed: Optional[(meta, prompt_text)]
        """
        # TODO
        raise NotImplementedError

    @staticmethod
    def _load_package(skill_dir: Path) -> None:
        """
        加载单个 skill 文件夹（单包加载）。

        TODO【必写（通用框架）】:
          - 读取并解析 SKILL.md（调用 _parse_skill_md）
          - 定位 scripts/handler.py，动态 import 模块
          - 找到 SkillHandler 类并实例化
          - 缺文件/缺类/解析失败：给 warning 并跳过（不中断）

        TODO【必写（ETF任务相关）】:
          - 将 meta.name/chinese_name/description 与 prompt 注入到 instance
          - folder 名与 meta.name 不一致时给提示（不阻断）

        TODO【选改（拓展位）】:
          - 重名 key 的处理策略：覆盖或跳过（保持一致并给提示）

        Args:
            skill_dir: inventory 下的某个技能目录路径

        Returns:
            None
        """
        # TODO
        raise NotImplementedError

    @staticmethod
    def get_skill(name: str) -> BaseSkill:
        """
        按 name 获取已注册的技能实例。

        TODO【必写（通用框架）】:
          - 若 _SKILL_CACHE 为空，先触发 load_all_skills()
          - 找不到 name：抛出 ValueError（信息可读）

        Args:
            name: 技能名（SKILL.md 中的 name）

        Returns:
            skill: BaseSkill 实例
        """
        # TODO
        raise NotImplementedError
```
</details>


### ▶️ 执行命令 Run

本关用 **pytest** 做最小验收。

1) 新建测试文件：`tests/test_skills.py`
   把下面代码完整复制进去：

   <details>
   <summary><b>tests/test_skills.py</b></summary>

   ```py
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

  ```
  </details>

2) 运行测试
   
```bash
uv run pytest -q tests/test_skills.py
```


### ✅ 验收标准 Pass

- 终端输出类似下面信息（数字可能不同，但核心是 **passed**）
  - `5 passed in ...s`
- 过程中没有出现 `ImportError`、`ValidationError`、`AttributeError: ...model_rebuild...`
- 如果失败，你应该能从报错快速定位到三类问题：
  - **SkillContext / Pydantic 校验不匹配**
    - 典型现象：`ValidationError: dossier Input should be ... Dossier`
    - 处理方式：测试里用 `SkillContext.model_construct(...)`（或按真实 Dossier 构造）
  - **safe_run / SkillResult 收敛不稳定**
    - 典型现象：断言 `success is False` 失败，或 `error_msg` 为空
    - 处理方式：确保 `execute` 返回非 `SkillResult` 时转 `SkillResult.fail`，异常被捕获并写入 `error_msg`
  - **registry 动态加载/降级逻辑断裂**
    - 典型现象：`good_skill` 没进 `_SKILL_CACHE` 或坏包导致整体中断
    - 处理方式：检查 `inventory` 扫描、`_parse_skill_md` 解析、`_load_package` 的 try/except


### 🔁 可迁移点 Transfer

> 本关的 `skills/base.py + skills/registry.py` 目标是：**把“技能（Skill）”做成可插件化加载、可结构化返回、可被 LLM 调用的最小协议层**。  
> 
> 迁移到别的任务时，你通常只需要换“技能目录内容（inventory）”，而不必重写注册与调用协议。

**1. 框架通用 不要动**

<details>
<summary><b>skills/base.py 不需要动的地方</b></summary>

- **Skill 的统一入口**
  - `BaseSkill.safe_run(ctx, **kwargs) -> SkillResult`
  - 原则：任何异常都必须收敛为 `SkillResult.fail(...)`，上层永不因单个 skill 崩溃

- **Tool 适配协议**
  - `BaseSkill.to_langchain_tool(ctx)`
  - 原则：统一返回 **JSON 字符串**；schema 选择规则稳定（显式 `args_schema` > 自动生成并缓存）

- **Pydantic schema 兜底**
  - `_ensure_schema_ready(schema, execute_fn=...)`
  - 原则：动态加载/forward-ref 解析失败时不崩（允许吞异常兜底）

</details>

<details>
<summary><b>skills/registry.py 不需要动的地方</b></summary>

- **插件式注册入口**
  - `SkillRegistry.load_all_skills(force_reload=False)`
  - 原则：扫描 `skills/inventory/*`；单包失败可降级，不影响整体

- **单包加载协议**
  - `_parse_skill_md(...)` + `_load_package(...)`
  - 原则：SKILL.md 负责元信息；handler.py 暴露 `SkillHandler`；注册只做“解析 + import + 注入 + 缓存”

- **缓存访问**
  - `_SKILL_CACHE` + `get_skill(name)`
  - 原则：用 name 作为稳定 key；找不到时抛可读错误

</details>

**2. 业务相关 可替换或新增**

这部分是“你要新增/替换 skill 的内容”，而不是改注册层。

- **新增一个新任务的技能**
  - 迁移方式：新增一个目录 `skills/inventory/<skill_name>/`
    - 写 `SKILL.md`（name/chinese_name/description + expert prompt）
    - 写 `scripts/handler.py`（暴露 `SkillHandler.execute(ctx, ...) -> SkillResult`）
  - 原则：只要遵守协议，registry 会自动识别并注册

- **替换/扩展元信息字段（谨慎）**
  - 如果你确实需要更多 meta（例如 `version/tags/owner`），优先只扩展 `SKILL.md` 的 YAML 字段，
    并在 `_load_package` 注入到 instance 的额外属性（不改变 BaseSkill 的核心协议）。


**‼️迁移时的“只改哪里”口诀**
- **不动**：`skills/base.py + skills/registry.py` 的协议骨架（`safe_run(ctx)->SkillResult` / `to_langchain_tool(ctx)` 统一 JSON 输出 / `load_all_skills()` 扫描注册与缓存）
- **只改**：`skills/inventory/<skill_name>/` 里的三件套：`SKILL.md`（元信息+expert prompt）/ `scripts/handler.py`（`SkillHandler.execute(ctx, ...)->SkillResult`）/ `references/*`（业务说明与输出字段）
- **谨慎改**：需要新增 meta 字段时，只扩 `SKILL.md` 的 YAML，并在 registry 注入到实例属性（别动 BaseSkill 契约）
- **一句话**：**协议不改、注册不动；只换 inventory 的内容（文档+handler+参考），就能换业务**

</details>

---


## 关卡-10｜工具 Tools：skill→tools 映射 + 统一调用入口 + 准入守卫

<details>
<summary><b>Checkpoint 10 — 工具 Tools 【详情】</b></summary>

> 本关要解决“**如何安全地让 LLM 调用**”：把 skill 装配成 tools，并在调用入口加上可治理的守卫机制。
>
> 本关闭环是“可调用 + 可拦截”的最小闭环：  
>
> **从 registry 取 skill → 转成 StructuredTool → 按角色 allowlist 装配 → Guard 拦截（白名单/上限/dedup）→ 统一返回稳定 JSON → 写 tool_trace**
>
> 这一关的目标不是更聪明，而是更可控：  
> - 不同角色看到的可用工具集合不同
> - 不合规调用不会抛异常，而是返回可审计的拒绝结果
> - 每次调用都留下 trace，便于复盘与测试


### 🎯 目标收获 Outcome
- 学会把 skill 装成 **StructuredTool** 并按 role 输出 tools 列表
- 实现“统一调用入口”：
  - 调用前：schema 过滤 + policy 注入
  - 调用中：allowlist / max_calls / dedup 拦截
  - 调用后：统一 JSON 输出 + tool_trace 记录（含 denied/produced_n）
- 让准入机制 **可测试**：通过 monkeypatch 配置与 registry，不依赖真实技能数量


### 🧱 约束契约 Contract
- 本关优先只改：
  - `src/debate_mas/core/tools.py`
- 不改：
  - `skills/base.py` 的 `BaseSkill.execute/safe_run/to_langchain_tool` 协议
  - `protocol/SkillResult`、`loader/Dossier` 的既有契约
- 任何被拒绝或异常的 tool 调用都必须 **返回 JSON 字符串**，而不是抛异常


### 🗺️ 任务清单（TODO Map）

**必看**
- `src/debate_mas/core/tools.py`
  - `build_ctx(...)`：给 tool 注入 ctx（dossier/role/ref_date）
  - `build_tools_for_role(...)`：从 allowlist 装配 tools
  - `_wrap_tool_with_guard(...)`：统一调用入口（policy → guard → invoke → trace）
  - `tool_guard_check(...)`：allowlist / max_calls / dedup 三道闸
  - `_append_tool_trace(...)`：把调用结果写进 state.tool_trace

**必写（框架通用）**
- `build_tools_for_role(...)`：只装配 allowlist 内工具；registry 获取 skill 后转 tool
- `tool_guard_check(...)`：三道闸必须可读可诊断（返回 bool, reason）
- `_wrap_tool_with_guard(...)`：
  - 调用前：按 schema keys 过滤参数 + policy 注入
  - 调用失败/拒绝：必须返回 JSON（GUARD_DENY / SkillResult.fail）
  - 调用后：写 trace（ok/denied/elapsed_ms/produced_n）

**必写（ETF任务相关，迁移可替换）**
- `quantitative_sniper` 的 policy 注入（strategy/defaults/profile/enforce/top_k 上限）
- `portfolio_allocator` 从 state 注入 `candidates/risk_reports`（仅当 schema 支持）

**选改（拓展位）**
- `ToolNode` 前对 tool_calls args 做轻量清洗（字符串 list → list）
- `produced_n` 统计：从 `data.items/candidates/results` 推断产出条数（便于 summary）
  
<details>
<summary><b>📄 src/debate_mas/core/tools.py 练习骨架</b></summary>

```py
# src/debate_mas/core/tools.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from contextvars import ContextVar

from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from debate_mas.skills.registry import SkillRegistry
from debate_mas.skills.base import SkillContext
from debate_mas.protocol import SkillResult

from .config import CONFIG
from .state import DebateState, mark_guard_denied


# ============================================================
# SECTION 0) 类型与运行时 state 注入
# ============================================================
_CURRENT_STATE: ContextVar[Optional[DebateState]] = ContextVar("_CURRENT_STATE", default=None)
ToolRunner = Callable[[DebateState], DebateState]


def _get_runtime_state(fallback: DebateState) -> DebateState:
    """
    TODO【选改（拓展位）】:
      - 从 ContextVar 读取“运行时 state”
      - 若取不到则回退到 fallback
      - 目的：避免闭包捕获旧 state

    Args:
        fallback: 兜底用的 state

    Returns:
        st: 当前运行时应使用的 state
    """
    # TODO
    raise NotImplementedError


# ============================================================
# SECTION 1) 指纹 / 稳定序列化
# ============================================================
def _json_dumps_stable(obj: Any) -> str:
    """
    TODO【必写（通用框架）】:
      - 将 obj 稳定序列化为 JSON 字符串
      - 优先：sort_keys + 固定 separators
      - 失败：退化为 json.dumps(str(obj))
      - 目的：dedup 指纹输入必须稳定

    Args:
        obj: 任意可序列化对象

    Returns:
        s: 稳定 JSON 字符串
    """
    # TODO
    raise NotImplementedError


def fingerprint(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """
    TODO【必写（通用框架）】:
      - 组合 tool_name 与稳定序列化后的 tool_args
      - 用 sha1 计算指纹（hex digest）
      - 目的：同回合同 tool 同参数的 dedup

    Args:
        tool_name: 工具名
        tool_args: 工具参数（dict）

    Returns:
        fp: 指纹字符串（sha1 hex）
    """
    # TODO
    raise NotImplementedError


# ============================================================
# SECTION 2) ctx 构建（注入 role/ref_date/dossier）
# ============================================================
def build_ctx(dossier, role: str, ref_date: Optional[str]) -> SkillContext:
    """
    TODO【必写（通用框架）】:
      - 构造 SkillContext
      - 注入 dossier / agent_role / ref_date
      - 不在这里做业务校验（只负责装配）

    Args:
        dossier: 数据案卷（Dossier）
        role: 角色名（hunter/auditor/pm）
        ref_date: 基准日（可选）

    Returns:
        ctx: SkillContext
    """
    # TODO
    raise NotImplementedError


# ============================================================
# SECTION 3) schema 过滤工具
# ============================================================
def _is_empty_value(v: Any) -> bool:
    """
    TODO【必写（通用框架）】:
      - 判空：None 或空字符串（strip 后为空）
      - 目的：policy 注入时区分“未提供” vs “提供了有效值”

    Args:
        v: 任意值

    Returns:
        is_empty: 是否为空值
    """
    # TODO
    raise NotImplementedError


def _schema_keys_from_tool(base_tool: StructuredTool) -> Optional[Set[str]]:
    """
    TODO【必写（通用框架）】:
      - 从 base_tool.args_schema 提取字段名集合
      - 兼容 pydantic v2 的 model_fields 与 v1 的 __fields__
      - 失败返回 None（表示不做 schema 过滤）

    Args:
        base_tool: StructuredTool（含 args_schema）

    Returns:
        keys: schema 字段名集合；未知则 None
    """
    # TODO
    raise NotImplementedError


def _filter_to_schema(args: Dict[str, Any], schema_keys: Optional[Set[str]]) -> Dict[str, Any]:
    """
    TODO【必写（通用框架）】:
      - 若 schema_keys 为 None：原样返回
      - 否则只保留 keys 内的参数
      - 目的：避免“未知字段”触发 args_schema 校验失败

    Args:
        args: 原始参数
        schema_keys: schema 字段集合或 None

    Returns:
        filtered: 过滤后的参数
    """
    # TODO
    raise NotImplementedError


def _fill_missing(args: Dict[str, Any], defaults: Dict[str, Any], schema_keys: Optional[Set[str]]) -> Dict[str, Any]:
    """
    TODO【必写（通用框架）】:
      - 只对缺失/空值的字段补 defaults（不覆盖有效值）
      - 若 schema_keys 给定：只补 schema 支持的字段
      - 目的：给工具提供稳定默认值但不抢 LLM 的显式输入

    Args:
        args: 原参数
        defaults: 默认值集合
        schema_keys: schema 字段集合或 None

    Returns:
        out: 补齐后的参数
    """
    # TODO
    raise NotImplementedError


def _force_override(args: Dict[str, Any], overrides: Dict[str, Any], schema_keys: Optional[Set[str]]) -> Dict[str, Any]:
    """
    TODO【必写（通用框架）】:
      - 强制覆盖：overrides 中字段直接写入
      - 若 schema_keys 给定：只覆盖 schema 支持的字段
      - 目的：少量硬强控参数（上层治理需要）

    Args:
        args: 原参数
        overrides: 强控覆盖项
        schema_keys: schema 字段集合或 None

    Returns:
        out: 覆盖后的参数
    """
    # TODO
    raise NotImplementedError


# ============================================================
# SECTION 4) policy 应用
# ============================================================
def _apply_tool_policy(tool_name: str, tool_args: Dict[str, Any], schema_keys: Optional[Set[str]]) -> Dict[str, Any]:
    """
    TODO【必写（通用框架）】:
      - 对入参做最小“纠错 + 默认值 + 强控”
      - 最后必须 _filter_to_schema(...)，避免未知字段

    TODO【必写（ETF任务相关）】:
      - quantitative_sniper：strategy/defaults/profile/enforce/top_k 上限
      - composite_weights：允许 str/list 兜底成 dict[str,float]（仅 schema 支持时写回）
      - portfolio_allocator：本函数里只做 enforce，candidates/risk_reports 注入在 wrapper 里做

    Args:
        tool_name: 工具名
        tool_args: 原始参数
        schema_keys: schema 字段集合或 None

    Returns:
        args: policy 处理后的参数（已按 schema 过滤）
    """
    # TODO
    raise NotImplementedError


# ============================================================
# SECTION 5) Guard：白名单/上限/去重
# ============================================================
def tool_guard_check(
    role: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    state: DebateState,
) -> Tuple[bool, str]:
    """
    TODO【必写（通用框架）】:
      - allowlist：tool 不在角色白名单直接拒绝
      - max_calls：超过本轮上限拒绝（读取 state 里的计数）
      - dedup：同回合同 tool+args 指纹命中则拒绝（可开关）
      - 返回 (allowed, reason)，reason 必须可读可诊断

    Args:
        role: 角色名
        tool_name: 工具名
        tool_args: 工具参数
        state: 运行时 state（含本轮计数/指纹集）

    Returns:
        allowed: 是否允许调用
        reason: 允许则 "ok"，拒绝则给出原因
    """
    # TODO
    raise NotImplementedError


def _guard_deny_payload(reason: str) -> str:
    """
    TODO【必写（通用框架）】:
      - 将拒绝原因包装成 SkillResult.fail
      - 统一返回 JSON 字符串（ensure_ascii=False）
      - error_msg 需带 [GUARD_DENY] 前缀便于定位

    Args:
        reason: 拒绝原因

    Returns:
        payload_json: JSON 字符串
    """
    # TODO
    raise NotImplementedError


# ============================================================
# SECTION 6) tool 输出解析：用于 trace 统计 produced_n
# ============================================================
def _try_parse_tool_json(text: str) -> Optional[Dict[str, Any]]:
    """
    TODO【必写（通用框架）】:
      - 尝试把 text 解析成 dict
      - 失败返回 None（不抛异常）
      - 目的：trace/produced_n 统计需要结构化结果

    Args:
        text: 工具输出文本（期望为 JSON）

    Returns:
        obj: dict 或 None
    """
    # TODO
    raise NotImplementedError


def _count_produced(obj: Optional[Dict[str, Any]]) -> int:
    """
    TODO【选改（拓展位）】:
      - 从 SkillResult 结构推断产出条数
      - 优先 data.items / data.candidates / data.results 的 list 长度
      - 兜底：顶层 items 若存在也可统计

    Args:
        obj: 解析后的工具输出 dict

    Returns:
        n: 产出条数（无法统计则 0）
    """
    # TODO
    raise NotImplementedError


def _append_tool_trace(
    state: DebateState,
    *,
    role: str,
    tool: str,
    args: Dict[str, Any],
    ok: bool,
    insight: str = "",
    error_msg: Optional[str] = None,
    visuals: Optional[List] = None,
    elapsed_ms: Optional[int] = None,
    denied: bool = False,
    produced_n: Optional[int] = None,
) -> None:
    """
    TODO【必写（通用框架）】:
      - 确保 state.tool_trace 是 list
      - 追加一条 trace：role/tool/args/ok/denied/elapsed_ms/produced_n
      - round_idx/ts 可写入（便于复盘），字段缺失也不应报错

    Args:
        state: 运行时 state（会被原地更新）
        role: 角色名
        tool: 工具名
        args: 调用参数
        ok: 调用是否成功（按 success 推断）
        insight: 结果摘要（可选）
        error_msg: 错误信息（可选）
        visuals: 可视化信息（可选）
        elapsed_ms: 耗时毫秒（可选）
        denied: 是否被 guard 拒绝
        produced_n: 产出条数（可选）

    Returns:
        None
    """
    # TODO
    raise NotImplementedError


# ============================================================
# SECTION 7) Tool 构建：给 LangChain 的 StructuredTool
# ============================================================
def build_tools_for_role(
    role: str,
    ctx: SkillContext,
    state: DebateState,
) -> List[StructuredTool]:
    """
    TODO【必写（通用框架）】:
      - 触发 SkillRegistry.load_all_skills()
      - 读取 CONFIG.ROLE_TOOL_ALLOWLIST[role]
      - 只为 allowlist 中每个 tool_name：
        - skill = SkillRegistry.get_skill(tool_name)
        - base_tool = skill.to_langchain_tool(ctx)
        - tools.append(_wrap_tool_with_guard(...))
      - 若 allowlist 为空：返回空 list（不报错）

    Args:
        role: 角色名
        ctx: SkillContext（已注入 dossier/role/ref_date）
        state: 运行时 state（用于 guard/trace）

    Returns:
        tools: 可调用的 StructuredTool 列表
    """
    # TODO
    raise NotImplementedError


def _wrap_tool_with_guard(
    *,
    role: str,
    tool_name: str,
    base_tool: StructuredTool,
    state: DebateState,
) -> StructuredTool:
    """
    TODO【必写（通用框架）】:
      - 统一调用入口：policy → guard → invoke → trace
      - 调用前：
        - schema_keys = _schema_keys_from_tool(base_tool)
        - tool_args = _apply_tool_policy(tool_name, kwargs, schema_keys)
      - 调用失败/拒绝：
        - 必须 return JSON（_guard_deny_payload 或 SkillResult.fail）
      - 调用后：
        - out_json 必须是 str（若是 dict 要 json.dumps）
        - 解析 out_obj，推断 ok 与 produced_n，并写 trace

    TODO【必写（ETF任务相关）】:
      - portfolio_allocator：从 state 注入 candidates/risk_reports（仅当 schema 支持）
      - quantitative_sniper：可记录 strategy 使用情况（写入 state 的统计字段，非必须）

    Args:
        role: 角色名
        tool_name: 工具名
        base_tool: skill.to_langchain_tool(ctx) 得到的 tool
        state: 运行时 state（会被原地更新）

    Returns:
        wrapped: 带 guard 的 StructuredTool
    """
    # TODO
    raise NotImplementedError


# ============================================================
# SECTION 8) ToolNode 构建（给 graph.py 用）
# ============================================================
def build_tool_node_for_role(
    role: str,
    tools: List[StructuredTool],
    state: DebateState,
) -> Optional[ToolRunner]:
    """
    TODO【选改（拓展位）】:
      - 若 tools 为空：返回 None
      - 用 ToolNode(tools=tools) 构建原始节点
      - 运行时把 state_in 注入 ContextVar，保证 wrapper 读到最新 state
      - 可对 last_msg.tool_calls 的 args 做轻量清洗（字符串 list → list）

    Args:
        role: 角色名
        tools: 该角色可用 tools
        state: fallback state（闭包兜底）

    Returns:
        node: ToolRunner(state)->state；无 tools 时返回 None
    """
    # TODO
    raise NotImplementedError


# ============================================================
# SECTION 9) 便利函数：按 role 直接装配
# ============================================================
def build_role_tools_and_node(
    *,
    role: str,
    dossier,
    ref_date: Optional[str],
    state: DebateState,
) -> Tuple[List[StructuredTool], ToolRunner, SkillContext]:
    """
    TODO【必写（通用框架）】:
      - ctx = build_ctx(...)
      - tools = build_tools_for_role(...)
      - node = build_tool_node_for_role(...)
      - 若 node 为 None：抛 ValueError（信息可读）

    Args:
        role: 角色名
        dossier: 数据案卷（Dossier）
        ref_date: 基准日（可选）
        state: 运行时 state

    Returns:
        tools: 可调用 tools 列表
        node: ToolRunner（可被 graph 调用）
        ctx: SkillContext
    """
    # TODO
    raise NotImplementedError

```
</details>


### ▶️ 执行命令 Run

本关用 **pytest** 做最小验收。

1) 新建测试文件：`tests/test_tools.py`
   把下面代码完整复制进去：

   <details>
   <summary><b>tests/test_tools.py</b></summary>

   ```py
    import json
    from typing import Any, Dict, List, Optional

    import pytest
    from pydantic import BaseModel, Field

    from debate_mas.protocol import SkillResult
    from debate_mas.skills.base import SkillContext
    from debate_mas.core import tools as t


    class _FakeConfig:
        ROLE_TOOL_ALLOWLIST = {
            "hunter": ["quantitative_sniper"],
            "pm": ["portfolio_allocator"],
            "auditor": [],
        }
        ROLE_TOOL_MAX_CALLS = {"hunter": 1, "pm": 2, "auditor": 0}
        FORBID_SAME_TOOL_SAME_ARGS_IN_SAME_ROUND = True

        HUNTER_PIPELINE_SNIPER_STRATEGY = "composite"
        HUNTER_RERANK_OUTPUT_TOPN = 3

        SNIPER_DEFAULTS = {"top_k": 5, "min_amount": 0}
        SNIPER_PROFILES = {
            "composite": {"top_k": 9},  
            "momentum": {"top_k": 7},
        }
        SNIPER_ENFORCE = {}  
        SNIPER_LIMITS = {"max_top_k": 4}  
        PM_PORTFOLIO_ALLOCATOR_ENFORCE = {}


    class _DummyDossier:
        """最小 dossier 占位。测试不依赖其内容。"""


    def _ctx_stub(role: str) -> SkillContext:
        return SkillContext.model_construct(dossier=_DummyDossier(), agent_role=role, ref_date="2025-01-01")


    class _SniperArgs(BaseModel):
        strategy: Optional[str] = None
        top_k: int = 10
        composite_weights: Optional[dict] = None


    class _AllocatorArgs(BaseModel):
        candidates: List[dict] = Field(default_factory=list)
        risk_reports: List[dict] = Field(default_factory=list)


    class _FakeStructuredTool:
        """
        轻量替身：只提供 invoke() / args_schema / name / description
        （避免 LangChain 版本差异导致测试脆弱）
        """
        def __init__(self, *, name: str, args_schema, handler):
            self.name = name
            self.description = ""
            self.args_schema = args_schema
            self._handler = handler

        def invoke(self, tool_args: Dict[str, Any]):
            return self._handler(tool_args)


    class _FakeSkill:
        def __init__(self, name: str, base_tool: _FakeStructuredTool):
            self.name = name
            self._tool = base_tool

        def to_langchain_tool(self, ctx: SkillContext):
            return self._tool

    def test_build_ctx_injects_fields():
        ctx = t.build_ctx(_DummyDossier(), role="hunter", ref_date="2025-01-01")
        assert ctx.agent_role == "hunter"
        assert ctx.ref_date == "2025-01-01"
        assert ctx.dossier is not None


    def test_build_tools_for_role_uses_allowlist_only(monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)

        called = {"get": [], "load": 0}

        def fake_load_all_skills():
            called["load"] += 1

        def fake_get_skill(name: str):
            called["get"].append(name)
            tool = _FakeStructuredTool(
                name=name,
                args_schema=_SniperArgs,
                handler=lambda _: json.dumps(SkillResult.ok(data={"ping": 1}).model_dump(), ensure_ascii=False),
            )
            return _FakeSkill(name, tool)

        monkeypatch.setattr(t.SkillRegistry, "load_all_skills", staticmethod(fake_load_all_skills), raising=True)
        monkeypatch.setattr(t.SkillRegistry, "get_skill", staticmethod(fake_get_skill), raising=True)

        st = {}
        ctx = _ctx_stub("hunter")
        tools = t.build_tools_for_role("hunter", ctx, st)

        assert called["load"] == 1
        assert called["get"] == ["quantitative_sniper"]
        assert len(tools) == 1
        assert tools[0].name == "quantitative_sniper"


    def test_guard_denies_not_in_allowlist(monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)
        st = {"_round_tool_calls": {"hunter": 0}, "_round_fingerprints": set()}
        ok, reason = t.tool_guard_check("hunter", "not_allowed", {}, st)
        assert ok is False
        assert "白名单" in reason


    def test_guard_denies_over_max_calls(monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)
        st = {"_round_tool_calls": {"hunter": 1}, "_round_fingerprints": set()}
        ok, reason = t.tool_guard_check("hunter", "quantitative_sniper", {}, st)
        assert ok is False
        assert "上限" in reason


    def test_guard_denies_dedup_same_tool_same_args(monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)
        args = {"x": 1}
        fp = t.fingerprint("quantitative_sniper", args)

        st = {"_round_tool_calls": {"hunter": 0}, "_round_fingerprints": {fp}}
        ok, reason = t.tool_guard_check("hunter", "quantitative_sniper", args, st)
        assert ok is False
        assert "dedup" in reason.lower() or "重复" in reason


    def test_wrap_tool_with_guard_denied_returns_json_and_writes_trace(monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)

        base_tool = _FakeStructuredTool(
            name="quantitative_sniper",
            args_schema=_SniperArgs,
            handler=lambda _: (_ for _ in ()).throw(RuntimeError("should not call")),
        )

        st = {"round_idx": 0, "_round_tool_calls": {"hunter": 0}, "_round_fingerprints": set()}
        wrapped = t._wrap_tool_with_guard(role="hunter", tool_name="not_allowed", base_tool=base_tool, state=st)

        out = wrapped.invoke({"top_k": 10})
        obj = json.loads(out)
        assert obj["success"] is False
        assert "[GUARD_DENY]" in (obj.get("error_msg") or "")
        assert st.get("tool_trace") and st["tool_trace"][-1]["denied"] is True


    def test_wrap_tool_with_guard_ok_returns_json_and_trace(monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)

        def handler(args: Dict[str, Any]) -> str:
            payload = SkillResult.ok(data={"items": [1, 2]}, insight="ok").model_dump()
            payload["data"]["seen_top_k"] = args.get("top_k")
            payload["data"]["seen_strategy"] = args.get("strategy")
            return json.dumps(payload, ensure_ascii=False)

        base_tool = _FakeStructuredTool(
            name="quantitative_sniper",
            args_schema=_SniperArgs,
            handler=handler,
        )

        st = {"round_idx": 0, "_round_tool_calls": {"hunter": 0}, "_round_fingerprints": set()}
        wrapped = t._wrap_tool_with_guard(role="hunter", tool_name="quantitative_sniper", base_tool=base_tool, state=st)

        out = wrapped.invoke({})
        obj = json.loads(out)

        assert obj["success"] is True
        assert st["tool_trace"][-1]["produced_n"] == 2
        assert st["tool_trace"][-1]["ok"] is True

        assert obj["data"]["seen_strategy"] == "composite"
        assert int(obj["data"]["seen_top_k"]) <= 4


    def test_portfolio_allocator_injects_state_fields_when_schema_supports(monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)

        def handler(args: Dict[str, Any]) -> str:
            payload = SkillResult.ok(data={"candidates": args.get("candidates"), "risk_reports": args.get("risk_reports")}).model_dump()
            return json.dumps(payload, ensure_ascii=False)

        base_tool = _FakeStructuredTool(
            name="portfolio_allocator",
            args_schema=_AllocatorArgs,
            handler=handler,
        )

        st = {
            "round_idx": 0,
            "_round_tool_calls": {"pm": 0},
            "_round_fingerprints": set(),
            "candidates_cur": [{"code": "510300"}],
            "risk_reports": [{"code": "510300", "risk": "low"}],
        }

        wrapped = t._wrap_tool_with_guard(role="pm", tool_name="portfolio_allocator", base_tool=base_tool, state=st)
        out = wrapped.invoke({}) 
        obj = json.loads(out)

        assert obj["success"] is True
        assert obj["data"]["candidates"] == [{"code": "510300"}]
        assert obj["data"]["risk_reports"] == [{"code": "510300", "risk": "low"}]


    def test_runtime_state_contextvar_affects_wrapper_trace(monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)

        def handler(_args: Dict[str, Any]) -> str:
            payload = SkillResult.ok(data={"items": [1]}).model_dump()
            return json.dumps(payload, ensure_ascii=False)

        base_tool = _FakeStructuredTool(
            name="quantitative_sniper",
            args_schema=_SniperArgs,
            handler=handler,
        )

        fallback_state = {"round_idx": 0, "_round_tool_calls": {"hunter": 0}, "_round_fingerprints": set()}
        wrapped = t._wrap_tool_with_guard(
            role="hunter",
            tool_name="quantitative_sniper",
            base_tool=base_tool,
            state=fallback_state,
        )

        state_in = {"round_idx": 7, "_round_tool_calls": {"hunter": 0}, "_round_fingerprints": set()}
        token = t._CURRENT_STATE.set(state_in)
        try:
            out = wrapped.invoke({})
            assert json.loads(out)["success"] is True
        finally:
            t._CURRENT_STATE.reset(token)

        assert state_in.get("tool_trace")
        assert state_in["tool_trace"][-1]["round_idx"] == 7

   ```
   </details>


2) 运行测试
   
```bash
uv run pytest -q tests/test_tools.py
```


### ✅ 验收标准 Pass

- 终端输出类似下面信息（数字可能不同，但核心是 **passed**）
  - `9 passed in ...s`
- 过程中没有出现 `ImportError`、`pydantic ValidationError`、`AttributeError: ...args_schema...`、`KeyError: tool_trace`
- 如果失败，你应该能从报错快速定位到三类问题：
  - **allowlist/guard 没接对**
    - 典型现象：`not_allowed` 仍能调用成功，或 deny 的 reason 不含“白名单/上限/dedup”
    - 处理方式：检查 `tool_guard_check(...)` 三道闸与 CONFIG 字段名是否与测试的 `_FakeConfig` 一致
  - **wrapper 没统一 JSON + trace**
    - 典型现象：`json.loads(out)` 失败 / `tool_trace` 为空 / 缺字段（如 `denied/produced_n/ok`）
    - 处理方式：确保 deny/exception 也返回 JSON，并在所有分支写 `_append_tool_trace`
  - **runtime state 注入失效**
    - 典型现象：trace 的 `round_idx` 不是 7（仍是闭包旧值）
    - 处理方式：检查 `ContextVar set/reset` 与 `_get_runtime_state` 是否真正使用注入的 是否真的优先使用注入的 `state_in`


### 🔁 可迁移点 Transfer

> 本关的 `core/tools.py` 设计目标是：**把“技能 Skill”封装成可控的 Tools，并提供统一的调用入口（policy → guard → invoke → trace）。**
>
> 迁移到别的任务时，你通常只需要换“allowlist / policy 规则 / 注入的业务状态字段”，而不必重写工具装配与守卫框架。

**1. 框架通用 不要动**

<details>
<summary><b>tools.py 不需要动的地方</b></summary>

- **ctx 注入（SkillContext）**
  - `build_ctx(dossier, role, ref_date) -> SkillContext`
  - 原则：只负责装配上下文（dossier/role/ref_date），不在此处写业务校验

- **skill → tools 装配入口**
  - `build_tools_for_role(role, ctx, state) -> List[StructuredTool]`
  - 原则：由 allowlist 决定“装哪些”，由 registry 决定“有哪些”，工具层只做封装

- **统一调用入口（wrapper）**
  - `_wrap_tool_with_guard(role, tool_name, base_tool, state) -> StructuredTool`
  - 原则：所有分支都遵守同一链路：policy → guard → invoke → trace，并统一返回 JSON 字符串

- **三道闸治理**
  - `tool_guard_check(role, tool_name, tool_args, state) -> (bool, reason)`
  - 原则：可读、可诊断；拒绝必须解释“为何拒绝”（白名单/上限/dedup）

- **trace 稳定落盘**
  - `_append_tool_trace(state, ...)`
  - 原则：trace 永远是 list；字段缺失不崩；确保 `ok/denied/elapsed_ms/produced_n/round_idx` 可用于复盘与统计

</details>

**2. 业务相关 可替换或重写**

这部分属于“你们当前是 ETF 的示例业务”。  

迁移到别的任务时可以**整体替换**，但建议保留同一条原则：**少而硬的 policy（Strong Policy）+ 明确可诊断的 guard（Diagnosable Guard）**，让工具层长期稳定、可测试。

- **准入策略：allowlist / 配额 / dedup**
  - ETF 当前用 `CONFIG.ROLE_TOOL_ALLOWLIST / ROLE_TOOL_MAX_CALLS / FORBID_SAME_TOOL_SAME_ARGS_IN_SAME_ROUND`
  - 迁移方式：改 CONFIG 即可，不建议在函数里写死角色/工具名

  <details>
  <summary><b>示例 TODO：把“角色准入”换成你的业务准入规则</b></summary>

  ```py
  # TODO：
  # 1) 配置不同 role 的 allowlist（按岗位/权限分层）
  # 2) 给每个 role 配 max_calls（控制成本/频率）
  # 3) 是否开启 dedup（避免同轮重复调用）
  #
  # 规则尽量放 CONFIG；tool_guard_check 只读配置并给出可诊断 reason
  pass
  ```
  </details>

- **policy 注入规则**
  - ETF 当前在 `_apply_tool_policy(...)` 里做“纠错 + 默认值 + 强控”
  - 迁移方式：保留函数框架，按你的工具名增加分支（每个分支都应最终 `_filter_to_schema`）

  <details>
  <summary><b>示例 TODO：为你的关键工具写 policy 注入</b></summary>

  ```py
  # TODO：
  # 1) 对输入做轻量纠错（str/list/dict 兜底）
  # 2) defaults：只补缺失（不覆盖 LLM 显式值）
  # 3) enforce：少量硬强控（治理需要）
  # 4) 最后按 schema 过滤，避免未知字段导致校验失败
  pass

  ```
  </details>

- **状态注入字段（从 state → tool_args）**
  - ETF 当前在 wrapper 内给 portfolio_allocator 注入 `candidates/risk_reports`
  - 迁移方式：把“需要吃硬状态的字段”集中在 wrapper 的少数分支里，且先检查 schema 是否支持

  <details>
  <summary><b>示例 TODO：让某些工具吃“运行时 state”字段</b></summary>

  ```py
  # TODO：
  # 1) 仅对少数确实需要的 tool 注入 state 字段
  # 2) 注入前先判断 schema_keys 是否包含该字段
  # 3) 注入值尽量来自 state 的稳定键（cur/history 分层更清晰）
  pass

  ```
  </details>

- **产出统计（produced_n）规则**
  - ETF 当前从 `data.items/candidates/results` 推断条数，用于 summary 的真实统计
  - 迁移方式：按你的 SkillResult.data 结构替换 key 列表即可（保持“推断失败=0”的兜底）

  <details>
  <summary><b>示例 TODO：把 produced_n 统计改成你的业务输出结构</b></summary>

  ```py
  # TODO：
  # 1) 约定你的结果 list 字段名（如 data.issues / data.recommendations）
  # 2) 统计优先级：最常用字段优先
  # 3) 推断失败返回 0（不报错）
  pass

  ```
  </details>

**‼️迁移时的“只改哪里”口诀**
- **不动**：`build_ctx / build_tools_for_role / _wrap_tool_with_guard / tool_guard_check / _append_tool_trace` 这套 **装配+统一入口+三道闸+trace** 框架（policy→guard→invoke→trace 链路不破、返回统一 JSON 不破）
- **可换**：`CONFIG` 里的 **allowlist / max_calls / dedup 开关** + `_apply_tool_policy` 的 **工具分支规则** + wrapper 里少数需要的 **state→tool_args 注入字段**（先验 schema 再注入）
- **一句话**：**框架不改、链路不破；换准入、换 policy、换注入字段，就能换业务**

</details>

---


## ✅ 练习完成清单（10 关回顾）

你已经把 “Debate MAS” 从 **能跑** 走到了 **可控、可审计、可扩展**：

- **关卡 01**：跑通 Demo、会读产物（memo / csv / log / transcript）
- **关卡 02**：共享账本 State（cur + history + stop_reason）
- **关卡 03**：结构化协议 Protocol（schema + renderer 三件套）
- **关卡 04**：证据案卷 Loader（folder → dossier）
- **关卡 05**：提示词工厂 Personas（白名单 + 输出契约）
- **关卡 06**：流程编排 Graph（跳转 + 停机）
- **关卡 07**：引擎串联 Engine（最小循环跑完）
- **关卡 08**：写一个 Skill（可调用、可结构化返回）
- **关卡 09**：注册与准入（registry + allowlist）
- **关卡 10**：工具封装与守卫（skill → tool + 统一入口 + 拦截）

> 你不需要“记住所有代码细节”。  
> 真正重要的是：你已经掌握了**一套能反复复用的骨架**——从证据输入、到多角色协作、到结构化交付。

---

## 🧩 运行整套项目之前的小提醒

如果你准备完整跑通项目（而不只是做关卡单测），请确保你在项目里准备好必要的配置文件：

- 需要你**根据 README 的参数表**创建并填写 `config.py`（路径、阈值、开关等）。  
  这里直接跳转到参数表：**[README.md › 6.2 核心参数速查表](README.md#62-核心参数速查表)**

> 建议做法：先按参数表写一个“最小可运行”的 `config.py`，跑通后再逐项精调。  
> 这能显著减少你在“环境/路径/默认值”上的时间损耗。

---

## 📚 附录：参考资源（官方文档）

以下资源能帮你把练习里的概念“对照到真实世界的工程生态”，都给你放了官方入口：

- **LangChain**（消息/工具/模型接口）：[LangChain Documentation](https://python.langchain.com/docs/)
- **LangGraph**（状态图编排、循环与分支）：[LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- **Pydantic**（结构化协议、数据校验）：[Pydantic Documentation](https://docs.pydantic.dev/)
- **pytest**（单测与验收）：[pytest Documentation](https://docs.pytest.org/)
- **uv**（更快的 Python 包与环境管理）：[uv Documentation](https://docs.astral.sh/uv/)
- **pandas**（表格数据处理）：[pandas Documentation](https://pandas.pydata.org/docs/)
- **pypdf**（PDF 读取，可选依赖）：[pypdf Documentation](https://pypdf.readthedocs.io/)
- **python-docx**（Word 读取，可选依赖）：[python-docx Documentation](https://python-docx.readthedocs.io/)

---

## 🎉 你已经具备的能力

**恭喜你完成了 Debate MAS 三段式实战练习！**  
到这里，你已经能够：

- [ ] 用最少的改动跑通一个可审计的多角色决策系统
- [ ] 用 State 让角色“共享事实”，而不是共享口头结论
- [ ] 用 Schema + Renderer 把产物变成可交付的三件套（json / md / csv）
- [ ] 用 Loader 把杂乱材料统一进 Dossier，给系统一个稳定的证据入口
- [ ] 用 Personas + Allowlist 把工具权责与输出契约“写死到提示词里”
- [ ] 用 Graph 把流程变成可控的状态机，知道为什么停、在哪停、怎么继续
- [ ] 写出能被系统调用的 Skill，并把它纳入治理（注册/准入/守卫）

> 接下来最值得做的一件事：  
> **把你最熟悉的业务材料丢进 dossier**，然后用同样的流程让系统产出第一版“可复盘”的决策 memo。  
> 你会非常直观地感受到：框架一旦稳定，剩下就是替换材料与指标，而不是重写系统。

---

### 🫶 小小的鼓励

你做完的不是“十个练习题”，而是完成了一个很难得的工程能力闭环：  
**把 Agent 从“能说”变成“能跑、能控、能交付”。**

继续往前走——你会越来越像一个真正的 AI Agent Builder。 🔧✨