# skills/base.py
"""
Layer 3 - 技能基类定义 (The Skill Constitution)
定义了所有金融技能必须遵守的输入输出协议与开发标准。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, get_type_hints

import inspect
import json
import traceback

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_object_dtype, is_string_dtype

from pydantic import BaseModel, Field, ConfigDict, create_model

# 引入通用协议与案卷
from ..protocol import SkillResult
from ..loader.dossier import Dossier

# ==========================================
# 1. 定义运行上下文 (The Runtime Context)
# ==========================================
class SkillContext(BaseModel):
    """
    [上下文环境]
    每次技能被调用时系统会传入这个对象：
    - dossier：全量案卷数据（只读）
    - agent_role：当前调用技能的角色（hunter/auditor/pm等）
    - ref_date：决策基准日（禁止使用 ref_date 当天及之后的数据）
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    dossier: Dossier = Field(..., description="全量案卷数据(只读)")
    agent_role: str = Field(default="unknown", description="当前调用技能的角色")
    ref_date: Optional[str] = Field(default=None, description="决策基准日(T日)，只能使用T-1及以前的数据")

# ==========================================================
# 2) Pydantic schema 兜底（动态加载 + postponed annotations 常见坑）
# ==========================================================
def _ensure_schema_ready(schema: Optional[type[BaseModel]], *, execute_fn=None) -> None:
    """
    对 schema 做一次 model_rebuild，避免：
    - 动态加载模块（importlib）时 sys.modules/globalns 不稳定
    - postponed annotations 下 List/Optional/Union 等无法解析
    """
    if schema is None:
        return
    if not hasattr(schema, "model_rebuild"):
        return

    # 1) 直接 rebuild
    try:
        schema.model_rebuild(force=True) 
        return
    except Exception:
        pass

    # 2) 带 execute_fn globals 作为 types namespace
    if execute_fn is not None:
        try:
            ns = getattr(execute_fn, "__globals__", {}) or {}
            schema.model_rebuild(force=True, _types_namespace=ns) 
            return
        except Exception:
            pass

    # 3) 最后兜底：空 namespace
    try:
        schema.model_rebuild(force=True, _types_namespace={})  # type: ignore[attr-defined]
    except Exception:
        pass


def _auto_args_schema_from_execute(execute_fn, *, model_name: str) -> type[BaseModel]:
    """
    从 execute(self, ctx: SkillContext, ...) 的签名自动生成 args_schema
    - 排除 self / ctx
    - 排除 **kwargs/*args
    - default 缺省 -> 必填(...)
    """
    sig = inspect.signature(execute_fn)
    fields: Dict[str, Any] = {}

    try:
        resolved_hints = get_type_hints(execute_fn, globalns=getattr(execute_fn, "__globals__", {}) or {})
    except Exception:
        resolved_hints = {}

    for name, p in sig.parameters.items():
        if name in ("self", "ctx"):
            continue
        if p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue

        ann = resolved_hints.get(name, Any)  
        default = p.default if p.default is not inspect._empty else ...
        fields[name] = (ann, default)

    model = create_model(
        f"{model_name}_Args",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )

    # 确保 forward ref 最终完全解析
    try:
        model.model_rebuild(force=True, _types_namespace=getattr(execute_fn, "__globals__", {}) or {})
    except Exception:
        pass

    return model

# ==========================================
# 3) 通用技能底座 (The Engine Chassis)
# ==========================================
class BaseSkill(ABC):
    """
    【通用技能基类】
    处理所有脏活累活：错误捕获、日志记录、LangChain 适配。
    """
    name: str = ""
    chinese_name: str = ""
    description: str = ""
    expert_mindset: str = "" # 从 SKILL.md 注入

    args_schema: Optional[type[BaseModel]] = None
    @abstractmethod
    def execute(self, ctx: SkillContext, **kwargs) -> SkillResult:
        """子类实现：核心业务逻辑"""
        raise NotImplementedError

    def safe_run(self, ctx: SkillContext, **kwargs) -> SkillResult:
        """ 系统调用的实际入口 (Template Method 模式)"""
        try:
            result = self.execute(ctx, **kwargs)
            if not isinstance(result, SkillResult):
                return SkillResult.fail(f"代码错误: execute 必须返回 SkillResult，实际返回 {type(result)}")  
            return result
            
        except Exception as e:
            # 捕获所有未预料的异常 (防崩坏)
            traceback.print_exc()
            return SkillResult.fail(f"{self.name or self.__class__.__name__} 运行异常: {str(e)}")

    def _dump_result(self, result: SkillResult) -> Dict[str, Any]:
        """
        将 SkillResult 转 dict
        - 正常用 model_dump
        - 极端兜底手动拼
        """
        try:
            return result.model_dump()
        except Exception:
            return {
                "success": result.success,
                "data": result.data,
                "insight": result.insight,
                "visuals": result.visuals,
                "error_msg": result.error_msg,
            }

    def to_langchain_tool(self, ctx: SkillContext):
        """
        适配成 LangChain StructuredTool
        - args_schema：优先用子类显式 schema，否则自动生成并缓存
        - return：统一 JSON 字符串（避免 dict -> str() 单引号污染）
        """
        from langchain_core.tools import StructuredTool

        # 1) 描述拼接
        full_desc = self.description or ""
        if self.expert_mindset:
            full_desc += f"\n\n[Expert Guide]\n{self.expert_mindset[:2000]}"

        # 2) schema 选择 / 自动生成
        schema = getattr(self, "args_schema", None)
        if schema is None:
            cached = getattr(self, "_lc_args_schema", None)
            if cached is None:
                cached = _auto_args_schema_from_execute(self.execute, model_name=self.name or self.__class__.__name__)
                setattr(self, "_lc_args_schema", cached)
            schema = cached
        
        _ensure_schema_ready(schema, execute_fn=self.execute)

        def _func(**kwargs):
            res = self.safe_run(ctx, **kwargs)
            payload = self._dump_result(res)
            # 统一返回 JSON 字符串
            return json.dumps(payload, ensure_ascii=False)

        return StructuredTool(
            name=self.name,
            description=full_desc,
            args_schema=schema,
            func=_func,
        )
    
# ==========================================
# 4) 金融特化版 (The Business Template)
# ==========================================
class BaseFinanceSkill(BaseSkill):
    """
    【金融技能模板】
    业务人员继承它即可，重点提供两类能力：
    1) 防未来：apply_date_filter
    2) 常用切片与截面排序：get_entity_data / rank_by_column
    """
    _DATE_COL_CANDIDATES = ("date", "data", "tradingdate", "pub_date", "time", "timestamp", "setup_date", "list_date")
    _ID_COL_CANDIDATES = ("symbol", "code", "id", "user_id", "fund_code", "ts_code", "masterfundcode")

    # --- 核心工具 1: 时间时光机 (防止未来函数) ---
    def apply_date_filter(self, df: pd.DataFrame, ref_date: Optional[str]) -> pd.DataFrame:
        """
        [时间切割] 根据 ref_date 切掉“未来数据”。
        逻辑：保留 date < ref_date 的数据 (假设 ref_date 是调仓执行日，只能看前一天收盘)
        """
        if df is None or df.empty or not ref_date:
            return df
                
        date_col = next((c for c in df.columns if c.lower() in ["date", "tradingdate", "setup_date","time"]), None)
        if not date_col: return df

        try:
            out = df.copy()

            if not is_datetime64_any_dtype(out[date_col]):
                if is_object_dtype(out[date_col]) or is_string_dtype(out[date_col]):
                    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
                else:
                    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

            target_dt = pd.to_datetime(ref_date, errors="coerce")
            if pd.isna(target_dt):
                return df

            out = out.dropna(subset=[date_col])
            return out[out[date_col] < target_dt]
        except Exception as e:
            print(f"⚠️ [DateFilter] 时间切割失败: col={date_col}, ref_date={ref_date}, err={e}")
            return df

    # --- 核心工具 2: 数据切片 ---
    def get_entity_data(
        self,
        ctx: SkillContext,
        table_name: str,
        entity_id: str,
        id_col_names: Optional[List[str]] = None,  
    ) -> pd.DataFrame:
        """
        [数据切片] 从案卷大表中切出指定实体（ETF/股票/用户/网点...）的数据
        - 自动识别 ID 列（code/symbol/id...）
        - 自动按日期排序（如果存在日期列）
        - 自动防未来（按 ctx.ref_date 过滤）
        """
        df = ctx.dossier.get_table(table_name)
        if df is None or df.empty:
            raise ValueError(f"案卷中找不到表格 '{table_name}' 或表为空")

        df = self.apply_date_filter(df, ctx.ref_date)
        if df.empty:
            return df

        candidates = tuple((id_col_names or list(self._ID_COL_CANDIDATES)))
        target_col = next((col for col in df.columns if str(col).strip().lower() in candidates), None)
        if not target_col:
            raise ValueError(f"表格 '{table_name}' 中找不到 ID 列，无法切片")

        df_sub = df[df[target_col].astype(str) == str(entity_id)].copy()
        if df_sub.empty:
            return df_sub   
        
        # 按日期排序（如果存在日期列）
        date_col = next((c for c in df_sub.columns if str(c).strip().lower() in self._DATE_COL_CANDIDATES), None)
        if date_col:
            try:
                if df_sub[date_col].dtype == "object":
                    df_sub[date_col] = pd.to_datetime(df_sub[date_col], errors="coerce")
                df_sub = df_sub.sort_values(date_col)
            except Exception:
                pass

        return df_sub
            

    # --- 核心工具 3: 截面排序 ---
    def rank_by_column(self, 
                    ctx: SkillContext, 
                    table_name: str,
                    score_col: str, 
                    ascending: bool = False, 
                    top_k: int = 5) -> pd.DataFrame:
        """
        [截面排序] 默认选“最新日期”的截面后排序取 TopK
        用途示例：
        - ETF：选成交额 amount 最大的前 5
        - 银行：选流失最严重的网点
        """
        df = ctx.dossier.get_table(table_name)
        if df is None or df.empty:
            return pd.DataFrame()

        df = self.apply_date_filter(df, ctx.ref_date)
        if df is None or df.empty:
            return pd.DataFrame()

        # 1) 锁定最新日期截面（如果存在日期列）
        date_col = next((c for c in df.columns if str(c).strip().lower() in self._DATE_COL_CANDIDATES), None)
        if date_col:
            try:
                d = df.copy()
                if not is_datetime64_any_dtype(d[date_col]):
                    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
                d = d.dropna(subset=[date_col])
                if not d.empty:
                    latest = d[date_col].max()
                    d = d[d[date_col] == latest].copy()
                df = d
            except Exception as e:
                print(f"⚠️ [Rank] 日期列解析失败: col={date_col}, err={e}")

        # 2) 校验排序列
        if score_col not in df.columns:
            raise ValueError(f"无法排序，列 '{score_col}' 不存在")

        # 3) 排序取 TopK
        df_sorted = df.sort_values(by=score_col, ascending=ascending)
        return df_sorted.head(int(top_k))
