from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from debate_mas.protocol import EtfCandidate, SkillResult
from debate_mas.skills.base import BaseFinanceSkill, SkillContext

from .mapping import (
    GUARDRAIL_BUCKETS,
    GUARDRAIL_NOTE,
    INDUSTRY_FUZZY_MAP,
    THEME_KEYWORDS_MAP,
)
from .ontology import get_concept_meta

Mode = Literal["ontology_mapping", "industry_frequency", "guardrail_pool", "user_custom"]

class SkillHandler(BaseFinanceSkill):
    """
    [Hunter] 主题挖掘机 - Recall 模块 (The Theme Miner)
    
    设计:
    - Recall (召回): 从海量 ETF 中找出与“概念/行业/政策”相关的标的。
    - Evidence (证据): 不仅给代码，还要给“含权量” (Match Count) 和“政策强度” (Policy Strength)。
    - No Rank (无排序): 排序交给 quantitative_sniper，这里只做“海选”。
    """
    SKILL_NAME = "theme_miner"
    TABLE_ETF = "etf_basic"
    TABLE_GOV = "govcn"
    OUTPUT_TYPE = "EtfCandidateList"

    def execute(self, 
                ctx: SkillContext, 
                mode: Mode = "ontology_mapping",
                keyword: str = "",
                event_text: str = "",
                days: int = 30, 
                top_k: int = 10,
                top_industries: int = 3,
                guardrail_buckets: Optional[List[str]] = None,  
                per_bucket_k: int = 3,
                ) -> SkillResult:
        # ==========================================================
        # 1. 基础数据准备 (Data Preparation)
        # ==========================================================
        df_etf = ctx.dossier.get_table(self.TABLE_ETF)
        if df_etf is None:
            return SkillResult.fail("案卷中缺失 'etf_basic' 基础信息表。")

        df_etf = self._norm_cols(df_etf)
        df_etf = self._apply_etf_setup_date_filter(df_etf, ctx.ref_date)

        # 识别关键列（code/name）
        self._code_col, self._name_col = self._infer_etf_cols(df_etf)
        if not self._code_col or not self._name_col:
            return SkillResult.fail(f"etf_basic 表缺少必要的 code 或 name 列。现有: {list(df_etf.columns)}")

        # ==========================================================
        # 2. 模式路由 (Mode Dispatch)
        # ==========================================================
        # Mode A: 结构兜底 (Guardrail Pool) - 纯规则，不依赖政策表
        if mode == "guardrail_pool":
            return self._run_guardrail_pool(
                df_etf=df_etf,
                top_k=top_k,
                guardrail_buckets=guardrail_buckets,
                per_bucket_k=per_bucket_k,
                ctx=ctx,
            )
        
        # 以下模式需要政策表
        df_gov = ctx.dossier.get_table(self.TABLE_GOV)
        if df_gov is None:
            return SkillResult.fail("案卷中缺失 'govcn' 政策表。")

        df_gov = self.apply_date_filter(df_gov, ctx.ref_date)
        if df_gov.empty:
            return SkillResult.fail(f"截止 {ctx.ref_date} 无可用政策数据。")
        df_gov = self._norm_cols(df_gov)

        # Mode B: 主题映射 (Ontology Mapping) - 核心模式
        if mode == "ontology_mapping":
            if not keyword:
                return SkillResult.fail("ontology_mapping 模式必须提供 keyword 参数。")
            return self._run_ontology_mapping(df_gov, df_etf, keyword=keyword, days=days, top_k=top_k, ctx=ctx)

        # Mode C: 行业词频 (Industry Frequency) - 数据驱动
        if mode == "industry_frequency":
            return self._run_industry_frequency(df_gov, df_etf, days=days, top_k=top_k, top_industries=top_industries, ctx=ctx)

        # Mode D: 自定义 (User Custom) - 练习者接口
        if mode == "user_custom":
            return self._user_custom_logic(df_gov, df_etf, keyword=keyword, days=days, top_k=top_k)

        return SkillResult.fail(f"不支持的模式: {mode}")

    def _ok_candidates(
        self,
        ctx: SkillContext,
        out: List[EtfCandidate],
        insight: str,
        meta: Dict[str, Any],
    ) -> SkillResult:
        data = {
            "type": self.OUTPUT_TYPE,
            "items": [c.model_dump() for c in out],
            "meta": {
                "ref_date": ctx.ref_date,
                "agent_role": ctx.agent_role,
                **(meta or {}),
            },
        }
        return SkillResult.ok(data=data, insight=insight)

    # =========================================================================
    # 核心业务逻辑 (Core Business Logic)
    # =========================================================================
    def _run_ontology_mapping(self, df_gov: pd.DataFrame, df_etf: pd.DataFrame, keyword: str, days: int, top_k: int, ctx: SkillContext) -> SkillResult:
        """
        [策略] 知识图谱映射：从“宏观概念”推导到“具体 ETF”
        
        Step 1: 概念扩展 (Concept Expansion) -> 从 ontology.yaml 查表
        Step 2: 政策验证 (Policy Verification) -> 去 govcn 找证据
        Step 3: 标的召回 (ETF Recall) -> 去 etf_basic 模糊匹配
        """
        # --- Step 1: Concept Expansion ---
        meta = get_concept_meta(keyword)
        if not meta.get("found", False):
            return SkillResult.fail(f"知识库未收录 '{keyword}'，请先在 references/ontology.yaml 中补充 aliases/expands_to。")

        std_theme = str(meta.get("name", keyword))
        expansions = [str(x) for x in (meta.get("expansions", []) or []) if str(x).strip()]
        static_weight = float(meta.get("weight", 1.0))

        # --- Step 2: Policy Verification ---
        matched_docs, evidence_str, policy_strength = self._search_documents(df_gov, keyword=std_theme, days=days)
        if std_theme != keyword and len(matched_docs) == 0:
             matched_docs, evidence_str, policy_strength = self._search_documents(df_gov, keyword=keyword, days=days)

        # --- Step 3: ETF Recall ---
        terms = self._uniq_keep([keyword, std_theme] + expansions)

        hits = []
        for t in terms:
            h = self._match_etfs(df_etf, t)
            if not h.empty:
                h = h.copy()
                h["_match_term"] = t 
                hits.append(h)

        if not hits:
            return SkillResult.fail(f"主题命中，但未在 etf_basic 中召回到 ETF（尝试词: {terms[:6]}）。")

        # --- Step 4: Aggregation & Scoring ---
        m = pd.concat(hits, ignore_index=True)
        m[self._code_col] = m[self._code_col].astype(str)

        agg = m.groupby(self._code_col).agg(
            etf_name=(self._name_col, "first"),
            hit_terms=("_match_term", "nunique"), 
        ).reset_index()

        agg["score"] = (10.0 + 5.0 * agg["hit_terms"]).clip(lower=1.0, upper=60.0)
        agg = agg.sort_values(["hit_terms", self._code_col], ascending=[False, True]).head(int(top_k))

        # --- Step 5: Wrap Result ---
        out: List[EtfCandidate] = []
        for _, r in agg.iterrows():
            code = str(r[self._code_col])
            reason = f"主题:{std_theme}(w={static_weight}) | 命名命中:{int(r['hit_terms'])}项 | 证据:{evidence_str if evidence_str else '无'}"
            
            out.append(EtfCandidate(
                symbol=code,
                score=float(round(r["score"], 2)),
                reason=reason,
                source_skill=self.SKILL_NAME,
                extra={
                    "mode": "ontology_mapping",
                    "theme": std_theme,
                    "static_weight": static_weight,
                    "search_terms": terms[:10],
                    "policy_docs": int(len(matched_docs)) if matched_docs is not None else 0,
                    "policy_strength": float(policy_strength),
                },
            ))

        insight = f"[ontology_mapping] 主题'{std_theme}' 召回 {len(out)} 只 ETF，政策证据 {len(matched_docs)} 条。"
        return self._ok_candidates(
            ctx=ctx,
            out=out,
            insight=insight,
            meta={
                "mode": "ontology_mapping",
                "keyword": keyword,
                "std_theme": std_theme,
                "days": int(days),
                "top_k": int(top_k),
                "policy_docs": int(len(matched_docs)) if matched_docs is not None else 0,
                "policy_strength": float(policy_strength),
            },
        )

    def _run_guardrail_pool(self, df_etf: pd.DataFrame, top_k: int, guardrail_buckets: Optional[List[str]], per_bucket_k: int, ctx: SkillContext) -> SkillResult:
        """
        [策略] 结构兜底：保证组合的“骨架”完整 (Bond, Gold, Cash...)
        不看新闻，只看配置需求。
        """
        buckets = guardrail_buckets or (GUARDRAIL_BUCKETS[:] if GUARDRAIL_BUCKETS else [])
        if not buckets:
            return self._ok_candidates(
                ctx=ctx,
                out=[],
                insight="[guardrail_pool] 未配置 GUARDRAIL_BUCKETS，跳过。",
                meta={"mode": "guardrail_pool", "buckets": []},
            )

        top_k = int(max(1, top_k))
        per_bucket_k = int(max(1, per_bucket_k))

        rows: List[Dict[str, Any]] = []

        for b in buckets:
            b = str(b).strip()
            terms = [str(x).strip() for x in (THEME_KEYWORDS_MAP.get(b, []) or []) if str(x).strip()]
            
            cnt = 0
            seen = set()

            for t in terms:
                h = self._match_etfs(df_etf, t)
                if h.empty: 
                    continue
                
                for _, r in h.iterrows():
                    code = str(r[self._code_col])
                    if code in seen: 
                        continue
                    seen.add(code)
                    
                    rows.append({
                        "code": code,
                        "etf_name": str(r.get(self._name_col, "")),
                        "bucket": b,
                        "hit_term": t,
                    })
                    cnt += 1
                    if cnt >= per_bucket_k: break
                if cnt >= per_bucket_k: break

        if not rows:
            return SkillResult.fail("[guardrail_pool] 未召回到任何 ETF (Mapping 配置可能为空)。")

        m = pd.DataFrame(rows)
        agg = (
            m.groupby("code")
            .agg(
                etf_name=("etf_name", "first"),
                buckets=("bucket", lambda xs: sorted(set(xs))),
                buckets_n=("bucket", "nunique"),
                hit_terms_n=("hit_term", "nunique"),
            )
            .reset_index()
        )

        agg["score"] = (12.0 + 3.0 * agg["buckets_n"] + 2.0 * agg["hit_terms_n"]).clip(1.0, 60.0)
        agg = agg.sort_values(["buckets_n", "hit_terms_n", "code"], ascending=[False, False, True]).head(top_k)

        out: List[EtfCandidate] = []
        for _, r in agg.iterrows():
            reason = f"结构兜底:{','.join(r['buckets'])} | {GUARDRAIL_NOTE}"
            out.append(
                EtfCandidate(
                    symbol=str(r["code"]),
                    score=float(round(r["score"], 2)),
                    reason=reason,
                    source_skill=self.SKILL_NAME,
                    extra={
                        "mode": "guardrail_pool",
                        "buckets": r["buckets"],
                        "note": GUARDRAIL_NOTE,
                    },
                )
            )

        insight = f"[guardrail_pool] Buckets={buckets} 产出 {len(out)} 只 ETF。"
        return self._ok_candidates(
            ctx=ctx,
            out=out,
            insight=insight,
            meta={
                "mode": "guardrail_pool",
                "buckets": buckets,
                "top_k": int(top_k),
                "per_bucket_k": int(per_bucket_k),
            },
        )
    
    def _run_industry_frequency(self, df_gov: pd.DataFrame, df_etf: pd.DataFrame, days: int, top_k: int, top_industries: int,ctx: SkillContext) -> SkillResult:
        """[策略] 行业词频：数据驱动的“热点发现”"""
        if "industry_name" not in df_gov.columns:
            return SkillResult.fail("govcn 表缺少 industry_name 字段。")

        # 1. 切片与统计
        d = self._slice_lookback(df_gov, days)
        if d.empty: return SkillResult.fail(f"窗口内无数据 (days={days})。")

        s = d["industry_name"].astype(str).map(self._clean_industry_name)
        s = s[s.str.len() > 0]
        if s.empty: return SkillResult.fail("industry_name 清洗后为空。")

        # 2. 统计 Top N 行业
        vc = s.value_counts()
        top_n = int(max(1, min(int(top_industries), len(vc))))
        top_inds = vc.head(top_n) 

        # 3. 映射 ETF

        hits: List[pd.DataFrame] = []
        for ind, freq in top_inds.items():
            terms = self._industry_terms(ind)
            for t in terms:
                h = self._match_etfs(df_etf, t)
                if h.empty:
                    continue
                h = h.copy()
                h["_industry"] = ind
                h["_freq"] = int(freq)
                h["_match_term"] = t
                hits.append(h)

        if not hits:
            return SkillResult.fail(f"Top行业 {list(top_inds.index)} 未能召回 ETF。")

        # 4. 聚合打分
        m = pd.concat(hits, ignore_index=True)
        m[self._code_col] = m[self._code_col].astype(str)

        agg = (
            m.groupby(self._code_col)
            .agg(
                etf_name=(self._name_col, "first"),
                industry=("_industry", "first"),
                freq=("_freq", "max"),
                hit_terms=("_match_term", "nunique"),
            )
            .reset_index()
        )

        agg["score"] = (10.0 + 2.0 * agg["freq"] + 3.0 * agg["hit_terms"]).clip(1.0, 60.0)
        agg = agg.sort_values(["freq", "hit_terms", self._code_col], ascending=[False, False, True]).head(int(top_k))

        # 5. 封装
        out: List[EtfCandidate] = []
        for _, r in agg.iterrows():
            reason = f"行业高频:{r['industry']}({r['freq']}) | 窗口:{days}天"
            out.append(EtfCandidate(
                symbol=str(r[self._code_col]),
                score=float(round(r["score"], 2)),
                reason=reason,
                source_skill=self.SKILL_NAME,
                extra={
                    "mode": "industry_frequency",
                    "industry": r["industry"],
                    "freq": int(r["freq"]),
                    "lookback_days": int(days),
                },
            ))

        insight = f"[industry_frequency] Top行业={list(top_inds.index)} 召回 {len(out)} 只 ETF。"
        return self._ok_candidates(
            ctx=ctx,
            out=out,
            insight=insight,
            meta={
                "mode": "industry_frequency",
                "days": int(days),
                "top_k": int(top_k),
                "top_industries": int(top_industries),
                "top_industry_list": list(top_inds.index),
            },
        )

    # ================= 🚀 练习者扩展接口 (Student Lab) =================
    def _user_custom_logic(self, df_gov: pd.DataFrame, df_etf: pd.DataFrame, keyword: str, days: int, top_k: int) -> SkillResult:
        """
        [TODO] 练习者请在此处编写你的自定义召回逻辑
        输入: df_gov 为已防未来切片后的 govcn；df_etf 为已列标准化且已过滤 setup_date<=ref_date 的 etf_basic
        输出: 必须返回 SkillResult(data.type="EtfCandidateList")，items 内 symbol/score/reason/extra
        验收: mode="user_custom" 时能稳定召回；extra 至少写入 mode/keyword/lookback_days 等复现信息
        """
        return SkillResult.fail("自定义召回逻辑尚未实现：请在 _user_custom_logic 内补全代码。")

    # ================= 工具函数 (Helper Methods) =================
    def _norm_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=lambda x: str(x).strip().lower())

    def _infer_etf_cols(self, df_etf: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """智能推断 Code 和 Name 列名"""
        code_col = next((c for c in df_etf.columns if c in ["code", "symbol", "ts_code", "masterfundcode"]), None)
        name_col = next((c for c in df_etf.columns if c in ["cname", "csname", "name", "extname"]), None)
        return code_col, name_col
    
    def _apply_etf_setup_date_filter(self, df_etf: pd.DataFrame, ref_date: str) -> pd.DataFrame:
        """防未来: 剔除 ref_date 之后成立的 ETF"""
        d = df_etf.copy()
        ref = pd.to_datetime(ref_date, errors="coerce")
        if pd.isna(ref): return d

        date_col = next((c for c in ["setup_date", "list_date", "pub_date", "base_date", "date"] if c in d.columns), None)
        if not date_col: return d

        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col])
        return d[d[date_col] <= ref].copy()
    
    def _slice_lookback(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """时间窗口切片"""
        d = df.copy()
        date_col = next((c for c in d.columns if c in ["date", "pub_date", "time"]), None)
        if not date_col: return d
        
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col])
        if d.empty: return d
        
        latest_dt = d[date_col].max()
        cutoff = latest_dt - pd.Timedelta(days=int(days))
        return d[d[date_col] >= cutoff]

    def _uniq_keep(self, xs: List[str]) -> List[str]:
        """保持顺序去重"""
        out: List[str] = []
        for x in xs:
            x = str(x).strip()
            if x and x not in out: out.append(x)
        return out

    def _clean_industry_name(self, s: str) -> str:
        """行业名清洗 (e.g. '汽车制造行业' -> '汽车制造')"""
        s = str(s).strip()
        if not s or s.lower() in ["nan", "none"]: return ""
        if s in ["综合", "其他服务业", "其他制造业", "其他金融业"]: return ""
        for suf in ["行业", "产业", "领域", "相关"]:
            if s.endswith(suf): s = s[: -len(suf)]
        return s.strip()

    def _industry_terms(self, industry_name: str) -> List[str]:
        """行业名 -> 搜索词库 (利用 mappings.yaml 进行模糊扩展)"""
        ind = str(industry_name).strip()
        if not ind: return []

        base = ind
        for suf in ["制造业", "服务业", "加工业", "供应业", "管理业", "运输业", "开采业", "采选业", "建筑业", "利用业", "业"]:
            if base.endswith(suf): base = base[: -len(suf)]
        base = base.strip()

        extras: List[str] = []
        if base in INDUSTRY_FUZZY_MAP:
            extras.extend(INDUSTRY_FUZZY_MAP.get(base, []) or [])
        else:
            for k, vs in (INDUSTRY_FUZZY_MAP or {}).items():
                k = str(k).strip()
                if not k: continue
                if k in base or base in k:
                    extras.extend(vs or [])

        return self._uniq_keep([ind, base] + [str(x).strip() for x in extras if str(x).strip()])
    
    def _search_documents(self, df: pd.DataFrame, keyword: str, days: int) -> Tuple[pd.DataFrame, str, float]:
        """[工具] 搜索政策文档 (支持时间衰减评分)"""
        text_cols = [c for c in df.columns if c in ["title", "content", "context", "industry_name"]]
        if not text_cols: return pd.DataFrame(), "", 0.0

        kw = str(keyword).strip()
        if not kw: return pd.DataFrame(), "", 0.0

        mask = pd.Series(False, index=df.index)
        for col in text_cols:
            mask |= df[col].astype(str).str.contains(kw, na=False)

        matched = df[mask].copy()
        if matched.empty: return matched, "", 0.0

        # 时间过滤
        date_col = next((c for c in matched.columns if c in ["date", "pub_date", "time"]), None)
        if date_col:
            try:
                matched[date_col] = pd.to_datetime(matched[date_col], errors="coerce")
                matched = matched.dropna(subset=[date_col])
                if not matched.empty:
                    anchor = matched[date_col].max()
                    cutoff = anchor - pd.Timedelta(days=int(days))
                    matched = matched[matched[date_col] >= cutoff].sort_values(date_col, ascending=False)
            except Exception: pass

        # 政策强度计算 (衰减模型)
        half_life_days = 30.0
        lam = math.log(2.0) / half_life_days
        policy_strength = float(len(matched))

        if date_col and not matched.empty:
            try:
                latest_dt = matched[date_col].max()
                age_days = (latest_dt - matched[date_col]).dt.days.clip(lower=0)
                weights = (-lam * age_days).apply(math.exp)
                policy_strength = float(weights.sum())
            except Exception: pass

        # 证据摘要
        show_col = "title" if "title" in text_cols else text_cols[0]
        evidence_list = matched[show_col].head(3).tolist()
        evidence_str = " | ".join([str(e)[:30] + "..." for e in evidence_list])

        return matched, evidence_str, policy_strength

    def _match_etfs(self, df: pd.DataFrame, keyword: str) -> pd.DataFrame:
        """[工具] 简单的模糊匹配 (Contains)"""
        kw = str(keyword).strip()
        if not self._name_col or not kw: return pd.DataFrame()
        mask = df[self._name_col].astype(str).str.contains(kw, na=False)
        return df[mask].copy()