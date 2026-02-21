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
