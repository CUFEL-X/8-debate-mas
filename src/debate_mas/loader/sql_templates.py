from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional


def get_universal_query(
    table_name: str,
    # 原有参数
    date_col: str | None = None,
    limit: int | None = 10000,
    order: str = "DESC",
    filters: dict | None = None,
    columns: list[str] | None = None,
    # 调仓日 + 回溯天数
    ref_date: str | None = None,       
    lookback_days: int | None = None,  
    **kwargs
) -> str:
    """
    [万能模版: universal_select]
    功能：
    1) columns 为空 -> SELECT *
    2) 支持简单等值 filters -> AND col = 'val'
    3) ref_date + lookback_days 做时间窗口：
       区间：[ref_date - lookback_days, ref_date) —— 严格小于 ref_date，防未来数据
    4) 默认按时间倒序排序
    5) limit=None 时不加 LIMIT
    """
    TABLE_DATE_MAP: Dict[str, str] = {
        "etf_daily": "TradingDate",
        # TODO: 可扩展
    }

    # 按表默认列名映射
    if (date_col is None or date_col == "date") and table_name in TABLE_DATE_MAP:
        date_col = TABLE_DATE_MAP[table_name]

    # SELECT 部分
    if not columns or columns == ["*"]:
        select_part = "*"
    else:
        select_part = ", ".join(columns)

    sql = f"SELECT {select_part} FROM {table_name} WHERE 1 = 1"

    # filters（等值）
    if filters:
        for col, val in filters.items():
            if isinstance(val, str):
                sql += f" AND {col} = '{val}'"
            else:
                sql += f" AND {col} = {val}"

    # 时间窗口
    if date_col and ref_date and lookback_days is not None:
        try:
            ref_dt = datetime.fromisoformat(ref_date).date()
            start_dt = ref_dt - timedelta(days=int(lookback_days))
            sql += f" AND {date_col} >= toDate('{start_dt.isoformat()}')"
            sql += f" AND {date_col} < toDate('{ref_dt.isoformat()}')"
        except Exception:
            pass

    # 排序（如果有时间列，就按时间排序）
    if date_col:
        sql += f" ORDER BY {date_col} {order}"

    # Limit（兜底防止查爆）
    if limit is not None:
        sql += f" LIMIT {limit}"

    return sql


TEMPLATE_REGISTRY = {
    "universal": get_universal_query
}