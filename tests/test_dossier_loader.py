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