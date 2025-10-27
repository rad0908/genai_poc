import os
import json
from pathlib import Path
import importlib.util
import pandas as pd
import pytest

# Dynamic import of the module under test (assumes tests/ is sibling to module file)
MODULE_NAME = "generate_diff_site"
_mod_spec = importlib.util.spec_from_file_location(
    MODULE_NAME,
    str(Path(__file__).resolve().parent.parent / "generate_diff_site.py"),
)
if _mod_spec and _mod_spec.loader:  # pragma: no cover
    _mod = importlib.util.module_from_spec(_mod_spec)
    _mod_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
else:  # pragma: no cover
    import generate_diff_site as _mod


def test_as_list_helper():
    assert _mod._as_list(None) == []
    assert _mod._as_list("id") == ["id"]
    assert _mod._as_list(["a", "b"]) == ["a", "b"]


def test_eq_with_nulls_series():
    a = pd.Series([1, None, "x", pd.NA])
    b = pd.Series([1, None, "y", pd.NA])
    eq = _mod._eq_with_nulls_series(a, b)
    assert eq.tolist() == [True, True, False, True]


def test_generate_singlefile_report_pk_mismatch_counts(tmp_path: Path):
    df1 = pd.DataFrame({"ID": [2, 1, 3, 4], "Amount": [10.0, 20.0, None, 7.5], "Status": ["A", "B", "C", "D"]})
    df2 = pd.DataFrame({"ID": [1, 2, 3, 4], "Amount": [20.0, 10.0, None, 7.5], "Status": ["B", "A", "C", "Z"]})
    out_html = tmp_path / "report.html"
    m = _mod.generate_singlefile_report(df1, df2, primary_key=["ID"], out_html=str(out_html))

    assert m["rows_compared"] == 4
    assert m["row_mismatch_count"] >= 1
    assert m["cell_mismatch_count"] >= 1
    assert out_html.exists()
    assert (tmp_path / "summary.txt").exists()


def test_banner_no_diffs_and_no_red_cells(tmp_path: Path):
    df1 = pd.DataFrame({"ID": [1, 2], "V": [10, 20]})
    df2 = df1.copy()
    out_html = tmp_path / "ok.html"
    _mod.generate_singlefile_report(df1, df2, primary_key=["ID"], out_html=str(out_html))
    html = out_html.read_text(encoding="utf-8")
    assert "ok-banner" in html
    assert "No differences found" in html
    assert "class='bad'" not in html and 'class="bad"' not in html


def test_banner_with_diffs_and_red_cells(tmp_path: Path):
    df1 = pd.DataFrame({"ID": [1, 2], "V": [10, 20]})
    df2 = pd.DataFrame({"ID": [1, 2], "V": [10, 99]})
    out_html = tmp_path / "warn.html"
    _mod.generate_singlefile_report(df1, df2, primary_key=["ID"], out_html=str(out_html))
    html = out_html.read_text(encoding="utf-8")
    assert "warn-banner" in html
    assert ("class='bad'" in html) or ('class="bad"' in html)


def test_summary_txt_contents(tmp_path: Path):
    df1 = pd.DataFrame({"ID": [1, 2], "V": [10, 20]})
    df2 = pd.DataFrame({"ID": [1, 2], "V": [10, 99]})
    out_html = tmp_path / "r.html"
    _mod.generate_singlefile_report(df1, df2, primary_key=["ID"], out_html=str(out_html))
    txt = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "----------------------count check---------------------" in txt
    assert "----------------------column names check---------------" in txt
    assert "----------------------row level comparison-----------------" in txt
    assert "-----------------------column level difference-------------" in txt


def test_api_response_and_paths(tmp_path: Path):
    df1 = pd.DataFrame({"ID": [1, 2, 3], "V": [10, 20, 30]})
    df2 = pd.DataFrame({"ID": [1, 2, 3], "V": [10, 20, 30]})

    def _fetch(db, sql):
        return df1 if db == "db1" else df2

    out_dir = tmp_path / "api"
    resp = _mod.datacompare_api(
        db1="db1", sql1="s1", db2="db2", sql2="s2", fetch_df_func=_fetch, primary_key=["ID"], out_dir=str(out_dir)
    )
    assert resp["status"] == "pass"
    assert (out_dir / "report.html").exists()
    assert (out_dir / "summary.txt").exists()
