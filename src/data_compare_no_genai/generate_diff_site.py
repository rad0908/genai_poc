# Static Data Compare — single-file HTML report + API wrapper
# Consolidated, dependency-light module (no web mode)

from __future__ import annotations
from typing import List, Optional, Union, Dict, Tuple

import os
import time
import json
from datetime import datetime

import pandas as pd

# ==============================================================
# Helpers
# ==============================================================

def _to_json_scalar(v):
    """Safe JSON-ish scalar conversion. Keeps numbers as numbers; datetimes to ISO; None stays None."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [str(x)]


def _eq_with_nulls_series(a: pd.Series, b: pd.Series) -> pd.Series:
    """Null-safe equality without pd.NA ambiguity.
    Strategy: replace nulls in *both* series with the SAME unique sentinel, then compare.
    Null==Null becomes True; returns a plain boolean Series.
    """
    SENTINEL = object()
    a2 = a.mask(a.isna(), other=SENTINEL)
    b2 = b.mask(b.isna(), other=SENTINEL)
    eq = a2.eq(b2)
    return pd.Series(eq, index=a.index).astype(bool)


def _stable_row_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Stable alignment key for no-PK case; duplicates preserved via (hash, order)."""
    if not cols:
        return pd.Series(range(len(df)), index=df.index)
    return pd.util.hash_pandas_object(df[cols], index=False)


def _deterministic_sort_no_pk(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Stable order via row hash + original position; avoids heavy multi-col sorts."""
    if not cols:
        return df.reset_index(drop=True)
    key = _stable_row_key(df, cols)
    order = (
        pd.DataFrame({"__k": key, "__i": range(len(df))})
        .sort_values(["__k", "__i"], kind="mergesort")
        ["__i"]
    )
    return df.iloc[order.values].reset_index(drop=True)


# ==============================================================
# HTML utilities
# ==============================================================

def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )

def _display_cell(v) -> str:
    """Pretty printer preserving type semantics for display.
    Strings are quoted, numbers/booleans use repr, datetimes ISO, nulls as ∅.
    """
    try:
        if pd.isna(v):
            return "∅"
    except Exception:
        pass
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, str):
        return f'"{v}"'
    return repr(v)

def _render_table(headers: List[str], rows: List[List[str]]) -> str:
    th = "".join(f"<th>{_escape_html(h)}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = "".join(f"<td>{_escape_html(x)}</td>" for x in r)
        trs.append(f"<tr>{tds}</tr>")
    table_html = f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
    return f"<div class='scroll'>{table_html}</div>"


def _render_table_with_cell_classes(
    headers: List[str],
    rows: List[List[str]],
    classes: List[List[Optional[str]]] | None = None,
) -> str:
    """Render a table where each cell may have a CSS class (e.g., 'bad')."""
    th = "".join(f"<th>{_escape_html(h)}</th>" for h in headers)
    trs = []
    for i, r in enumerate(rows):
        cls_row = classes[i] if classes and i < len(classes) else [None] * len(r)
        tds = []
        for j, val in enumerate(r):
            cls = (cls_row[j] or "").strip()
            attr = f' class="{cls}"' if cls else ""
            tds.append(f"<td{attr}>{_escape_html(val)}</td>")
        trs.append(f"<tr>{''.join(tds)}</tr>")
    table_html = f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
    return f"<div class='scroll'>{table_html}</div>"


# ==============================================================
# Summary.txt writer
# ==============================================================

def _write_summary_log(
    out_txt: str,
    *,
    dataset_name_1: str,
    dataset_name_2: str,
    count_df1: int,
    count_df2: int,
    rows_compared: int,
    row_mismatch_count: int,
    cell_mismatch_count: int,
    cols_only_in_df1: List[str],
    cols_only_in_df2: List[str],
    row_examples: List[Dict[str, object]],
    per_col_counts: Dict[str, int],
    col_examples: Dict[str, List[Tuple[str, str, Dict[str, object]]]],
    pk: List[str],
    elapsed_seconds: int,
) -> None:
    lines: List[str] = []
    lines.append("----------------------count check---------------------")
    lines.append(f"Dataset A: {dataset_name_1}")
    lines.append(f"Dataset B: {dataset_name_2}")
    lines.append(f"Rows in A - {count_df1}")
    lines.append(f"Rows in B - {count_df2}")
    lines.append(f"Rows compared - {rows_compared}")
    lines.append(f"Row mismatches - {row_mismatch_count}")
    lines.append(f"Cell mismatches - {cell_mismatch_count}")
    lines.append("")

    lines.append("----------------------column names check---------------")
    lines.append("Only in A: " + (", ".join(cols_only_in_df1) if cols_only_in_df1 else "(none)"))
    lines.append("Only in B: " + (", ".join(cols_only_in_df2) if cols_only_in_df2 else "(none)"))
    lines.append("")

    lines.append("----------------------row level comparison-----------------")
    if not row_examples:
        lines.append("(no row-level mismatches)")
    else:
        for e in row_examples[:5]:
            ident = ", ".join([f"{k}={e['pk'][k]}" for k in pk]) if pk else f"row={e['pk']['row']}"
            lines.append(ident)
            for (c, v1, v2) in e["cols"]:
                lines.append(f"  {c}: A={_display_cell(v1)} | B={_display_cell(v2)}")
    lines.append("")

    lines.append("-----------------------column level difference-------------")
    for c in sorted([k for k, v in per_col_counts.items() if v > 0], key=lambda x: per_col_counts[x], reverse=True):
        lines.append(f"column name - {c}")
        lines.append("df1\tdf2")
        examples = col_examples.get(c, [])
        if not examples:
            lines.append("(no examples)")
        else:
            for (v1, v2, _pkdict) in examples[:10]:
                lines.append(f"{_display_cell(v1)}\t{_display_cell(v2)}")

        lines.append("")

    lines.append(f"Time took to compare - {elapsed_seconds} sec")

    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==============================================================
# Single-file HTML Report (static mode)
# ==============================================================

def generate_singlefile_report(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    primary_key: Optional[Union[str, List[str]]] = None,
    dataset_name_1: str = "df1",
    dataset_name_2: str = "df2",
    out_html: str = "report.html",
) -> Dict[str, object]:
    t0 = time.perf_counter()

    pk = _as_list(primary_key)

    # Column presence
    cols1, cols2 = list(df1.columns), list(df2.columns)
    shared = [c for c in cols1 if c in cols2]
    cols_only_in_df1 = [c for c in cols1 if c not in cols2]
    cols_only_in_df2 = [c for c in cols2 if c not in cols1]

    # Counts
    count_df1, count_df2 = len(df1), len(df2)
    count_equal = (count_df1 == count_df2)

    # Align rows
    if pk:
        for c in pk:
            if c not in df1.columns:
                raise KeyError(f"PK column '{c}' missing in df1.")
            if c not in df2.columns:
                raise KeyError(f"PK column '{c}' missing in df2.")
        left = df1.set_index(pk, drop=False)
        right = df2.set_index(pk, drop=False)
        common = sorted(set(left.index.tolist()) & set(right.index.tolist()))
        left, right = left.loc[common].reset_index(drop=True), right.loc[common].reset_index(drop=True)
    else:
        # Deterministic, scalable alignment without PK
        left = _deterministic_sort_no_pk(df1, shared)
        right = _deterministic_sort_no_pk(df2, shared)
        m = min(len(left), len(right))
        left, right = left.iloc[:m], right.iloc[:m]

    n_compared = min(len(left), len(right))

    # Mismatch tracking
    per_col_counts: Dict[str, int] = {c: 0 for c in shared}
    row_mismatch_count = 0
    row_examples: List[Dict[str, object]] = []
    col_examples: Dict[str, List[Tuple[str, str, Dict[str, object]]]] = {c: [] for c in shared}

    for i in range(n_compared):
        bad_cols: List[str] = []
        for c in shared:
            eq = _eq_with_nulls_series(left.iloc[i:i+1][c], right.iloc[i:i+1][c]).iloc[0]
            if not bool(eq):
                bad_cols.append(c)
                per_col_counts[c] += 1
                # Capture up to 10 examples per column
                if len(col_examples[c]) < 10:
                    pkdict = {k: left.iloc[i][k] for k in pk} if pk else {"row": i}
                    v1 = left.iloc[i][c]
                    v2 = right.iloc[i][c]
                    col_examples[c].append((v1, v2, pkdict))  # store raw values
        if bad_cols:
            row_mismatch_count += 1
            if len(row_examples) < 10:
                entry: Dict[str, object] = {
                    "pk": ({k: left.iloc[i][k] for k in pk} if pk else {"row": i}),
                    "cols": []
                }
                for c in bad_cols:
                    v1 = left.iloc[i][c]
                    v2 = right.iloc[i][c]
                    entry["cols"].append((c, v1, v2))  # store raw values
                row_examples.append(entry)

    cell_mismatch_count = sum(per_col_counts.values())
    cols_with_mismatch = [c for c, k in per_col_counts.items() if k > 0]

    # --------- Build HTML ---------
    styles = """
    <style>
      body{font-family:Arial,Helvetica,sans-serif;margin:20px;color:#222}
      h1{margin:0 0 6px 0}
      h2{margin-top:28px}
      .muted{color:#666}
      .card{border:1px solid #ddd;padding:10px;border-radius:10px;margin-bottom:10px}
      .ok-banner{padding:10px 12px;border:1px solid #b7e1cd;background:#e9f7ef;color:#136f39;font-weight:700;border-radius:10px;margin:12px 0}
      .warn-banner{padding:10px 12px;border:1px solid #f4c7c3;background:#fdecea;color:#b71c1c;font-weight:700;border-radius:10px;margin:12px 0}
      .scroll{overflow-x:auto;max-width:100%}
      table{border-collapse:collapse;width:max-content;min-width:600px}
      th,td{border:1px solid #ddd;padding:6px 10px;white-space:nowrap}
      thead th{background:#f5f5f5}
      td.bad{background:#ffe0e0}
    </style>
    """

    # Section 1 — dataset names
    sec1 = f"""
    <h2>Section 1 — Dataset names</h2>
    <div class='card'>
      <div><b>Dataset A</b>: {_escape_html(dataset_name_1)}</div>
      <div><b>Dataset B</b>: {_escape_html(dataset_name_2)}</div>
    </div>
    """

    # Section 2 — count check
    sec2_tbl = _render_table(
        ["Metric", "Value"],
        [
            ["Rows in A", str(count_df1)],
            ["Rows in B", str(count_df2)],
            ["Rows compared", str(n_compared)],
            ["Counts equal?", "Yes" if count_equal else "No"],
            ["Row mismatches", str(row_mismatch_count)],
            ["Cell mismatches", str(cell_mismatch_count)],
        ],
    )
    sec2 = f"<h2>Section 2 — Count check</h2>{sec2_tbl}"

    # Section 3 — column names check
    sec3_rows = [
        ["Common columns", str(len(shared))],
        ["Columns only in A", ", ".join(cols_only_in_df1) or "(none)"],
        ["Columns only in B", ", ".join(cols_only_in_df2) or "(none)"],
        ["Columns mismatched?", "Yes" if (cols_only_in_df1 or cols_only_in_df2) else "No"],
    ]
    sec3 = f"<h2>Section 3 — Column names check</h2>{_render_table(['Metric','Value'], sec3_rows)}"

    # Section 4 — Top 10 mismatched rows (cell-level red)
    row_hdr = (["row"] if not pk else pk) + ["column", f"{dataset_name_1}", f"{dataset_name_2}"]
    row_rows: List[List[str]] = []
    row_cls: List[List[Optional[str]]] = []
    for e in row_examples:
        id_cells = ([str(e['pk']['row'])] if not pk else [str(e['pk'][k]) for k in pk])
        for (c, v1, v2) in e["cols"]:
            row_rows.append(id_cells + [c, _display_cell(v1), _display_cell(v2)])
            row_cls.append([''] * (len(id_cells) + 1) + ['bad', 'bad'])
    sec4 = (
        f"<h2>Section 4 — Top 10 mismatched rows</h2>"
        + _render_table_with_cell_classes(
            row_hdr, row_rows or [["(none)"]*len(row_hdr)], row_cls or [[None]*len(row_hdr)]
        )
    )

    # Section 5 — Column-level mismatches (10 examples per column)
    cols_sorted = sorted([c for c in shared if per_col_counts[c] > 0], key=lambda c: per_col_counts[c], reverse=True)
    col_blocks: List[str] = []
    for c in cols_sorted:
        examples = col_examples.get(c, [])
        hdr = (["row"] if not pk else pk) + [f"{dataset_name_1}", f"{dataset_name_2}"]
        rows5: List[List[str]] = []
        rows5_cls: List[List[Optional[str]]] = []
        for (v1, v2, pkdict) in examples[:10]:
            id_cells = ([str(pkdict['row'])] if not pk else [str(pkdict[k]) for k in pk])
            rows5.append(id_cells + [_display_cell(v1), _display_cell(v2)])
            rows5_cls.append([''] * len(id_cells) + ['bad', 'bad'])
        block = f"<h3>{_escape_html(c)} — {per_col_counts[c]} mismatches</h3>" + _render_table_with_cell_classes(hdr, rows5 or [["(none)"]*len(hdr)], rows5_cls or [[None]*len(hdr)])
        col_blocks.append(block)
    sec5 = f"<h2>Section 5 — Column-level mismatches</h2>" + "".join(col_blocks)

    elapsed = int(round(time.perf_counter() - t0))

    no_diffs = (
        row_mismatch_count == 0 and cell_mismatch_count == 0 and not cols_only_in_df1 and not cols_only_in_df2
    )
    banner = (
        "<div class='ok-banner'><b>No differences found</b></div>"
        if no_diffs
        else "<div class='warn-banner'><b>Differences found</b> — mismatches highlighted in red below.</div>"
    )

    html = f"""<!doctype html>
<html>
<head><meta charset='utf-8'><title>Data Comparison Report</title>{styles}</head>
<body>
  <h1>Data Comparison Report</h1>
  <div class='muted'>Generated {datetime.now().isoformat(timespec='seconds')} · Time took - {elapsed} sec</div>
  {banner}
  {sec1}
  {sec2}
  {sec3}
  {sec4}
  {sec5}
</body></html>"""

    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    # Also write summary.txt
    out_txt = os.path.join(os.path.dirname(out_html) or ".", "summary.txt")
    _write_summary_log(
        out_txt,
        dataset_name_1=dataset_name_1,
        dataset_name_2=dataset_name_2,
        count_df1=count_df1,
        count_df2=count_df2,
        rows_compared=n_compared,
        row_mismatch_count=row_mismatch_count,
        cell_mismatch_count=cell_mismatch_count,
        cols_only_in_df1=cols_only_in_df1,
        cols_only_in_df2=cols_only_in_df2,
        row_examples=row_examples,
        per_col_counts=per_col_counts,
        col_examples=col_examples,
        pk=pk,
        elapsed_seconds=elapsed,
    )

    return {
        "out_html": out_html,
        "rows_compared": n_compared,
        "row_mismatch_count": row_mismatch_count,
        "columns_with_mismatch": len(cols_with_mismatch),
        "cell_mismatch_count": cell_mismatch_count,
        "columns_only_in_df1": cols_only_in_df1,
        "columns_only_in_df2": cols_only_in_df2,
        "elapsed_seconds": elapsed,
    }


# ==============================================================
# API Orchestration (DB fetch -> report)
# ==============================================================

def datacompare_api(
    *,
    db1: str,
    sql1: str,
    db2: str,
    sql2: str,
    fetch_df_func,
    primary_key: Optional[Union[str, List[str]]] = None,
    out_dir: str = "diff_report",
    dataset_name_1: str = "df1",
    dataset_name_2: str = "df2",
) -> Dict[str, object]:
    t0 = time.perf_counter()

    df1 = fetch_df_func(db1, sql1)
    df2 = fetch_df_func(db2, sql2)

    os.makedirs(out_dir, exist_ok=True)
    out_html = os.path.join(out_dir, "report.html")

    manifest = generate_singlefile_report(
        df1,
        df2,
        primary_key=primary_key,
        dataset_name_1=dataset_name_1,
        dataset_name_2=dataset_name_2,
        out_html=out_html,
    )

    diff_found = (
        manifest["row_mismatch_count"] > 0
        or manifest["cell_mismatch_count"] > 0
        or manifest["columns_with_mismatch"] > 0
        or manifest["columns_only_in_df1"]
        or manifest["columns_only_in_df2"]
    )

    elapsed = int(round(time.perf_counter() - t0))

    return {
        "status": "fail" if diff_found else "pass",
        "message": "differences found" if diff_found else "comparsion passed no differences found",
        "elapsed_seconds": elapsed,
        "summary": manifest,
        "artifact": {"html": out_html, "summary_txt": os.path.join(out_dir, "summary.txt")},
        "params": {
            "db1": db1,
            "db2": db2,
            "primary_key": primary_key,
        },
    }


# ==============================================================
# Inline smoke tests (run this file directly)
# ==============================================================

def _run_inline_tests():
    print("Running inline tests...")

    # Helper tests
    assert _as_list(None) == []
    assert _as_list("id") == ["id"]
    assert _as_list(["a", "b"]) == ["a", "b"]

    a = pd.Series([1, None, "x", pd.NA])
    b = pd.Series([1, None, "y", pd.NA])
    eq = _eq_with_nulls_series(a, b)
    assert eq.tolist() == [True, True, False, True]

    # Static report — basic PK diff
    df1 = pd.DataFrame({"ID": [2, 1, 3, 4], "Amount": [10.0, 20.0, None, 7.5], "Status": ["A", "B", "C", "D"]})
    df2 = pd.DataFrame({"ID": [1, 2, 3, 4], "Amount": [20.0, 10.0, None, 7.5], "Status": ["B", "A", "C", "Z"]})
    m = generate_singlefile_report(df1, df2, primary_key=["ID"], dataset_name_1="A", dataset_name_2="B", out_html="_out_report.html")
    assert m["rows_compared"] == 4
    assert m["row_mismatch_count"] >= 1
    assert m["cell_mismatch_count"] >= 1

    # Column names difference
    df3 = pd.DataFrame({"ID": [1, 2], "V": [None, 5]})
    df4 = pd.DataFrame({"ID": [1, 2], "V": [None, 5], "Extra": [9, 9]})
    m2 = generate_singlefile_report(df3, df4, primary_key=["ID"], dataset_name_1="A", dataset_name_2="B", out_html="_out_report2.html")
    assert m2["columns_only_in_df2"] == ["Extra"]

    # API pass/fail
    def _fetch(db, sql):
        return df3 if db == "db1" else df4

    resp = datacompare_api(db1="db1", sql1="s1", db2="db2", sql2="s2", fetch_df_func=_fetch, primary_key=["ID"], out_dir="_api")
    assert resp["status"] == "fail"

    print("All inline tests passed.")


if __name__ == "__main__":
    _run_inline_tests()
    # Quick demo artifact
    df1 = pd.DataFrame({"ID": [1, 2, 3], "Value": [10, 20, 30]})
    df2 = pd.DataFrame({"ID": [1, 2, 3], "Value": [10, 99, 30]})
    print(generate_singlefile_report(df1, df2, primary_key="ID", out_html="_demo/report.html"))
