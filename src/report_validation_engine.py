import os
import json
import pandas as pd
import openai
import sqlite3
from typing import List, Dict
from bs4 import BeautifulSoup

openai.api_key = "api_key"

def read_raw_excel(filepath: str) -> List[List[str]]:
    df_raw = pd.read_excel(filepath, header=None).fillna("")
    return df_raw.values.tolist()


def read_raw_html(filepath: str) -> List[List[str]]:
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    table = soup.find("table")
    if not table:
        return []
    rows = []
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        rows.append([cell.get_text(strip=True) for cell in cells])
    return rows


def extract_table_with_genai(raw_data: List[List[str]]) -> pd.DataFrame:
    prompt = f"""
    You are a data assistant. Extract the main table from this report dump.
    Identify headers and return only a valid JSON list of dictionaries (no explanation).

    Dump:
    {json.dumps(raw_data[:30])}
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response.choices[0].message.content.strip()
    try:
        json_str = content[content.index('['):content.rindex(']')+1]
        return pd.DataFrame(json.loads(json_str))
    except Exception as e:
        print(f"Failed to parse GPT response.\n{content}\n")
        raise e


def validate_dataframe(df: pd.DataFrame, rules: Dict, auto_create_db: bool = True) -> List[str]:
    errors = []

    for col in rules.get("required_columns", []):
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    for obj in rules.get("required_rows", []):
        if not any(df["Objective"].astype(str) == obj):
            errors.append(f"Missing row with Objective: {obj}")

    for key, expected in rules.get("cell_validations", {}).items():
        row_name, col_name = key.split(".")
        match = df[df["Objective"].astype(str) == row_name]
        if match.empty:
            errors.append(f"Row '{row_name}' not found for cell validation.")
        elif col_name not in df.columns:
            errors.append(f"Column '{col_name}' missing for validation on '{row_name}'")
        else:
            actual = match.iloc[0][col_name]
            if str(actual).replace(",", "") != str(expected):
                errors.append(f"Mismatch at {row_name}.{col_name}: expected {expected}, got {actual}")

    # SQL-based validations
    if "sql_validations" in rules:
        db_path = rules["sql_validations"].get("db_path", ":memory:")
        if auto_create_db and not os.path.exists(db_path) and db_path != ":memory:":
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE report_summary (objective TEXT, loan_count INTEGER, upb INTEGER)")
            cursor.executemany("INSERT INTO report_summary VALUES (?, ?, ?)", [
                ("Shared Equity", 343, 45),
                ("High needs rural population", 94567, 3456)
            ])
            conn.commit()
            conn.close()
        with sqlite3.connect(db_path) as conn:
            for rule in rules["sql_validations"].get("queries", []):
                query = rule["query"]
                compare_row = rule["row"]
                compare_col = rule["column"]
                expected_result = pd.read_sql_query(query, conn).iloc[0, 0]
                match = df[df["Objective"].astype(str) == compare_row]
                if match.empty:
                    errors.append(f"Row '{compare_row}' not found for SQL validation.")
                elif compare_col not in df.columns:
                    errors.append(f"Column '{compare_col}' missing for SQL validation on '{compare_row}'")
                else:
                    actual = match.iloc[0][compare_col]
                    if str(actual).replace(",", "") != str(expected_result):
                        errors.append(f"Mismatch from SQL at {compare_row}.{compare_col}: expected {expected_result}, got {actual}")

    return errors


def run_validations_on_folder(folder: str, auto_create_db: bool = True) -> List[tuple]:
    results = []
    for file in os.listdir(folder):
        if file.endswith(".xlsx") or file.endswith(".html"):
            path = os.path.join(folder, file)
            base_name = os.path.splitext(file)[0]
            rules_file = os.path.join(folder, f"{base_name}_rules.json")

            if not os.path.exists(rules_file):
                results.append((file, "ERROR", [f"Rules file not found: {rules_file}"]))
                continue

            try:
                with open(rules_file) as f:
                    rules = json.load(f)

                raw_data = read_raw_excel(path) if file.endswith(".xlsx") else read_raw_html(path)
                if not raw_data:
                    raise ValueError("No table found in the report.")
                df = extract_table_with_genai(raw_data)
                errors = validate_dataframe(df, rules, auto_create_db=auto_create_db)
                results.append((file, "PASS" if not errors else "FAIL", errors))
            except Exception as e:
                results.append((file, "ERROR", [str(e)]))

    with open("validation_summary.html", "w", encoding="utf-8") as f:
        f.write("<html><head><title>Validation Summary</title></head><body>")
        f.write("<h1>Validation Report Summary</h1>")
        total = len(results)
        passed = sum(1 for r in results if r[1] == 'PASS')
        failed = sum(1 for r in results if r[1] == 'FAIL')
        errors = sum(1 for r in results if r[1] == 'ERROR')
        f.write(f"<p>Total Reports: {total} | Passed: {passed} | Failed: {failed} | Errors: {errors}</p>")
        f.write("<h2>Validation Results</h2>")

        for file, status, details in results:
            base_name = os.path.splitext(file)[0]
            rules_file = os.path.join(folder, f"{base_name}_rules.json")
            with open(rules_file) as rf:
                rules = json.load(rf)

            top_color = '#c8e6c9' if status == 'PASS' else '#ffcdd2'
            f.write(f"<details><summary style='background-color:{top_color};padding:5px;'><strong>{file}</strong> - {status}</summary>")
            f.write("<table border='1' cellpadding='5'><tr><th>Rule</th><th>Target</th><th>Result</th><th>Message</th></tr>")

            all_rules = []
            for col in rules.get("required_columns", []):
                msg = "PASS" if all(f"Missing column: {col}" not in d for d in details) else f"Missing column: {col}"
                all_rules.append((f"Required Column: {col}", col, "PASS" if "Missing column" not in msg else "FAIL", msg))
            for row in rules.get("required_rows", []):
                msg = "PASS" if all(f"Missing row with Objective: {row}" not in d for d in details) else f"Missing row with Objective: {row}"
                all_rules.append((f"Required Row: {row}", row, "PASS" if "Missing row" not in msg else "FAIL", msg))
            for key, val in rules.get("cell_validations", {}).items():
                row_name, col_name = key.split(".")
                expected = str(val)
                fail_msg = f"Mismatch at {row_name}.{col_name}: expected {expected}"
                msg = "PASS" if not any(d.startswith(fail_msg) for d in details) else fail_msg
                all_rules.append(("Cell Value", key, "PASS" if msg == "PASS" else "FAIL", msg))
            for rule in rules.get("sql_validations", {}).get("queries", []):
                key = f"{rule['row']}.{rule['column']}"
                fail_msg = f"Mismatch from SQL at {key}: expected"
                found = any(d.startswith(fail_msg) for d in details)
                all_rules.append((f"SQL: {rule['query']}", key, "FAIL" if found else "PASS", fail_msg if found else "PASS"))

            for rule_type, rule_value, rule_status, msg in all_rules:
                color = "#c8e6c9" if rule_status == "PASS" else "#ffcdd2"
                f.write(f"<tr style='background-color:{color}'><td>{rule_type}</td><td>{rule_value}</td><td>{rule_status}</td><td>{msg}</td></tr>")

            f.write("</table></details>")

        f.write("</body></html>")

    print("\nValidation complete. Summary saved to validation_summary.html")
     # Write CSV output as well
    csv_rows = []
    for file, status, details in results:
        base_name = os.path.splitext(file)[0]
        rules_file = os.path.join(folder, f"{base_name}_rules.json")
        with open(rules_file) as rf:
            rules = json.load(rf)

        all_rules = []
        for col in rules.get("required_columns", []):
            msg = "PASS" if all(f"Missing column: {col}" not in d for d in details) else f"Missing column: {col}"
            all_rules.append((file, f"Required Column: {col}", col, "PASS" if "Missing column" not in msg else "FAIL", msg))
        for row in rules.get("required_rows", []):
            msg = "PASS" if all(f"Missing row with Objective: {row}" not in d for d in details) else f"Missing row with Objective: {row}"
            all_rules.append((file, f"Required Row: {row}", row, "PASS" if "Missing row" not in msg else "FAIL", msg))
        for key, val in rules.get("cell_validations", {}).items():
            row_name, col_name = key.split(".")
            expected = str(val)
            fail_msg = f"Mismatch at {row_name}.{col_name}: expected {expected}"
            msg = "PASS" if not any(d.startswith(fail_msg) for d in details) else fail_msg
            all_rules.append((file, "Cell Value", key, "PASS" if msg == "PASS" else "FAIL", msg))
        for rule in rules.get("sql_validations", {}).get("queries", []):
            key = f"{rule['row']}.{rule['column']}"
            fail_msg = f"Mismatch from SQL at {key}: expected"
            found = any(d.startswith(fail_msg) for d in details)
            all_rules.append((file, f"SQL: {rule['query']}", key, "FAIL" if found else "PASS", fail_msg if found else "PASS"))

        csv_rows.extend(all_rules)

    pd.DataFrame(csv_rows, columns=["File", "Rule Type", "Target", "Result", "Message"]).to_csv("validation_summary.csv", index=False)
    return results
def generate_mock_sqlite_db(path: str):
    """
    Creates a mock SQLite database with sample report_summary data.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS report_summary")
        cursor.execute("""
            CREATE TABLE report_summary (
                objective TEXT,
                plan_target INTEGER,
                loan_count INTEGER,
                upb INTEGER,
                dts_rural REAL,
                dts_percent REAL
            )
        """)
        cursor.executemany("""
            INSERT INTO report_summary VALUES (?, ?, ?, ?, ?, ?)
        """, [
            ("Shared Equity", 344, 343, 45, 0.15, 0.012),
            ("High needs rural population", None, 94567, 3456, 0.15, 0.005),
            ("Manufactured Homes Titled as Real Property", 6484, 6093, 8734, 0.2, 0.0)
        ])
        conn.commit()
    print(f"Mock SQLite database created at: {path}")

def validate_post_report_with_genai(
    post_filepath: str,
    db_path: str = None,
    required_columns: List[str] = [],
    required_rows: List[str] = [],
    cell_validations: List[str] = []
) -> List[Dict]:
    post_data = read_raw_excel(post_filepath) if post_filepath.endswith(".xlsx") else read_raw_html(post_filepath)
    if not post_data:
        return ["The post report is empty or unreadable."]

    # extract table data using genAI prompt
    post_df = extract_table_with_genai(post_data)

    db_sample = []
    if db_path and not os.path.exists(db_path):
        generate_mock_sqlite_db(db_path)
        
    if db_path and os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path) as conn:
                tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
                if not tables.empty:
                    first_table = tables.iloc[0, 0]
                    db_sample = pd.read_sql_query(f"SELECT * FROM {first_table}", conn).to_dict(orient='records')
        except Exception as e:
            db_sample = [f"Database error: {e}"]

    prompt = f"""
    You are a validation assistant. Analyze this post-run report data:

    Report Data:
    {post_df.to_json(orient='records', indent=2)}

    Tasks:
    1. Check for these required columns:
       {required_columns}
    2. Ensure the following rows exist based on 'Objective':
       {required_rows}
    3. Validate the following cell values:
       {cell_validations}
    4. Detect any other data anomalies.
    5. Optionally validate against DB sample if available:

    Database Sample:
    {json.dumps(db_sample, indent=2)}

    Return a JSON list where each entry contains:
    - 'type' (e.g. 'missing_column', 'row_error', 'value_mismatch')
    - 'column' (if applicable)
    - 'row_identifier'
    - 'description'
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    try:
        results = json.loads(content)
        if isinstance(results, list):
            html = "<html><head><title>Validation Results</title></head><body><h1>Validation Summary</h1><table border='1' cellpadding='5'><tr><th>Type</th><th>Column</th><th>Row Identifier</th><th>Description</th></tr>"
            for item in results:
                row = item if isinstance(item, dict) else {"type": "error", "description": str(item)}
                color = '#c8e6c9' if row.get('type') == 'info' else ('#ffcdd2' if row.get('type') == 'value_mismatch' else '#ffe0b2')
                html += f"<tr style='background-color:{color}'><td>{row.get('type', '')}</td><td>{row.get('column', '')}</td><td>{row.get('row_identifier', '')}</td><td>{row.get('description', '')}</td></tr>"
            html += "</table></body></html>"
            with open("post_report_validation_summary.html", "w", encoding="utf-8") as f:
                f.write(html)
            print("Validation summary saved to post_report_validation_summary.html")
            return results
        else:
            return ["Unexpected GPT output format."]
    except Exception as e:
        return [f"Error parsing GPT output: {e}\nRaw Output: {content}"]
