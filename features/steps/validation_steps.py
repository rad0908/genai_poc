from behave import given, when, then
# from src.report_validation_engine import run_validations_on_folder
# from src.report_validation_engine import validate_post_report_with_genai
# from src.report_validation_genai import validate_post_report_with_genai
import os
import json
import pandas as pd
import openai
import sqlite3
from typing import List, Dict
from bs4 import BeautifulSoup
import re

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

def get_db_schema_as_dict(db_path: str) -> Dict[str, List[str]]:
    schema = {}
    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        for table in tables["name"]:
            cols = pd.read_sql(f"PRAGMA table_info({table})", conn)["name"].tolist()
            schema[table] = cols
    return schema

def generate_sql_query_with_genai(schema: Dict[str, List[str]], db_check_prompt: str) -> str:
    sql_prompt = f"""
    You are a SQL assistant. Based on the following database schema:

    {json.dumps(schema, indent=2)}

    Generate a SQL query to fulfill this task:
    "{db_check_prompt}"

    Only return the SQL query string. Do not explain or format it as code.
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": sql_prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def validate_post_report_with_genai(
    post_filepath: str,
    db_path: str,
    required_columns: List[str] = [],
    required_rows: List[str] = [],
    cell_validations: List[str] = [],
    db_check: List[str] = []
) -> List[Dict]:
    post_data = read_raw_excel(post_filepath) if post_filepath.endswith(".xlsx") else read_raw_html(post_filepath)
    if not post_data:
        return ["The post report is empty or unreadable."]

    # extract table data using genAI prompt
    post_df = extract_table_with_genai(post_data)

    db_sample = []
    db_schema = {}
    executed_sql = ""
    if db_path and os.path.exists(db_path) and db_check:
        try:
            db_schema = get_db_schema_as_dict(db_path)
            db_check_task = " ".join(db_check)
            executed_sql = generate_sql_query_with_genai(db_schema, db_check_task)

            with sqlite3.connect(db_path) as conn:
                db_sample_df = pd.read_sql_query(executed_sql, conn)
                db_sample = db_sample_df.to_dict(orient='records')
        except Exception as e:
            db_sample = [f"Database error: {e}"]

    prompt = f"""
    You are a validation assistant. Analyze this post-run report data.

    Report Data:
    {post_df.to_json(orient='records', indent=2)}

    Tasks:
    1. Check for these required columns:
       {required_columns}
    2. Ensure the following rows exist based on 'Objective':
       {required_rows}
    3. Validate the following cell values:
       {cell_validations}
    4. Compare the following against database values (do NOT compare target counts, only actuals):
       {db_check}
       SQL Query Used:
       {executed_sql}
    Database Sample:
    {json.dumps(db_sample, indent=2)}

    Output:
    Return a JSON list where each entry contains:
    - "type" (e.g., "required_columns", "required_rows", "cell_validations", "db_check", "db_mismatch", "anomaly")
    - "status": "Pass" or "Fail"
    - "column" (if applicable)
    - "query" (if applicable)
    - "row_identifier"
    - "description": show source and target values
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=1
    )

    content = response.choices[0].message.content.strip()
    with open("Validation_genAI_report.txt", "w", encoding="utf-8") as f:
        f.write(content)
    try:
        # Extract just the JSON list from the response (first [ ... ] block)
        match = re.search(r"\[\s*{.*?}\s*\]", content, re.DOTALL)
        if not match:
            raise ValueError("No JSON list found in LLM output.")
    
        json_str = match.group(0)
        results = json.loads(json_str)
        # results = json.loads(content)
        if isinstance(results, list):
            html = "<html><head><title>Validation Results</title></head><body><h1>Validation Summary</h1><table border='1' cellpadding='5'><tr><th>Type</th><th>Status</th><th>Column</th><th>Row Identifier</th><th>Description</th></tr>"
            for item in results:
                row = item if isinstance(item, dict) else {"type": "error", "description": str(item)}
                color = '#c8e6c9' if row.get('status') == 'Pass' else ('#ffe0b2' if row.get('type') == 'anomaly' else '#ffcdd2')
                html += f"<tr style='background-color:{color}'><td>{row.get('type', '')}</td><td>{row.get('status', '')}</td><td>{row.get('column', '')}</td><td>{row.get('row_identifier', '')}</td><td>{row.get('description', '')}</td></tr>"
            html += "</table></body></html>"
            with open("post_report_validation_summary.html", "w", encoding="utf-8") as f:
                f.write(html)
            print("Validation summary saved to post_report_validation_summary.html")
            return results
        else:
            print("Unexpected GPT output format.")
            return None
    except Exception as e:
        print(f"Error parsing GPT output: {e}\nRaw Output: {content}")
        return None



@given('the post report "{report_path}" and database "{db_path}"')
def step_given_report_and_db(context, report_path, db_path):
    context.report_path = report_path
    context.db_path = db_path

@when("I run GenAI-powered validations")
def step_when_run_genai_validations(context):
    context.required_columns = []
    context.required_rows = []
    context.cell_validations = []
    for row in context.table:
        if row["Prompt Type"] == "required_columns":
            context.required_columns = [col.strip() for col in row["Prompt Content"].split(",")]
        elif row["Prompt Type"] == "required_rows":
            context.required_rows = [row_val.strip() for row_val in row["Prompt Content"].split(",")]
        elif row["Prompt Type"] == "cell_validations":
            context.cell_validations = [cv.strip() for cv in row["Prompt Content"].split(",")]
        elif row["Prompt Type"] == "db_check":
            context.db_check = [db.strip() for db in row["Prompt Content"].split(",")]
    context.results = validate_post_report_with_genai(
        context.report_path,
        context.db_path,
        context.required_columns,
        context.required_rows,
        context.cell_validations,
        context.db_check
    )
    if context.results == None:
        assert False

@then("I should see validation results for")
def step_then_check_prompts(context):
    assert hasattr(context, "results"), "No validation results found."
    expected_prompts = {
        "required_columns": ["UPB($1M)", "Loan Count", "% Units DTS Rural", "DTS%"],
        "required_rows": [
            "Shared Equity",
            "High needs rural population",
            "Manufactured Homes Titled as Real Property"
        ],
        "cell_validations": [
            "Shared Equity.Plan Target = 344",
            "Manufactured Homes Titled as Real Property.UPB($1M) = 8734"
        ]
    }

    found_keys = set()
    for row in context.results:
        description = row.get("description", "")
        for prompt_type, values in expected_prompts.items():
            if any(v in description for v in values):
                found_keys.add(prompt_type)

    for key in expected_prompts.keys():
        assert key in found_keys, f"Missing validation results for: {key}"


@given("I have a folder of reports with rule files")
def step_given_folder_with_reports(context):
    context.report_folder = "./data/reports"

@when("I run the validation engine")
def step_when_run_validation(context):
    context.results = run_validations_on_folder(context.report_folder)

@then("all reports should pass validations or report appropriate errors")
def step_then_check_results(context):
    assert context.results, "No results were returned."
    failing = []
    for file, status, errors in context.results:
        print(f"{file} => {status}")
        if status == "FAIL" or status == "ERROR":
            failing.append((file, status, errors))

    if failing:
        for file, status, errors in failing:
            print(f"\nFAILED: {file} - {status}")
            for e in errors:
                print(f"  - {e}")
        raise AssertionError(f"{len(failing)} report(s) failed validation. See log above.")
