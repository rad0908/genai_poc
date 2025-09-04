import streamlit as st
import sqlite3
from openai import OpenAI
import pandas as pd
import os
import glob


api_key ="api_key"
# Set your OpenAI API key
client = OpenAI(api_key = api_key)
# DB_PATH = "data/validation.sqlite"
# DB_PATH = "data/mock_loan_db_large.sqlite"

# List available SQLite databases
# List valid SQLite databases in /data folder (with or without .sqlite extension)
data_folder = "./data"
db_files = []
for f in glob.glob(os.path.join(data_folder, "*")):
    if os.path.isfile(f) and not f.endswith(".py") and not f.endswith(".txt"):
        try:
            with sqlite3.connect(f) as conn:
                conn.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
            db_files.append(f)
        except Exception:
            pass

selected_db = st.selectbox("Select SQLite Database", db_files)

# Function to get schema for context
def get_schema(db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            schema = ""
            for (table,) in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                schema += f"\nTable: {table}\n"
                for col in columns:
                    schema += f"  {col[1]} ({col[2]})\n"
        return schema.strip()
    except Exception:
        return "Invalid SQLite database."

# Function to execute SQL and return results
def execute_sql(sql_query, db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(sql_query, conn)
        return df
    except Exception as e:
        return f"Error executing query: {e}"

# Streamlit UI
st.title("GenAI SQL Query Generator for Loan DB")

schema_description = get_schema(selected_db)

if st.button("View DB Schema"):
    st.subheader("Database Schema")
    st.text(schema_description)

user_prompt = st.text_area("Enter your query in natural language:", height=150)

if st.button("Generate SQL Query"):
    if user_prompt:
        with st.spinner("Generating SQL..."):
            system_prompt = f"""You are a SQL assistant for a loan management system database. Based on the schema below, generate a syntactically correct SQLite SQL query only (do not explain it).\n\nSchema:\n{schema_description}\n"""
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
            )
            sql_query = response.choices[0].message.content.strip()
            st.subheader("Generated SQL Query:")
            st.code(sql_query, language='sql')

            # Execute SQL
            st.subheader("Query Results")
            result = execute_sql(sql_query, selected_db)
            if isinstance(result, pd.DataFrame):
                st.dataframe(result)
            else:
                st.error(result)
    else:
        st.warning("Please enter a prompt before submitting.")

# Validation section
st.markdown("---")
st.header("🔍 Table Data Validation")
validation_rule = st.text_area("Describe your validation rule (e.g., 'Loan amount should be less than 50% of annual income')")

if st.button("Run Validation Rule"):
    if validation_rule:
        with st.spinner("Checking validation using GenAI..."):
            schema_description = get_schema(selected_db)  # Re-fetch schema for current DB
            system_prompt = f"""You are a validation assistant. Based on the database schema below and the user's validation rule, generate and execute the validation logic using SQLite. Return the result of the validation directly — either 'Validation Passed' or a table of rows that violate the rule.\n\nSchema:\n{schema_description}\n\nValidation Rule:\n{validation_rule}"""
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.2,
            )
            validation_result = response.choices[0].message.content.strip()
            st.subheader("Validation Result:")
            st.markdown(validation_result)
    else:
        st.warning("Please enter a validation rule.")

# to run - streamlit run .\src\sql_generator.py