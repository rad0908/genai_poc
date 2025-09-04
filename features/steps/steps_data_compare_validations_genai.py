import openai
import pandas as pd
from behave import given, when, then
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import sqlite3
import datetime
import os

# Set OpenAI API key (ensure this is set correctly in your environment)
openai.api_key = "api_key"  # Or ensure the environment variable is set: OPENAI_API_KEY

# Load CSV dataset (source or target)
def load_csv(file_path):
    return pd.read_csv(file_path)

# Query the database (SQLite)
def query_db(db_name='mock_loans_50.db', table_name='target_table'):
    # Connect to SQLite database
    conn = sqlite3.connect(db_name)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Comparison Prompt: Compare CSV and Database Data
def generate_comparison_prompt(task, csv_data, db_data):
    comparison_prompt = PromptTemplate(
        input_variables=["task", "csv_preview", "db_preview"],
        template="""
        You are a data quality assistant. I have the following data from the CSV:
        {csv_preview}
        
        And the following data from the database:
        {db_preview}

        The task is: {task}

        Please do the following:
        1. Identify any discrepancies between the CSV and the database.
        2. List the rows/columns where the data does not match.
        3. compare row count
        4. Summarize the discrepancies clearly
        
        Return a summary of discrepancies and the rows/columns where the data does not match.
        """
    )

    # Set up LangChain with GPT-4 via ChatOpenAI
    chat_model = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=openai.api_key)
    chain = LLMChain(llm=chat_model, prompt=comparison_prompt)
    
    # Generate data previews (first 20 rows for example)
    csv_preview = csv_data.head(50).to_string(index=False)
    db_preview = db_data.head(50).to_string(index=False)
    
    # Run the comparison through LangChain
    return chain.run(task=task, csv_preview=csv_preview, db_preview=db_preview)

# Validation Rules Prompt: Validate CSV and Database Data
def generate_validation_prompt(validation_rules, csv_data, db_data):
    validation_prompt = PromptTemplate( 
    input_variables=["validation_rules", "csv_preview", "db_preview"],
    template="""
You are a data quality assistant.

Your task is to strictly apply the following validation rules:
{validation_rules}

Here is a sample of the CSV data:
{csv_preview}

Here is a sample of the Database data:
{db_preview}

Instructions:
- Apply each rule independently on both the CSV and the DB datasets.
- DO NOT compare CSV values against DB values.
- DO NOT apply rules to any column that is not explicitly mentioned.
- DO NOT show data descrepencies between CSV and DB
- For each violation, specify:
  - The dataset (CSV or DB)
  - The rule that was violated
  - Column name
  - Row index or identifier
  - The incorrect value and the expected format/value

Return a structured summary of:
1. Each validation rule and whether it passed or failed per dataset.
2. Details of rows that failed, if any.
"""
)

    # Set up LangChain with GPT-4 via ChatOpenAI
    chat_model = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=openai.api_key)
    chain = LLMChain(llm=chat_model, prompt=validation_prompt)
    
    # Generate data previews (first 5 rows for example)
    # csv_preview = csv_data.head(20).to_string(index=False)
    # db_preview = db_data.head(20).to_string(index=False)
    
    csv_preview = csv_data.head(20).to_json(orient='records', indent=2)
    db_preview = db_data.head(20).to_json(orient='records', indent=2)

    # Run the validation through LangChain
    return chain.run(validation_rules=validation_rules, csv_preview=csv_preview, db_preview=db_preview)



def generate_report_prompt(compare_task, validation_task, comparison_results, validation_results):
    report_prompt = PromptTemplate(
        input_variables=["compare_task", "validation_task", "comparison_results", "validation_results"],
        template="""
You are a data quality assistant. I have the following comparison results between the CSV and the database:
{comparison_results}

And the following validation results:
{validation_results}

The comparison task is: {compare_task}
The validation task is: {validation_task}

Please generate a clean and professional summary report with the following sections:

1. **Count of source and target 
    - show source and target row count. example csv count = 1234, db count = 345
    - Highlight if there is a mismatch 
2. **Column-wise and Row-wise Mismatch Summary Table**  
   - Show each column name.
   - Include total mismatches per column.
   - Highlight specific rows (with row number or key) where mismatches occurred. Limit number of mis matched rows to 5 
3. **Detailed mismatches table
   - Show top 5 mis matches for each column example column = ID, csv value = 1, db value = 2

4. **Validation Rules Table**  
   - List each validation rule checked.
   - Show status as Pass or Fail.
   - Optionally include brief details for failures (e.g., which row or value failed).

5. **Discrepancy and Validation Summary**  
   - Show validation results
   - Identify patterns (e.g., frequent nulls in a specific column).
   - Recommend fixes or improvements.

Present all tables in markdown format or a layout that can be easily converted into an HTML or PDF report.
"""
    )

    # Set up LangChain with GPT-4 via ChatOpenAI
    chat_model = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=openai.api_key)
    chain = LLMChain(llm=chat_model, prompt=report_prompt)

    # Run the report generation through LangChain
    return chain.run(
        compare_task=compare_task,
        validation_task=validation_task,
        comparison_results=comparison_results,
        validation_results=validation_results
    )



def save_report_to_html(compare_task, validation_task, report_text, output_dir="outputs"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"genai_data_report_{timestamp}.html")

    # Basic HTML template with task and report sections
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Data Comparison & Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; background-color: #f9f9f9; }}
            h1 {{ color: blue; }}
            h2 {{ color: red; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            pre {{ background-color: #d8f5d8; padding: 10px; border: 1px solid #ddd; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>Data Comparison & Validation Report</h1>

        <h2>Tasks</h2>
        <p><strong>Comparison Task:</strong> {compare_task}</p>
        <p><strong>Validation Task:</strong> {validation_task}</p>

        <h2>Report</h2>
        <pre>{report_text}</pre>
    </body>
    </html>
    """


    # Write to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report saved to: {file_path}")
    return file_path


@given('I have the CSV dataset at "{dataset}"')
def step_given_csv(context, dataset):
    context.csv_data = load_csv(dataset)
    context.csv_preview = context.csv_data.head().to_string(index=False)

@given('I have the database at "{dataset}"')
def step_given_target_csv(context, dataset):
    # Assume dataset is the SQLite file path
    context.target_db = dataset  # The SQLite file
    context.db_data = query_db(db_name=context.target_db, table_name='target_table')
    context.target_preview = context.db_data.head().to_string(index=False)

@when('I compare the datasets with task "{task}"')
def step_when_compare_datasets(context, task):
    # Store the task in context to use in the prompts
    context.comparetask = task
    context.validation_results = ""
    # Generate the comparison results using the comparison prompt
    context.comparison_results = generate_comparison_prompt(context.comparetask, context.csv_data, context.db_data)

@when('I run validations with task "{task}"')
def step_when_run_additional_validations(context, task):
    # Store the task in context if needed
    context.validationtask  = task
    
    # validation_rules = "loan_amount should be 4 decimals, cycle_date should be in mm/dd/yyyy format"  # Example validation rules
    
    # Generate the validation results using the validation prompt
    context.validation_results = generate_validation_prompt(context.validationtask , context.csv_data, context.db_data)
    
    # Ensure the validation results are stored properly
    if context.validation_results is None:
        raise AssertionError("Validation results are missing.")

@then('I should generate a GenAI validation report')
def step_then_generate_report(context):
    # Ensure both tasks are present in the context
    compare_task = context.comparetask if hasattr(context, 'comparetask') else "No comparison task provided"
    validation_task = context.validationtask if hasattr(context, 'validationtask') else "No validation task provided"
    
    # Ensure validation results are present before generating the report
    # if not hasattr(context, 'validation_results') or context.validation_results is None:
    #     raise AssertionError("Validation results are missing, cannot generate report.")
    
    # Generate the report using the report generation prompt
    report = generate_report_prompt(compare_task, validation_task, context.comparison_results, context.validation_results)
    save_report_to_html(compare_task, validation_task, report)
    # Optionally: Save the report to a file or display it
    create_detailed_report(report, context.comparison_results, context.validation_results, compare_task+". validation checks: " +validation_task)



# When Step to compare selected columns
@when('I compare the datasets for columns "{columns}" with task "{task}"')
def step_when_compare_columns(context, columns, task):
    context.comparetask = task
    columns_list = columns.split(", ")
    db_data_filtered = context.db_data[columns_list]
    csv_data_filtered = context.csv_data[columns_list]
    comparison_results = generate_comparison_prompt(task, csv_data_filtered, db_data_filtered)
    context.comparison_results = comparison_results
    context.validation_results = ""


# Then Step to check the comparison results
@then('I should see the comparison results')
def step_then_check_results(context):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define the file name
    filename = f"output_comparisonresults_{timestamp}.txt"
    
    assert context.comparison_results is not None
    # Open the file in write mode and write the comparison results
    with open(filename, 'w') as file:
        file.write(f"Comparison Results:\n{context.comparison_results}")
    # assert context.comparison_results is not None
    # print(f"Comparison Results: {context.comparison_results}")



@then('I should see the comparison results for the top mismatches')
def step_impl(context):
    # Ensure that comparison_results exist in the context
    if hasattr(context, 'comparison_results') and context.comparison_results is not None:
        # Assuming the comparison_results contain a summary of the discrepancies
        # If comparison_results is a table, we print it directly
        # Here, context.comparison_results could be a string or a structured result from GPT-4.
        print("\nTop Mismatches from the Comparison:\n")
        
        # If the comparison results are in string form
        if isinstance(context.comparison_results, str):
            print(context.comparison_results)
        
        # If the comparison results are in a structured format like a DataFrame
        elif isinstance(context.comparison_results, pd.DataFrame):
            # Display the top 5 rows of the mismatches (if it's a DataFrame)
            top_mismatches = context.comparison_results.head(5)
            print(top_mismatches)
        else:
            print("Comparison results format is unrecognized.")
    else:
        print("No comparison results found.")
    # No need to raise NotImplementedError here anymore



# Helper Function: Generate HTML Report with Timestamp
def create_detailed_report(genai_report, comparison_results, validation_results, task, filename_prefix="validation_report"):
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.html"
    filename2 = f"{filename_prefix}_{timestamp}.txt"
    # Generate a more presentable HTML output
    comparison_table = generate_html_table(comparison_results, "Comparison Results")
    validation_table = generate_html_table(validation_results, "Validation Results")
    
    # Task/Pprompt header
    task_header = f"<h3>Task/Prompt:</h3><p>{task}</p><br>"

    html_report = f"""
    <html>
    <head>
        <title>GenAI Validation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                margin: 20px;
                color: #333;
            }}
            h2 {{
                color: #4CAF50;
                text-align: center;
            }}
            h3 {{
                color: #4CAF50;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
                font-size: 14px;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                border-radius: 8px;
            }}
            th, td {{
                padding: 15px;
                text-align: left;
                border: 1px solid #ddd;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
                text-transform: uppercase;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #ddd;
            }}
            .mismatch {{
                background-color: #ffcccc;
                color: red;
                font-weight: bold;
            }}
            .valid {{
                background-color: #ccffcc;
                color: green;
            }}
            .warning {{
                background-color: #ffffcc;
                color: orange;
            }}
            .center {{
                text-align: center;
            }}
            .table-container {{
                margin: 0 auto;
                width: 80%;
            }}
            .table-title {{
                font-size: 18px;
                color: #4CAF50;
                margin-bottom: 10px;
                text-align: center;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h2>GenAI Comparison and Validation Report</h2>
        
        <!-- Task/Pprompt section -->
        {task_header}

        <div class="table-container">
            {comparison_table}
        </div>
        
        <div class="table-container">
            {validation_table}
        </div>
        
    </body>
    </html>
    """
    
    # Write the HTML report to a file
    with open(filename, 'w') as f:
        f.write(html_report)
    with open(filename2, 'w') as f:
        f.write(genai_report)   
    print(f"Report generated: {filename}")
    return filename  # Return the file name for later use if necessary

# Helper Function: Convert Results into HTML Table
def generate_html_table(results, title="Table"):
    """
    Helper function to convert the results (either comparison or validation) into an HTML table.
    Adds a title for each table and improves the presentation.
    """
    table_html = f"<div class='table-title'>{title}</div><table>\n<thead>\n<tr>"
    
    # Check if results are in DataFrame format (structured table)
    if isinstance(results, pd.DataFrame):
        # Add table headers
        table_html += ''.join(f"<th>{col}</th>" for col in results.columns)
        table_html += "</tr>\n</thead>\n<tbody>\n"
        
        # Add table rows
        for _, row in results.iterrows():
            table_html += "<tr>"
            for val in row:
                # Add cells with the data
                table_html += f"<td>{val}</td>"
            table_html += "</tr>\n"
    
    # If results are in string format (e.g., mismatches summary from GPT-4)
    elif isinstance(results, str):
        # Corrected: Replace newlines with <br> tags in the string output
        results = results.replace("\n", "<br>")
        table_html += f"<tr><td>{results}</td></tr>"
    
    table_html += "</tbody>\n</table>\n"
    return table_html