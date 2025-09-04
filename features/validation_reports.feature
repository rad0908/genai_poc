Feature: Report Validation

  @reports
  Scenario: Validate all structured reports in the reports folder
    Given I have a folder of reports with rule files
    When I run the validation engine
    Then all reports should pass validations or report appropriate errors
    
    @report_single
    Scenario: Validate post report with structured prompt using explicit validation rules
    Given the post report "data/reports/sample_sas_report.xlsx" and database "data/validation.sqlite"
    When I run GenAI-powered validations:
    | Prompt Type       | Prompt Content                                                                                  |
    | required_columns  | UPB($1M), Loan Count, % Units DTS Rural, DTS%, Error                                                  |
    | required_rows     | Shared Equity, High needs rural population, Manufactured Homes Titled as Real Property          |
    | cell_validations  | Shared Equity.Plan Target = 344, Manufactured Homes Titled as Real Property.UPB($1M) = 8734     |
    | db_check          | check if upb and loan count from db matches with report data for all objectives                               |   


    @report_dbcheck
    Scenario: Validate post report with structured prompt using explicit validation rules
    Given the post report "data/reports/full_mortgage_report.xlsx" and database "data/full_loan_data.db"
    When I run GenAI-powered validations:
    | Prompt Type       | Prompt Content                                                                                  |
    | required_columns  | Target Loan Count, Target UPB($M), Variance(Count), Actual Loan Count, Objective,  Category                                                  |
    | required_rows     | Affordable Housing, Rural Lending          |
    | cell_validations  | Affordable Housing.Shared Equity.Target Loan Count = 500    |
    | db_check          | Get count of loans for each objective by joining loans and objectives; compare it with 'Actual Loan Count' in the report |

