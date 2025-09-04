# File: features/data_compare_enhanced.feature
Feature: GenAI-driven data comparison

  @comparedata
  Scenario: Compare selected columns using a GenAI prompt
    Given the source CSV file is "data/source.csv"
    And the target database table is "target_table" in "data/target.db"
    And the comparison prompt is "Compare only loan_amount and cycle_date columns and list top 5 mismatches"
    When I run the GenAI-based enhanced comparison
    Then I should see an enhanced report generated

  @genAI_compare1
  Scenario: Compare two datasets and run validations
    Given I have the CSV dataset at "data/source.csv"
    And I have the database at "data/target.db"
    When I compare the datasets with task "Compare the CSV data with the database query result to check for discrepancies and run validations"
    Then I should see the comparison results
    And I should generate a GenAI validation report

  @genAI_compare1
  Scenario: Compare only loan_amount and cycle_date columns and list top 5 mismatches per column and run validations
    Given I have the CSV dataset at "data/source.csv"
    And I have the database at "data/target.db"
    When I compare the datasets for columns "loan_amount, cycle_date" with task "Compare the selected columns (loan_amount, cycle_date) from the CSV dataset and database, and list top 5 mismatches per column"
    Then I should see the comparison results for the top mismatches
    And I should generate a GenAI validation report

  @genAI_compare1
  Scenario: Compare two datasets and run pre defined validations
    Given I have the CSV dataset at "data/source.csv"
    And I have the database at "data/target.db"
    When I compare the datasets with task "Compare the CSV data all columns except column cycle_date with the database query result to check for discrepancies"
    And I run validations with task "loan_amount should have only 2 decimal places, cycle_date must be in 'yyyy-mm-dd' format e.g., 2024-02-01.origination_date must be in yyyy-mm-dd format (e.g., 2024-02-01).Run these validations on DB and csv"
    Then I should see the comparison results
    And I should generate a GenAI validation report

  @genAI_compare
  Scenario: Compare two datasets and run pre defined validations
    Given I have the CSV dataset at "data/mock_loans_50.csv"
    And I have the database at "data/mock_loans_50.db"
    When I compare the datasets with task "Compare the CSV data all columns except column cycle_date with the database query result to check for discrepancies and run validations"
    And I run validations with task "loan_amount should have only 2 decimal places, cycle_date must be in 'yyyy-mm-dd' format e.g., 2024-02-01.origination_date must be in yyyy-mm-dd format (e.g., 2024-02-01).Run these validations on DB and csv"
    Then I should see the comparison results
    And I should generate a GenAI validation report
