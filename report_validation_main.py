import sys
from src.report_validation_engine import run_validations_on_folder
from src.report_validator.table_extractor_full import process_excel_folder

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python main.py <report_folder> <validation_rules.json>")
#         sys.exit(1)

#     report_folder = sys.argv[1]
#     rules_file = sys.argv[2]

run_validations_on_folder(r"data\\reports")

# process_excel_folder("data\\reports", "features\\output")
