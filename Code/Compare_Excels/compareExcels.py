import pandas as pd

path_to_first_file = r'outputMM - PETORCA.xlsx'
path_to_second_file = r'check_petorca.xlsx'
# Load the Excel files
file1 = pd.read_excel(path_to_first_file, sheet_name=None)
file2 = pd.read_excel(path_to_second_file, sheet_name=None)

# Compare
are_equal = True
for sheet in file1:
    if sheet not in file2:
        are_equal = False
        break
    if not file1[sheet].equals(file2[sheet]):
        are_equal = False
        break

print("Files are equal." if are_equal else "Files are not equal")
