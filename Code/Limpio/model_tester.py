import pandas as pd
from model import BulkExcelLoader

#folder_path es directorio porque carga todos los excels que encuentra
folder_path = r"C:\GitHub\6020.Inecon_ModeloVanMedicion\6020.Inecon_ModeloVanMedicion\Code\Limpio\Excels"
# folder_path = r"C:\Users\toman\Downloads\carpeta"

#------ Load the model ----- #
model = BulkExcelLoader()
model.load_filesList_by_folderName(folder_path)

#-------- Run the algorithm --------#

valid_count, total_count = model.get_count_valid_and_total_filenames()
filename_list = model.get_valid_filenames()

# Loop through each valid Excel file and run the algorithm
for i, filename in enumerate(filename_list):
    print(f"Processing file {i + 1}/{valid_count}: {filename}")
    
    try:
        model.run_algo_allScenarios(filename)
        print(f"Successfully processed: {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

print("Completed processing all valid Excel files.")