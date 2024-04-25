import pandas as pd
import os
import time
import sys

from mm_calc import *
from modeloVan import *

class BulkExcelLoader:
    def __init__(self):
        # Determine the directory of the executable or script file
        if getattr(sys, 'frozen', False):
            basedir = os.path.dirname(sys.executable)
        else:
            basedir = os.path.dirname(__file__)
        
        # Initialize attributes
        self.folderName = None
        self.van_calc = None  # Optionally initialize a VanCalculator object
        self.param_path = None  # Placeholder for parameter path
        
            
    def load_vancalculator_parameters(self, param_path):
        # Preload the parameters for later use
        self.param_path = param_path
        
    def load_filesList_by_folderName(self, folderName):
        self.folderName = folderName
        self.all_filename_list = self.get_all_filenames_in_folder()
        self.valid_filename_list = [filename for filename in self.all_filename_list if filename.endswith('.xlsx')]
    
    def print_log_filtering(self):
        output = "===============\nFORMATO\n"
        for filename in self.all_filename_list:
            if filename.endswith('.xlsx'):
                output += f"{filename}: OK\n"
            else:
                output += f"{filename}: Descartado, formato inválido.\n"
        return output
        
    def get_preprocessed_filepath(self, filename):
        return f"{self.folderName}\\preprocessed\\preprocessed_mm_{filename}"
    
    def get_output_filepath(self, filename):
        # Ensure the correct extension for the output file
        filename = filename if filename.endswith('.xlsx') else f"{filename}.xlsx"
        return f"{self.folderName}\\output\\output_{filename}"

    def get_valid_filenames(self):
        return self.valid_filename_list
    
    def get_count_valid_and_total_filenames(self):
        return ( len(self.valid_filename_list), len(self.all_filename_list) )
        
    def get_all_filenames_in_folder(self):
        # Check if the folder name is set
        if not hasattr(self, 'folderName') or not self.folderName:
            raise ValueError ("Folder name is not set or empty.")

        # Check if the folder exists
        if not os.path.exists(self.folderName):
            raise ValueError (f"Folder '{self.folderName}' does not exist.")

        # List all files in the directory
        filenameList = [f for f in os.listdir(self.folderName) if os.path.isfile(os.path.join(self.folderName, f))]

        return filenameList
    
    def run_preprocessor_on_filename(self, filename):
        try:
            df = pd.read_excel(f"{self.folderName}\\{filename}", sheet_name=None)
            output = mm_calc_run_alg(df)
            
            # Ensure the output directory exists
            output_dir = os.path.dirname(self.get_preprocessed_filepath(filename))
            os.makedirs(output_dir, exist_ok=True)
        
            with pd.ExcelWriter(self.get_preprocessed_filepath(filename), engine='xlsxwriter') as writer:
                for sheet_name, df in output.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        except Exception as e:
            raise ValueError(f"run failed on filename {filename}: {e}")
            
    def run_algorithm_on_filename(self, filename):
       try:
           # Run the preprocessor first
           self.run_preprocessor_on_filename(filename)
           
           # Get the preprocessed file path
           preprocessed_filename = self.get_preprocessed_filepath(filename)

           # Initialize a VanCalculator on the preprocessed Excel file
           self.van_calc = VanCalculator()
           self.van_calc.load_df_from_excel(preprocessed_filename)

           # If parameters were preloaded, load them into the VanCalculator
           if self.param_path:
               self.van_calc.load_parameters(self.param_path)

           # Run the VanCalculator methods
           self.van_calc.run_all()

           # Ensure the output directory exists
           output_dir = os.path.dirname(self.get_output_filepath(filename))
           os.makedirs(output_dir, exist_ok=True)
           
           # Export the results to a new Excel file
           output_path = self.get_output_filepath(filename+".xlsx")
           self.van_calc.export_to_excel(output_path)

           print(f"Algorithm ran successfully on {filename}. Output saved to {output_path}.")

       except Exception as e:
           raise RuntimeError(f"Failed to run algorithm on {filename}: {str(e)}")
            
            