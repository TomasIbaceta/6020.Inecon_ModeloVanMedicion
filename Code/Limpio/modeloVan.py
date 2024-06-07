import pandas as pd
import numpy as np
import numpy_financial as npf

from config import AppConfig

config = AppConfig()

def validate_numeric_columns(df, column_names):
    for col_name in column_names:
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        df[col_name].fillna(0, inplace=True)
    return df

def check_rollover_conditions(row, i, inversion_col, valor_res_col):
    if i == 14 or i == 15:
        condition = (row['DIAMETRO_MEDIDOR'] >= 38) and (row['Curva proy'] == 'ULTRA')
        if condition:
            pass
            # print(f'No rollover applied for Instalacion {row["INSTALACION"]}: Diametro: {row["DIAMETRO_MEDIDOR"]}, curva_proy: {row["Curva proy"]}')
    return row

class VanCalculator:
    def __init__(self):
        # Start with empty attributes
        self.dfs = {}
        self.excel_path = None
        self.params = None
        self.scenario_params = None

    def load_df_from_excel(self, excel_path):
        # Load the main DataFrame(s) from the specified Excel path
        self.excel_path = excel_path
        self.dfs = pd.read_excel(excel_path, sheet_name=None)  # Dictionary of DataFrames
    
    def load_scenario_params(self):
        params_df=self.dfs["MANEJO DE ESCENARIOS"]
        self.scenario_params={
            "s1_van_percent": None,
            "s2_capex_percent": None,
            "s3_capex_percent": None,
            "s4_capex_flat": None,
            "s5_TIR": None,
            }
        
        scenario_mapping = {
            1: "s1_van_percent",
            2: "s2_capex_percent",
            3: "s3_capex_percent",
            4: "s4_capex_flat",
            5: "s5_TIR"
        }
        
        # Populate the dictionary with the values from the DataFrame
        for _, row in params_df.iterrows():
            scenario_number = row["Escenario"]
            value = row["Valor"]
            if scenario_number in scenario_mapping:
                key = scenario_mapping[scenario_number]
                if "%" in str(value):
                    value = float(value.strip('%')) / 100
                else:
                    value = float(str(value))
                self.scenario_params[key] = value
        
    def get_global_params(self):
        return self.params
    
    def get_global_params_default(self):
        return config.global_params_default
    
    def set_global_params_from_df(self):
        try:
            data = self.dfs["PARAMETROS GLOBALES"]
            self.params = pd.Series(data['Valor'].values, index=data['Nombre']).to_dict()
        except:
            self.params=self.get_default_params()
            
    def set_global_params_from_dict(self, scenario: dict):
        if not scenario:
            raise ValueError("The provided scenario is empty.")
        
        self.params = scenario
        
            
    def run_all(self):
        if not self.params:
            raise ValueError("Can't run VanCalculator without parameters.")
        
        self.get_flujo_diameter_table()
        self.merge_tarifa()
        self.calculate_vsub()
        self.calculate_vsub_proy()
        self.calculate_ingresos()
        self.calculate_C_E()
        self.calculate_inversion()
        self.calculate_depr()
        self.calculate_valor_restante()
        self.calculate_base_A()
        self.calculate_base_B()
        self.calculate_impuesto( tax_rate=self.params['Impuesto'] )
        self.calculate_flujo()
        self.calculate_van( rate=self.params['Tarifa'])
        
        self.load_scenario_params()
        self.run_scenario_1(van_value = self.scenario_params['s1_van_percent'])
        self.run_scenario_2(capex_ratio = self.scenario_params['s2_capex_percent'])
        self.run_scenario_3(capex_ratio = self.scenario_params['s3_capex_percent'])
        self.run_scenario_4(capex_max = self.scenario_params['s4_capex_flat'])
        self.run_scenario_5(tasa = self.params['Tarifa'])
        self.general_cleanup()
        
    # Assuming 'datos_df' has been loaded from the 'Datos' sheet
    def merge_tarifa(self, main_sheet_name="BBDD - Error Actual", datos_sheet_name='Datos_Tarifas'):
        main_df = self.dfs[main_sheet_name]
        datos_df = self.dfs[datos_sheet_name]
        
        main_localidad_col = 'LOCALIDAD'
        main_tarifa_col = 'Tarifa'
        datos_tarifa_col = 'Tarifa ($)'
        datos_localidad_col = 'Localidad'
        
        # Merge to get the 'tarifa' based on 'LOCALIDAD'
        merged_df = main_df.merge(datos_df[[datos_localidad_col, datos_tarifa_col]],
                                                  left_on  = main_localidad_col, 
                                                  right_on = datos_localidad_col,
                                                  how='left').drop(columns=[datos_localidad_col])
    
        # Rename 'Tarifa ($)' to 'Tarifa'
        merged_df.rename(columns={'Tarifa ($)': 'Tarifa'}, inplace=True)
        
        # Update the DataFrame in the dictionary with the new merged DataFrame
        self.dfs[main_sheet_name] = merged_df

    def calculate_vsub(self, sheet_name="BBDD - Error Actual"):
        df = self.dfs[sheet_name]
        # Initial V_sub 1 calculation as before
        consumo_promedio = df["CONSUMO PROMEDIO"]
        df["V_sub_1"] = (-1 * consumo_promedio * df["Error"] / (1 + df["Error"]))
        # df["V_sub_1"] = (-1 * consumo_promedio * df["Error_1"] / (1 + df["Error_1"]))
    
        # Calculate V_sub 2 to V_sub 15
        for i in range(2, 16):  # 2 to 15 inclusive
            error_column = f"Error_{i-1}"  # Assumes Error_1 for V_sub 2, Error_2 for V_sub 3, and so on
            if error_column in df.columns:
                df[f"V_sub_{i}"] = -1 * (consumo_promedio + df["V_sub_1"]) * df[error_column]
            else:
                # Handle the case where an expected Error_x column is missing
                pass
                # print(f"Warning: Missing {error_column} in dataframe. Cannot calculate V_sub {i}.")

        self.dfs[sheet_name] = df
        
    def calculate_vsub_proy(self, sheet_name="BBDD - Error Actual"):
        df = self.dfs[sheet_name]
        consumoPromedioCol = 'CONSUMO PROMEDIO'
        curvaProyectadaCol = 'Curva proy' 
        curvaCol = 'Clase corregida' 
        grupoCol = 'Grupo'
        antiguedadCol = 'Antiguedad ajustada' 
        
        for x in range(1, 16):
            error_column = f'Error_{x-1}.5'  # Error_x.5 column for each x
            v_sub_x = f'V_sub_{x}'  # V_sub(x) column for each x
            v_sub_proy = f'V_sub_proy_{x}'  # V_sub_proy(x) result column for each x
            v_sub_proy_dadovuelta = f'V_sub_{x-14}'
            

            # Apply the logic
            df[v_sub_proy] = df.apply(
                lambda row: row[v_sub_x] if ( row[curvaProyectadaCol] == row[grupoCol] and row[antiguedadCol] == 0.5 ) 
                else -row[consumoPromedioCol] * row[error_column] / (1 + row[error_column]),
                axis=1
            )
            
            #caso especial dado vuelta despues de los 14
            df[v_sub_proy] = df.apply(
                lambda row: row[v_sub_proy_dadovuelta] if ( row[curvaProyectadaCol] == row[grupoCol] and row[antiguedadCol] == 0.5 and x==15) 
                else row[v_sub_proy],
                axis=1
            )
            
            # Handling errors
            df[v_sub_proy] = pd.to_numeric(df[v_sub_proy], errors='coerce').fillna('')
            
        self.dfs[sheet_name] = df
        
    def calculate_ingresos(self, sheet_name="BBDD - Error Actual"):
        df = self.dfs[sheet_name]
        
        tarifa_col = 'Tarifa'
        df[tarifa_col] = pd.to_numeric(df[tarifa_col], errors='coerce')
        
        for x in range(1, 16):
            v_sub_x = f'V_sub_{x}'  # V_sub(x) column for each x
            v_sub_proy_x = f'V_sub_proy_{x}'  # V_sub_proy(x) column for each x
            ingresos_x = f'Ingresos_{x}'  # Ingresos(x) result column for each x
            
            df[v_sub_x] = pd.to_numeric(df[v_sub_x], errors='coerce')
            df[v_sub_proy_x] = pd.to_numeric(df[v_sub_proy_x], errors='coerce')
    
            # Apply the formula: (Vsubx - Vsubproyx) * tarifa * 12
            df[ingresos_x] = (df[v_sub_x] - df[v_sub_proy_x]) * df[tarifa_col] * 12
    
            # Handling errors, setting to empty string where the operation fails
            df[ingresos_x] = pd.to_numeric(df[ingresos_x], errors='coerce').fillna('')
    
        # Update the dataframe in our dictionary
        self.dfs[sheet_name] = df
           
    def retrieve_elasticity(self, flujo_sheet_name='FLUJO E1'):
        flujo_df = self.dfs[flujo_sheet_name]
        # Find the row index where 'Elasticidad' is located
        row_idx = flujo_df.index[flujo_df.isin(['Elasticidad']).any(axis=1)][0]
        
        # Find the column index in that row for 'Elasticidad'
        col_idx = flujo_df.columns.get_loc(flujo_df.loc[row_idx][flujo_df.loc[row_idx] == 'Elasticidad'].index[0])
        
        # Retrieve the value to the right of 'Elasticidad', which is one column over
        elasticidad_value = flujo_df.iloc[row_idx, col_idx + 1]
        return elasticidad_value
        
    def calculate_C_E(self, main_sheet_name="BBDD - Error Actual"):
        main_df = self.dfs[main_sheet_name]
        elasticity = self.retrieve_elasticity()
        
        consumo_promedio_col = 'CONSUMO PROMEDIO'
        
        for x in range(1, 16):
            v_sub_col = f'V_sub_{x}'
            v_sub_proy_col = f'V_sub_proy_{x}'
            c_e_col = f'C_E_{x}'
            tarifa_col = 'Tarifa'  # Make sure this column name matches after merge_tarifa
            
            # Apply the Excel formula logic with additional check to prevent division by zero
            main_df[c_e_col] = main_df.apply(
                lambda row: ((row[tarifa_col] - (1 + ((row[v_sub_col] - row[v_sub_proy_col]) / row[consumo_promedio_col]) * elasticity) * row[tarifa_col]) * row[consumo_promedio_col] * -1
                             if pd.notnull(row[tarifa_col]) and pd.notnull(row[v_sub_col]) and pd.notnull(row[v_sub_proy_col]) and row[consumo_promedio_col] != 0
                             else None), axis=1)
            
        self.dfs[main_sheet_name] = main_df
    
    def calculate_inversion(self, main_sheet_name="BBDD - Error Actual"):
        main_df = self.dfs[main_sheet_name]
        flujo_df = self.flujo_diameter_table
        
        diametro_col = 'DIAMETRO_MEDIDOR'
        ponderado_col = 'Ponderado'
        inversion_col = 'Inversion'
        
        # Retrieve the UF value
        try:
            UF_value = float(self.retrieve_UF())
        except (ValueError, TypeError):
            UF_value = None  # Set to None to indicate an invalid UF value
    
        # Define a local function to lookup the Ponderado value
        def lookup_ponderado(diameter):
            try:
                ponderado_value = flujo_df.loc[diameter, ponderado_col]
                return float(ponderado_value)  # Convert to float to ensure it's numeric
            except (KeyError, ValueError, TypeError):
                return None
    
        # Calculate the 'Inversion' for each 'DIAMETRO' in the main DataFrame
        main_df[inversion_col] = main_df[diametro_col].apply(
            lambda x: -1 * lookup_ponderado(x) * UF_value if lookup_ponderado(x) is not None and UF_value is not None else ""
        )
    
        # Update the DataFrame in our dictionary
        self.dfs[main_sheet_name] = main_df


    def get_flujo_diameter_table(self, flujo_sheet_name='FLUJO E1'):
        '''
        The table needs the column with the headers to be on the excel 'C' column,
        or the pandas 2 row(it's 0-indexing).'
        '''
        first_data = 'SI'
        last_data = 'Vida útil técnica'
        
        raw_data = self.dfs[flujo_sheet_name]
        
        #find the row where the beginning and end of the table are.
        #+1 just because the table is read with a fake header we need to remove.
        header_idx = raw_data [raw_data.iloc[:, 2].eq(first_data)].index.values[0] + 1
        footer_idx = raw_data [raw_data.iloc[:, 2].eq(last_data)].index.values[0] + 1

        # specify number of rows to read after header row
        num_rows = footer_idx - header_idx
        clean_data = pd.read_excel(
            io=self.excel_path,
            sheet_name=flujo_sheet_name,
            header=header_idx,
            nrows=num_rows,
            usecols="C:I"
        )
        
        #drop the rows with NaN in the first 
        clean_data = clean_data.dropna(subset=[clean_data.columns[0]])
        
        #move the first row as the new header
        clean_data = clean_data.T
        new_header = clean_data.iloc[0]
        clean_data = clean_data[1:]
        clean_data.columns = new_header
                
        self.flujo_diameter_table = clean_data
        return self.flujo_diameter_table

    def retrieve_UF(self):        
        flujo_df = self.flujo_diameter_table
        
        # Find the column index for 'UF'
        col_idx = flujo_df.columns.get_loc('UF')
        
        # Retrieve the value directly below 'UF', which is in the first row following the header
        UF_value = flujo_df.iloc[0, col_idx]  # Using 0 because it's the first row below the header
        return UF_value
    
    def calculate_depr(self, main_sheet_name="BBDD - Error Actual"):
        main_df = self.dfs[main_sheet_name]
        
        inversion_col = 'Inversion'
        Depr_col = 'Depr'
        
        # Data validation: Ensure 'Inversion' is numeric and handle non-numeric entries
        main_df[inversion_col] = pd.to_numeric(main_df[inversion_col], errors='coerce')
        
        # Fill any NaN values that result from non-numeric to numeric conversion (optional)
        main_df[inversion_col].fillna(0, inplace=True)
    
        # Calculate depreciation for the first three periods
        for i in range(1, 4):
            main_df[f'{Depr_col} {i}'] = main_df[inversion_col] / (10/3)
        
        # Calculate the sum of the first three periods
        main_df['Depr sum'] = main_df[[f'{Depr_col} {i}' for i in range(1, 4)]].sum(axis=1)
        
        # Calculate the fourth depreciation, adjusting for any difference
        main_df[f'{Depr_col} 4'] = main_df[inversion_col] - main_df['Depr sum']
        
        # Drop the temporary sum column
        main_df.drop('Depr sum', axis=1, inplace=True)
        
        # Update the DataFrame in your dictionary
        self.dfs[main_sheet_name] = main_df

    def calculate_valor_restante(self, main_sheet_name="BBDD - Error Actual"):
        main_df = self.dfs[main_sheet_name]
        inversion_col = 'Inversion'
        val_res_col = 'Valor res'
        # Calculate 'Valor restante'
        # Negate the result of subtracting one-tenth of 'Inversion' from 'Inversion'
        main_df[val_res_col] = -(main_df[inversion_col] - (main_df[inversion_col] / 10))
        
        # Replace NaN values in 'Valor restante' with 0 after calculation
        main_df[val_res_col].fillna(0, inplace=True)
    
        # Update the DataFrame in your dictionary
        self.dfs[main_sheet_name] = main_df

            
    def calculate_base_A(self, main_sheet_name="BBDD - Error Actual"):
        main_df = self.dfs[main_sheet_name]
        
        for i in range(1, 5):  # from {1, 2, 3, 4}
            base_col = f'base_A {i}'
            depr_col = f'Depr {i}'
            c_e_col = f'C_E_{i}'
            ingreso_col = f'Ingresos_{i}'
            
            main_df = validate_numeric_columns(main_df, [depr_col, c_e_col, ingreso_col])
            
            # Correct the list of columns for summation
            main_df[base_col] = main_df[[depr_col, c_e_col, ingreso_col]].sum(axis=1)
            main_df[base_col].fillna(0, inplace=True)
            
        for i in range(5, 16):
            base_col = f'base_A {i}'
            c_e_col = f'C_E_{i}'
            ingreso_col = f'Ingresos_{i}'
            
            main_df = validate_numeric_columns(main_df, [c_e_col, ingreso_col])
            
            # Correct the list of columns for summation
            main_df[base_col] = main_df[[c_e_col, ingreso_col]].sum(axis=1)
            main_df[base_col].fillna(0, inplace=True)
        
        #para la base 15, también se suma la depr (como si se reinstalara uno nuevo y se vendiera el viejo)
        #excepto si es un ULTRA+38, ahi no se hace nada

        i = 15
        base_col = f'base_A {i}'
        c_e_col = f'C_E_{i}'
        ingreso_col = f'Ingresos_{i}'
        depr_col = f'Depr 1'
        main_df = validate_numeric_columns(main_df, [depr_col, c_e_col, ingreso_col])
         
        # Create condition for ULTRA+38 exclusion
        condition = (main_df['DIAMETRO_MEDIDOR'] < 38) | (main_df['Curva proy'] != 'ULTRA')
    
        main_df[base_col] = np.where(
            condition,
            main_df[base_col] + main_df[depr_col],
            main_df[base_col]
        )
        
        # # Apply condition and adjust base_A 15
        # main_df[base_col] = np.where(
        #     condition,
        #     main_df[[c_e_col, ingreso_col]].sum(axis=1),  # Exclude Depr 1 if condition is True
        #     main_df[[depr_col, c_e_col, ingreso_col]].sum(axis=1)  # Include Depr 1 if condition is False
        # )
        
        main_df[base_col].fillna(0, inplace=True)
        
        self.dfs[main_sheet_name] = main_df
        
    def calculate_base_B(self, main_sheet_name="BBDD - Error Actual"):
        main_df = self.dfs[main_sheet_name]
    
        # Initialize base_B 1 to be equal to base_A 1
        main_df['base_B 1'] = main_df['base_A 1']
    
        # Loop through the range from 2 to 16 to calculate subsequent base_B values
        for i in range(2, 16):  # range(2, 17) to cover base_B 2 to base_B 16
            previous_base_B = f'base_B {i - 1}'
            current_base_A = f'base_A {i}'
            current_base_B = f'base_B {i}'
    
            # Apply logic: if previous base_B < 0, add current base_A to it; otherwise, take current base_A
            main_df[current_base_B] = main_df.apply(
                lambda row: row[previous_base_B] + row[current_base_A] if row[previous_base_B] < 0 else row[current_base_A],
                axis=1
            )
    
        # Update the DataFrame in your dictionary
        self.dfs[main_sheet_name] = main_df
        
    def calculate_impuesto(self, main_sheet_name="BBDD - Error Actual", tax_rate=0.27):
        main_df = self.dfs[main_sheet_name]
    
        # Prepare a new DataFrame to hold the impuesto calculations
        impuesto_data = {}
    
        # Loop through the range from 1 to 16 to calculate impuesto for each base_B
        for i in range(1, 16):  # Adjusting based on your note about the range
            base_B_col = f'base_B {i}'
            impuesto_col = f'impuesto {i}'
    
            # Calculate impuesto and store in the dictionary
            impuesto_data[impuesto_col] = main_df.apply(
                lambda row: -1 * row[base_B_col] * tax_rate if row[base_B_col] >= 0 else 0,
                axis=1
            )
    
        # Convert dictionary to DataFrame
        impuesto_df = pd.DataFrame(impuesto_data, index=main_df.index)
    
        # Concatenate with the original DataFrame
        main_df = pd.concat([main_df, impuesto_df], axis=1)
    
        # Update the DataFrame in your dictionary
        self.dfs[main_sheet_name] = main_df
        
    # def calculate_flujo(self, main_sheet_name="BBDD - Error Actual"):
    #     main_df = self.dfs[main_sheet_name]
        
    #     valor_res_col = 'Valor res'
    #     inversion_col = 'Inversion'
        
    #     # Prepare a new DataFrame to hold the impuesto calculations
    #     flujo_data = {}
        
    #     for i in range(1,16):
    #         ingresos_col = f'Ingresos_{i}'
    #         C_e_col = f'C_E_{i}'
    #         impuesto_col = f'impuesto {i}'
    #         flujo_col = f'Flujo {i}'
            
    #         flujo_data[flujo_col] = main_df[[ingresos_col, C_e_col, impuesto_col]].sum(axis=1)
    #         if (i == 14):
    #             flujo_data[flujo_col] = flujo_data[flujo_col] + main_df[inversion_col]
    #         if (i == 15):
    #             flujo_data[flujo_col] = flujo_data[flujo_col] + main_df[valor_res_col]
            
    #         flujo_data[flujo_col] = pd.to_numeric(flujo_data[flujo_col], errors='coerce').fillna('')
                
    #     # Convert dictionary to DataFrame
    #     flujo_df = pd.DataFrame(flujo_data, index=main_df.index)
        
    #     # Concatenate with the original DataFrame
    #     main_df = pd.concat([main_df, flujo_df], axis=1)
    
    #     # Update the DataFrame in your dictionary
    #     self.dfs[main_sheet_name] = main_df        
    
    def calculate_flujo(self, main_sheet_name="BBDD - Error Actual"):
        main_df = self.dfs[main_sheet_name]
        
        valor_res_col = 'Valor res'
        inversion_col = 'Inversion'
    
        # Prepare a new dictionary to hold the flujo calculations
        flujo_data = {}
    
        for i in range(1, 16):
            ingresos_col = f'Ingresos_{i}'
            C_e_col = f'C_E_{i}'
            impuesto_col = f'impuesto {i}'
            flujo_col = f'Flujo {i}'
    
            # Calculate the flujo sum for this iteration
            flujo_data[flujo_col] = main_df[[ingresos_col, C_e_col, impuesto_col]].sum(axis=1)
            # print(f"Calculated {flujo_col} sum.")
    
            # Apply conditions for rollover
            if i in [14, 15]:
                # Check if either condition is true to apply rollover
                condition = (main_df['DIAMETRO_MEDIDOR'] < 38) | (main_df['Curva proy'] != 'ULTRA')
                rollover_addition = main_df[inversion_col] if i == 14 else main_df[valor_res_col]
                flujo_data[flujo_col] += np.where(condition, rollover_addition, 0)
                main_df.apply(lambda row: check_rollover_conditions(row, i, inversion_col, valor_res_col), axis=1)
    
        # Convert dictionary to DataFrame and update the main DataFrame
        flujo_df = pd.DataFrame(flujo_data, index=main_df.index)
        main_df = pd.concat([main_df, flujo_df], axis=1)
        # print("Updated main DataFrame with Flujo data.")
    
        self.dfs[main_sheet_name] = main_df

    def calculate_van(self, main_sheet_name="BBDD - Error Actual", rate=0.07):
        main_df = self.dfs[main_sheet_name]
    
        flujo_col = 'Flujo'
        # Define the columns for the cash flows
        flujo_cols = [f'{flujo_col} {i}' for i in range(1, 16)]  # Adjust range if necessary
        inversion_col = 'Inversion'
        van_col = 'VAN'
        van_col_aux = 'VAN_AUX'
        
                
        # Calculate NPV manually (numpy_f is failing us)
        main_df[van_col_aux] = main_df.apply(lambda row: sum(
            [row[col] / ((1 + rate) ** (i + 1)) for i, col in enumerate(flujo_cols)]
        ), axis=1)
                        
        #TODO: confirmar con INECON que la formula esté bien aplicada (instalan un medidor en año 16?)
        #agregar la inversion al final
        #... a menos que sea un ultra+38, en ese caso hay tasa de descuento en la inversión
        condition = (main_df['DIAMETRO_MEDIDOR'] < 38) | (main_df['Curva proy'] != 'ULTRA')
    
        main_df[van_col] = np.where(
            condition,
            main_df[van_col_aux] + main_df[inversion_col],
            main_df[van_col_aux] + main_df[inversion_col] / ((1 + rate) ** 16),
        )
        
        # Handle any errors by replacing non-numeric results with 0
        main_df[van_col] = pd.to_numeric(main_df[van_col], errors='coerce').fillna(0)
    
        # Update the DataFrame in your dictionary
        self.dfs[main_sheet_name] = main_df
             
        
    def create_summary(self, consumo, SubmedicionCon, SubmedicionSin,
                             DiferencialVol, DiferencialPorcentual,
                             IngresosRecuperados, ErrorFinalCon, ErrorFinalSin):
        
        summary_template = {
            "Año": list(range(1, 16)),
            "Consumo (m3/año)": None,
            "Submedición Con Proyecto (m3/año)": None,
            "Submedición Sin Proyecto (m3)": None,
            "Diferencial (volumen recuperado, m3/año)": None,
            "Diferencial (% c/r consumo renovados)": None,
            "Ingresos volumen recuperado ($/año)": None,
            "Con Proyecto - Error ponderado final": None,
            "Sin Proyecto - Error ponderado final": None    
        }
        
        #assuming consumo is always a single value
        consumo_list = [consumo for i in range(1,16)]
        
        
        summary_data = summary_template
        summary_data["Consumo (m3/año)"] = consumo_list
        summary_data["Submedición Con Proyecto (m3/año)"] = list(SubmedicionCon)
        summary_data["Submedición Sin Proyecto (m3)"] = list(SubmedicionSin)
        summary_data["Diferencial (volumen recuperado, m3/año)"] = list(DiferencialVol)
        summary_data["Diferencial (% c/r consumo renovados)"] = list(DiferencialPorcentual)
        summary_data["Ingresos volumen recuperado ($/año)"] = list(IngresosRecuperados)
        summary_data["Con Proyecto - Error ponderado final"] = list(ErrorFinalCon)
        summary_data["Sin Proyecto - Error ponderado final"] = list(ErrorFinalSin)
        return summary_data

    def run_scenario_1(self, van_value = 0):
        main_sheet_name = "BBDD - Error Actual"
        main_df = self.dfs[main_sheet_name]
        scenario_1_name = "RESUMEN E1"
        scenario_1_title = f"Escenario: medidores con VAN > {van_value}"
        
        consumo_col = "CONSUMO PROMEDIO"
        v_sub_columns = [f"V_sub_{i}" for i in range(1,16)]
        v_sub_proy_columns = [f'V_sub_proy_{i}' for i in range(1,16)]
        ingreso_columns = [f"Ingresos_{i}" for i in range(1,16)]
        
        van_col = "VAN"
        filtered_df = main_df[main_df[van_col] > van_value]
        
        # Calculate required fields
        consumo = np.array( filtered_df[consumo_col].sum()) * 12
        submedicion_con_proyecto = np.array( [filtered_df[vsub_proy].sum() for vsub_proy in v_sub_proy_columns] )*12
        submedicion_sin_proyecto = np.array( [filtered_df[vsub].sum() for vsub in v_sub_columns] )*12
                
        diferencial_volumen_recuperado = submedicion_sin_proyecto - submedicion_con_proyecto
        diferencial_porcentaje = (diferencial_volumen_recuperado / consumo)
        ingresos_volumen_recuperado = [filtered_df[ingreso].sum() for ingreso in ingreso_columns]
        
        
        con_proyecto_error_ponderado_final = -1*submedicion_con_proyecto / (submedicion_con_proyecto + consumo)
        sin_proyecto_error_ponderado_final = -1*submedicion_sin_proyecto / (submedicion_sin_proyecto + consumo)
        
        #---- Create the Dictionary
        
        summary = self.create_summary(consumo,
                                 submedicion_con_proyecto,
                                 submedicion_sin_proyecto,
                                 diferencial_volumen_recuperado,
                                 diferencial_porcentaje,
                                 ingresos_volumen_recuperado,
                                 con_proyecto_error_ponderado_final,
                                 sin_proyecto_error_ponderado_final)
        
        for key, value in summary.items():
            if not isinstance(value, list):
                summary[key] = [value] * len(summary["Año"])
        
        summary_df = pd.DataFrame.from_dict(summary, orient='index')
        
        #---- reorder the dataframe for proper output
        
        #move so it aligns with years
        newColumnNames = [str(int(col)+1) for col in summary_df.columns]
        summary_df = summary_df.rename({num: num+1 for num in summary_df.columns},axis="columns")
        
        summary_df["Año"] = summary_df.index
        
        #reorder
        colsCampoPrimero = ["Año"] + [col for col in summary_df.columns if col != "Año"]
        summary_df = summary_df[colsCampoPrimero]
        
        summary_df.iloc[0, summary_df.columns.get_loc("Año")] = scenario_1_title
        summary_df.iloc[0, 1:] = np.nan #remove duplicated years
        #----- Final processing end
        
        self.dfs[scenario_1_name] = summary_df
        
    
    def run_scenario_2(self, capex_ratio=0.65, sheetname = "RESUMEN E2"):
        self.run_scenario_percentCapex(sheetname = sheetname, capex_ratio = capex_ratio)
        
    def run_scenario_3(self, capex_ratio=0.30, sheetname = "RESUMEN E3"):
        self.run_scenario_percentCapex(sheetname = sheetname, capex_ratio = capex_ratio)
    
    def run_scenario_percentCapex(self, sheetname, capex_ratio):
        main_sheet_name = "BBDD - Error Actual"
        main_df = self.dfs[main_sheet_name]
        scenario_percent_title = f"Escenario: medidores con CAPEX ≈ {int(capex_ratio * 100)}% del CAPEX de E1"
        
        consumo_col = "CONSUMO PROMEDIO"
        v_sub_columns = [f"V_sub_{i}" for i in range(1,16)]
        v_sub_proy_columns = [f'V_sub_proy_{i}' for i in range(1,16)]
        ingreso_columns = [f"Ingresos_{i}" for i in range(1,16)]
        inversion_col = "Inversion"
        van_col = "VAN"
        
        #Calculate the CAPEX of Scenario 1
        positiveVan_df = main_df[main_df[van_col] > 0] 
        capex_scenario1 = -1 * (positiveVan_df[inversion_col].sum() * capex_ratio) #-1 because inversion is defined as a negative number
        
        sorted_df = main_df[main_df[van_col] > 0].sort_values(by=van_col, ascending=False)
        
        # Use cumsum to accumulate the Inversión until it meets or exceeds the target CAPEX
        sorted_df['Inversion_Cum'] = -1*sorted_df[inversion_col].cumsum()
        sorted_df = sorted_df[sorted_df['Inversion_Cum'] <= capex_scenario1]
        
        if sorted_df.empty:
            raise ValueError("No valid rows found that meet the CAPEX condition. Adjust the capex_ratio or check the data.")
        
        # Use the final DataFrame for calculations
        sorted_df = sorted_df.drop(columns='Inversion_Cum')
        
        # Calculate required fields
        consumo = np.array( sorted_df[consumo_col].sum()) * 12
        submedicion_con_proyecto = np.array( [sorted_df[vsub_proy].sum() for vsub_proy in v_sub_proy_columns] )*12
        submedicion_sin_proyecto = np.array( [sorted_df[vsub].sum() for vsub in v_sub_columns] )*12
                
        diferencial_volumen_recuperado = submedicion_sin_proyecto - submedicion_con_proyecto
        diferencial_porcentaje = (diferencial_volumen_recuperado / consumo)
        ingresos_volumen_recuperado = [sorted_df[ingreso].sum() for ingreso in ingreso_columns]
        
        
        con_proyecto_error_ponderado_final = -1*submedicion_con_proyecto / (submedicion_con_proyecto + consumo)
        sin_proyecto_error_ponderado_final = -1*submedicion_sin_proyecto / (submedicion_sin_proyecto + consumo)
        
        #---- Create the Dictionary
        
        summary = self.create_summary(consumo,
                                 submedicion_con_proyecto,
                                 submedicion_sin_proyecto,
                                 diferencial_volumen_recuperado,
                                 diferencial_porcentaje,
                                 ingresos_volumen_recuperado,
                                 con_proyecto_error_ponderado_final,
                                 sin_proyecto_error_ponderado_final)
        
        for key, value in summary.items():
            if not isinstance(value, list):
                summary[key] = [value] * len(summary["Año"])
        
        summary_df = pd.DataFrame.from_dict(summary, orient='index')
        
        #---- reorder the dataframe for proper output
        
        #move so it aligns with years
        newColumnNames = [str(int(col)+1) for col in summary_df.columns]
        summary_df = summary_df.rename({num: num+1 for num in summary_df.columns},axis="columns")
        
        summary_df["Año"] = summary_df.index
        
        #reorder
        colsCampoPrimero = ["Año"] + [col for col in summary_df.columns if col != "Año"]
        summary_df = summary_df[colsCampoPrimero]
        
        summary_df.iloc[0, summary_df.columns.get_loc("Año")] = scenario_percent_title
        summary_df.iloc[0, 1:] = np.nan #remove duplicated years
        #----- Final processing end
        
        self.dfs[sheetname] = summary_df
        
    def run_scenario_4(self, capex_max, sheetname= "RESUMEN E4"):
        main_sheet_name = "BBDD - Error Actual"
        main_df = self.dfs[main_sheet_name]
        scenario_percent_title = f"Escenario: medidores con CAPEX ≈ {int(capex_max)}"
        
        consumo_col = "CONSUMO PROMEDIO"
        v_sub_columns = [f"V_sub_{i}" for i in range(1,16)]
        v_sub_proy_columns = [f'V_sub_proy_{i}' for i in range(1,16)]
        ingreso_columns = [f"Ingresos_{i}" for i in range(1,16)]
        inversion_col = "Inversion"
        van_col = "VAN"
        
        #Calculate the CAPEX of Scenario 1
        positiveVan_df = main_df[main_df[van_col] > 0]     
        sorted_df = main_df[main_df[van_col] > 0].sort_values(by=van_col, ascending=False)
        
        # Use cumsum to accumulate the Inversión until it meets or exceeds the target CAPEX
        sorted_df['Inversion_acumulada'] = -1*sorted_df[inversion_col].cumsum()
        sorted_df = sorted_df[sorted_df['Inversion_acumulada'] <= capex_max]
        
        if sorted_df.empty:
            raise ValueError("No valid rows found that meet the CAPEX condition. Adjust the capex_flat or check the data.")
        
        # Use the final DataFrame for calculations
        sorted_df = sorted_df.drop(columns='Inversion_acumulada')
        
        # Calculate required fields
        consumo = np.array( sorted_df[consumo_col].sum()) * 12
        submedicion_con_proyecto = np.array( [sorted_df[vsub_proy].sum() for vsub_proy in v_sub_proy_columns] )*12
        submedicion_sin_proyecto = np.array( [sorted_df[vsub].sum() for vsub in v_sub_columns] )*12
                
        diferencial_volumen_recuperado = submedicion_sin_proyecto - submedicion_con_proyecto
        diferencial_porcentaje = (diferencial_volumen_recuperado / consumo)
        ingresos_volumen_recuperado = [sorted_df[ingreso].sum() for ingreso in ingreso_columns]
        
        
        con_proyecto_error_ponderado_final = -1*submedicion_con_proyecto / (submedicion_con_proyecto + consumo)
        sin_proyecto_error_ponderado_final = -1*submedicion_sin_proyecto / (submedicion_sin_proyecto + consumo)
        
        #---- Create the Dictionary
        
        summary = self.create_summary(consumo,
                                 submedicion_con_proyecto,
                                 submedicion_sin_proyecto,
                                 diferencial_volumen_recuperado,
                                 diferencial_porcentaje,
                                 ingresos_volumen_recuperado,
                                 con_proyecto_error_ponderado_final,
                                 sin_proyecto_error_ponderado_final)
        
        for key, value in summary.items():
            if not isinstance(value, list):
                summary[key] = [value] * len(summary["Año"])
        
        summary_df = pd.DataFrame.from_dict(summary, orient='index')
        
        #---- reorder the dataframe for proper output
        
        #move so it aligns with years
        newColumnNames = [str(int(col)+1) for col in summary_df.columns]
        summary_df = summary_df.rename({num: num+1 for num in summary_df.columns},axis="columns")
        
        summary_df["Año"] = summary_df.index
        
        #reorder
        colsCampoPrimero = ["Año"] + [col for col in summary_df.columns if col != "Año"]
        summary_df = summary_df[colsCampoPrimero]
        
        summary_df.iloc[0, summary_df.columns.get_loc("Año")] = scenario_percent_title
        summary_df.iloc[0, 1:] = np.nan #remove duplicated years
        #----- Final processing end
        
        self.dfs[sheetname] = summary_df

    def run_scenario_5(self, tasa = 0.07, sheetname= "RESUMEN E5"):
        main_sheet_name = "BBDD - Error Actual"
        main_df = self.dfs[main_sheet_name]
        scenario_percent_title = f"Escenario 5: TIR del flujo resultante igual a {int(tasa*100)}% (VAN = 0)"
        
        consumo_col = "CONSUMO PROMEDIO"
        v_sub_columns = [f"V_sub_{i}" for i in range(1,16)]
        v_sub_proy_columns = [f'V_sub_proy_{i}' for i in range(1,16)]
        ingreso_columns = [f"Ingresos_{i}" for i in range(1,16)]
        inversion_col = "Inversion"
        van_col = "VAN"
                
        # Use cumsum to accumulate the VAN until it's close to 0
        sorted_df = main_df.sort_values(by=van_col, ascending=False)
        sorted_df['VAN_acumulado'] = sorted_df[van_col].cumsum()
        
        closest_to_zero_index = sorted_df['VAN_acumulado'].abs().idxmin()

        # Get the subset of the DataFrame up to and including that row
        subset_df = sorted_df.loc[:closest_to_zero_index]
        
        if subset_df.empty:
            raise ValueError("No valid rows found that meet the TIR condition. Adjust the TIR_PERCENT or check the data.")
        
        # Use the final DataFrame for calculations
        subset_df = subset_df.drop(columns='VAN_acumulado')
        
        # Calculate required fields
        consumo = np.array( subset_df[consumo_col].sum()) * 12
        submedicion_con_proyecto = np.array( [subset_df[vsub_proy].sum() for vsub_proy in v_sub_proy_columns] )*12
        submedicion_sin_proyecto = np.array( [subset_df[vsub].sum() for vsub in v_sub_columns] )*12
                
        diferencial_volumen_recuperado = submedicion_sin_proyecto - submedicion_con_proyecto
        diferencial_porcentaje = (diferencial_volumen_recuperado / consumo)
        ingresos_volumen_recuperado = [subset_df[ingreso].sum() for ingreso in ingreso_columns]
        
        
        con_proyecto_error_ponderado_final = -1*submedicion_con_proyecto / (submedicion_con_proyecto + consumo)
        sin_proyecto_error_ponderado_final = -1*submedicion_sin_proyecto / (submedicion_sin_proyecto + consumo)
        
        #---- Create the Dictionary
        
        summary = self.create_summary(consumo,
                                 submedicion_con_proyecto,
                                 submedicion_sin_proyecto,
                                 diferencial_volumen_recuperado,
                                 diferencial_porcentaje,
                                 ingresos_volumen_recuperado,
                                 con_proyecto_error_ponderado_final,
                                 sin_proyecto_error_ponderado_final)
        
        for key, value in summary.items():
            if not isinstance(value, list):
                summary[key] = [value] * len(summary["Año"])
        
        summary_df = pd.DataFrame.from_dict(summary, orient='index')
        
        #---- reorder the dataframe for proper output
        
        #move so it aligns with years
        newColumnNames = [str(int(col)+1) for col in summary_df.columns]
        summary_df = summary_df.rename({num: num+1 for num in summary_df.columns},axis="columns")
        
        summary_df["Año"] = summary_df.index
        
        #reorder
        colsCampoPrimero = ["Año"] + [col for col in summary_df.columns if col != "Año"]
        summary_df = summary_df[colsCampoPrimero]
        
        summary_df.iloc[0, summary_df.columns.get_loc("Año")] = scenario_percent_title
        summary_df.iloc[0, 1:] = np.nan #remove duplicated years
        #----- Final processing end
        
        self.dfs[sheetname] = summary_df
            
    def general_cleanup(self):
        # List the sheet names to remove

        # Remove each specified sheet from self.dfs if it exists
        if config.remove_sheets:
            self.dfs = {sheet_name: df for sheet_name, df in self.dfs.items() if sheet_name in config.sheets_to_keep}

        # Check sheets_config and adjust columns of DataFrames
        if config.reorder_columns:
            for sheet_name, columns in config.sheets_config.items():
                if sheet_name in self.dfs:
                    # Reorder and select only the columns listed in sheets_config if they exist in the DataFrame
                    self.dfs[sheet_name] = self.dfs[sheet_name][[col for col in columns if col in self.dfs[sheet_name].columns]]

    def export_to_excel(self, filename_out):
        # Using XlsxWriter as the engine
        with pd.ExcelWriter(filename_out, engine='xlsxwriter') as writer:
            for sheet_name, df in self.dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        # print(f"Dataframe(s) exported successfully to {filename_out}")

if __name__ == "__main__":
    folder = r"C:\GitHub\6020.Inecon_ModeloVanMedicion\6020.Inecon_ModeloVanMedicion\Code\Limpio\Excels\preprocessed"
    filename_in = r"preprocessed_mm_modelo MM - TALCA 28.12_v4.xlsx"
    
    excel_path = f"{folder}\\{filename_in}"
    # param_path = f"{folder}\\{filename_param}"
    
    filename_out = r"OUTPUT.xlsx"

    # Initialize the class without loading any DataFrames
    van_calc = VanCalculator()

    # Load the main DataFrame(s) and parameters
    van_calc.load_df_from_excel(excel_path)
    # van_calc.load_parameters(param_path)

    # Perform your operations
    van_calc.run_all()

    # Export the result to an output Excel file
    van_calc.export_to_excel(filename_out)

