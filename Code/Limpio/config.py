import numpy as np

def generate_column_names(base_name, count, start=1, step=1, separator='_'):
    """
    Generates a list of column names with a base name followed by numbers from start to count.
    
    Args:
    base_name (str): The base name of the column.
    count (int): The number of sequential names to generate.
    start (int): The starting number for the sequence.
    step (float): The increment between each number in the sequence.
    separator (str): The separator between the base name and the number.

    Returns:
    list: List of formatted column names.
    """
    return [f"{base_name}{separator}{int(i) if i.is_integer() else i}"
               for i in np.arange(start, count * step + start, step)]

class AppConfig:
    def __init__(self):
        self.reorder_columns = True
        self.remove_sheets = True
        
        self.sheets_to_keep = ['BBDD - Error Actual', 'Resultados loc', 'Inicial y final',
                               'RESUMEN E1','RESUMEN E2', 'RESUMEN E3', 'RESUMEN E4', 'RESUMEN E5']
        
        
        self.sheets_config = {
            "BBDD - Error Actual": [
                "LOCALIDAD", "DIAMETRO_MEDIDOR", "FECHA_MONTAJE", "CLASE", "INSTALACION", 
                "CONSUMO PROMEDIO", "SECTOR AP", "Clase corregida", 
                "Diametro_max", "Antiguedad ajustada", "Curva proy", "Error", 
                *generate_column_names("Error", 15),
                "Clase", "Grupo", 
                *generate_column_names("Error", 16, start=0.5, step=1, separator='_'),  # Error_0.5 to Error_15.5
                "V sub", "Precision", "DN", "Tarifa", 
                *generate_column_names("V_sub", 15),
                *generate_column_names("V_sub_proy", 15),
                *generate_column_names("Ingresos", 15),
                *generate_column_names("C_E", 15),
                *generate_column_names("Depr", 4, separator=' '),
                "Valor res", 
                *generate_column_names("base_A", 15, separator=' '),
                *generate_column_names("base_B", 15, separator=' '),
                *generate_column_names("impuesto", 15, separator= ' '),
                *generate_column_names("Flujo", 15, separator=' '),
                "Inversion", 
                "VAN",
                *generate_column_names("Usado en E", 5, separator='')
            ]
        }
        
        global_params_default = {
            'Impuesto': 0.27,  # Default tax rate
            'Tarifa': 0.07      # Default discount rate
        }