import pandas as pd
import numpy as np
from itertools import product

def convert_to_float(value):
    """Converts a percentage string to a float, if applicable, otherwise converts directly to float."""
    if isinstance(value, str):
        return float(value.strip('%')) / 100 if '%' in value else float(value)
    return float(value)

def steps(_min, _max, intermediate_steps):
    if intermediate_steps == 0:
        return [_min, _max]
    return list(np.linspace(_min, _max, intermediate_steps + 2))

def create_scenarios(df):
    parameters = {}
    for _, row in df.iterrows():
        name = row['Nombre']
        valor = row['Valor']
        
        if not pd.isna(valor):
            parameters[name] = [convert_to_float(valor)]
        else:
            min_value = row['min']
            max_value = row['max']
            intermediate_steps = row['pasos intermedios']
            
            if pd.isna(min_value) or pd.isna(max_value) or pd.isna(intermediate_steps):
                raise ValueError(f"Missing min, max, or pasos intermedios for parameter '{name}'")
            
            _min = convert_to_float(min_value)
            _max = convert_to_float(max_value)
            
            parameters[name] = steps(_min, _max, int(intermediate_steps))

    keys, values = zip(*parameters.items())
    scenarios = [dict(zip(keys, v)) for v in product(*values)]
    
    ordered_scenarios = []
    #put Escenario number, then reorder
    for i, scenario in enumerate(scenarios, 1):
        scenario['Escenario'] = i
        # Reorder so 'Escenario' is first
        ordered_scenario = {'Escenario': scenario.pop('Escenario')}
        ordered_scenario.update(scenario)
        ordered_scenarios.append(ordered_scenario)
    
    print(scenarios_to_string(ordered_scenarios))
    return ordered_scenarios

def scenarios_to_string(scenarios):
    text = ""
    for i, scenario in enumerate(scenarios, 1):
        scenario['Escenario'] = i
        # Format the scenario into a string
        scenario_str = ", ".join(f"{k}: {v}" for k, v in scenario.items())
        text += f"{scenario_str}\n"
    return text

# Example dictionary for reference
example_parameters = {
    "Nombre": ["Tarifa", "Impuesto", "Otro"],
    "Notas": ["Para cálculo de VAN", "Para cálculo de impuesto", "Parámetro global de ejemplo"],
    "Valor": [None, "27%", None],
    "min": ["7%", "25%", "15"],
    "max": ["10%", "30%", "20"],
    "pasos intermedios": [0, None, 1]
}

if __name__ == "__main__":
    folder = r"C:\GitHub\6020.Inecon_ModeloVanMedicion\6020.Inecon_ModeloVanMedicion\Code\Limpio\Excels"
    filename = r"modelo MM - TALCA 28.12_v5.xlsx"
    full_path = f"{folder}/{filename}"
    df=pd.read_excel(full_path, sheet_name="PARAMETROS GLOBALES")
    
    scenarios = create_scenarios(df)
    print(scenarios_to_string(scenarios))
    