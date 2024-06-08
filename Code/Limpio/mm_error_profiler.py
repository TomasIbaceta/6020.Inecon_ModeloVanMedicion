from mm_calc import *

import time
import cProfile
import pstats
import pandas as pd
import numpy as np

def mm_calc_run_alg_all_TEST(exceldata):
    # Retrieve all localities
    localidades = get_localidades_from_excel(exceldata)
    
    # Initialize lists to hold combined results for each sheet
    combined_results = {}
    
    # We need certain sheets to go straight through to the excel for use further
    # down the pipeline.
    
    passthrough_sheets = ['FLUJO E1', 'Datos_Tarifas', 'PARAMETROS GLOBALES', 'MANEJO DE ESCENARIOS']
    
    for special_sheet in passthrough_sheets:
        # Check if the sheet exists in the original exceldata
        if special_sheet in exceldata:
            combined_results[special_sheet] = exceldata[special_sheet]
    
    # Iterate through all localities and process them
    for localidad in localidades:
        
        if localidad != "CABILDO":
            continue
        
        localidades_necesarias = get_localidades_in_DATOS(exceldata)
        if localidad not in localidades_necesarias:
            continue
        
        # Read parameters for the current locality
        parameters = read_parameters_from_excel(exceldata, localidad)
        
        # Run the main algorithm with the current parameters
        profiler = cProfile.Profile()
        profiler.enable()
        output = mm_calc_test_err(exceldata, parameters)
        profiler.disable()
        profiler.dump_stats(f"profiling_results_{localidad}.prof")
        
        # Concatenate each output into the combined_results dictionary
        for sheet_name, df in output.items():
            # If the sheet name already exists, append the dataframe
            if sheet_name in combined_results:
                combined_results[sheet_name] = pd.concat([combined_results[sheet_name], df], ignore_index=True)
            else:
                # Otherwise, create a new entry in the dictionary
                combined_results[sheet_name] = df
    
    return combined_results

def mm_calc_test_err(exceldata, parametros_localidad):
    """
    run the whole script until where it calculates the error. that way I can measure what's going on'
    """
    
    diametro_max = 38  # dummy
    error_ultra = -0.02
        
    data_completa = exceldata["DATOS"]
    df_clases = exceldata["CLASES"]
    df_edad = exceldata["EDAD"]
    df_caudal = exceldata["CAUDAL MODIFICADO"]
    df_caudal_original = exceldata["CAUDAL ORIGINAL"]
    # df_params =exceldata["PARAMETROS"]
    df_autocontrol = exceldata["PROGRAMA AUTOCONTROL"]
    df_porc_caudal_para_pendiente = exceldata["PORC CAUDAL PENDIENTE"]
    df_clase_grupo_decaimiento = exceldata["REFERENCIA CLASES DECAIMIENTO"]

    # Carga de parámetros de localidad
    fecha_estudio = parametros_localidad["fecha_estudio"]
    localidad = parametros_localidad["localidad"]
    sectores = parametros_localidad["sectores"]
    
    tipo_error_CAlto = parametros_localidad["tipo_error_CAlto"]
    edad_minima_decaimiento = parametros_localidad["edad_minima_decaimiento"]
    edad_maxima_decaimiento = parametros_localidad["edad_maxima_decaimiento"]
    edad_extra_decaimiento = parametros_localidad["edad_extra_decaimiento"]
    problema_calidad_agua = parametros_localidad["problema_calidad_agua"]
    autocontrol_diametro_alto = parametros_localidad["autocontrol_diametro_alto"]
    edad_inicial_iyf = parametros_localidad["edad_inicial_iyf"]
    edad_final_iyf = parametros_localidad["edad_final_iyf"]

    # parametros fijos
    intervalos_caudal_bajo = ['8-16', '16-32', '32-66']
    intervalos_caudal_bajo_c25 = ['16-32', '32-66']
    medidores_presicion = ['R400', 'R800']

    # seleccionar datos
    data = data_completa.loc[(data_completa["LOCALIDAD"] == localidad) & (data_completa["SECTOR AP"].isin(sectores))] 
    autocontrol = seleccionar_autocontrol(df_autocontrol, localidad, tipo_error_CAlto)

    # Calculo decaimiento
    pendiente = calcular_decaimiento(df_edad,
                                     problema_calidad_agua=problema_calidad_agua,
                                     caudal=df_porc_caudal_para_pendiente)
    caudal_ext = df_caudal.merge(pendiente[["Clase", "Pendiente"]], on='Clase')

    # Obtener clases corregidas
    data = data.merge(df_clases, left_on="CLASE", right_on="Clase")

    # Obtener variable dicotomica de diametro
    data["Diametro_max"] = data["DIAMETRO_MEDIDOR"] < diametro_max

    # Generar clase del medidor y grupo en el que pertenece
    data[["Clase", "Grupo"]] = data.apply(lambda x: armar_clase(x, diametro_max, medidores_presicion), axis=1, result_type="expand") 
    
    # Initial setup - calculate current adjusted age and error
    data["Antiguedad ajustada"] = calcular_antiguedad(data["FECHA_MONTAJE"],
                                                      fecha_estudio=fecha_estudio, 
                                                      edad_extra=edad_extra_decaimiento, 
                                                      edad_minima=edad_minima_decaimiento,
                                                      edad_maxima=edad_maxima_decaimiento)
    
    data = armar_curva_proyectada(data)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Let's profile this
    start_time = time.time()
    E = data.merge(df_clase_grupo_decaimiento, on="Grupo", how='left')\
            .merge(caudal_ext, left_on="Clase referencia", right_on="Clase", how='left')\
            .groupby("INSTALACION").apply(lambda x: error_test(x, autocontrol, autocontrol_diametro_alto,  
                                                               error_ultra, intervalos_caudal_bajo, 
                                                               intervalos_caudal_bajo_c25))
    end_time = time.time()
    print(f"time: {end_time - start_time} ")
    
    profiler.disable()
    profiler.dump_stats("profiling_results_internal.prof")
    
    # Expected output
    # expected_E = pd.Series({
    #     1244838.0: -0.056422,
    #     1245685.0: -0.057837,
    #     60209613.0: -0.056828,
    #     60298899.0: -0.042350
    # }, name='INSTALACION')
    
    expected_E = pd.read_csv("E_output.csv", index_col=0).squeeze("columns")
    
    # Now you can compare it with the computed E
    comparison = np.allclose(E.values, expected_E.values, atol=1e-8)
    print("Comparison result:", comparison)
    # print("Computed E:\n", E)
    # print("Expected E:\n", expected_E)

    return data

# def error_test(group, autocontrol, autocontrol_diametro_alto, error_ultra, intervalos_caudal_bajo, intervalos_caudal_bajo_c25, flag=1):
#     """Calcula el error de cada medidor tomando todas las consideraciones correspondientes."""

#     diametro = group["Diametro_max"].tolist()[0]
#     if flag == 1:    
#         grupo = group["Grupo"].tolist()[0]
#     else:
#         grupo = group["Curva proy"].tolist()[0]
        
#     if grupo == 'ULTRA':
#         return error_ultra
        
#     if grupo == 'C-25':
#         intervalos_cb = intervalos_caudal_bajo_c25
#     else:
#         intervalos_cb = intervalos_caudal_bajo
        
#     if diametro: 
#         group['Decaimiento'] = group.apply(lambda x: x["Año 0"] + x["Pendiente"] * np.log(x["Antiguedad ajustada"]) / 100 if x["Intervalo (l/h)"] in intervalos_cb else x["Año 0"], axis=1)
#         return autocontrol * (1 - group['% Consumo'].sum()) + sum(group['% Consumo'] * group['Decaimiento'])
#     else:
#         group['Decaimiento'] = group.apply(lambda x: x["Año 0"] + x["Pendiente"] * np.log(x["Antiguedad ajustada"]) / 100 if x["Intervalo (l/h)"] in intervalos_cb else x["Año 0"], axis=1)
#         return autocontrol_diametro_alto * (1 - group['% Consumo'].sum()) + sum(group['% Consumo'] * group['Decaimiento'])

def error_test(group, autocontrol, autocontrol_diametro_alto, error_ultra, intervalos_caudal_bajo, intervalos_caudal_bajo_c25, flag=1):
    """Calcula el error de cada medidor tomando todas las consideraciones correspondientes."""

    diametro = group["Diametro_max"].values[0]
    grupo = group["Grupo"].values[0] if flag == 1 else group["Curva proy"].values[0]
    
    if grupo == 'ULTRA':
        return error_ultra
    
    intervalos_cb = intervalos_caudal_bajo_c25 if grupo == 'C-25' else intervalos_caudal_bajo
    
    intervalos_cb_set = set(intervalos_cb)
    is_intervalo_cb = group["Intervalo (l/h)"].isin(intervalos_cb_set).values
    
    antiguedad_log = np.log(group["Antiguedad ajustada"].values)
    decaimiento = group["Año 0"].values + (group["Pendiente"].values * antiguedad_log / 100) * is_intervalo_cb
    
    consumo = group['% Consumo'].values
    consumo_sum = np.sum(consumo)
    consumo_decaimiento_sum = np.sum(consumo * decaimiento)
    
    if diametro:
        return autocontrol * (1 - consumo_sum) + consumo_decaimiento_sum
    else:
        return autocontrol_diametro_alto * (1 - consumo_sum) + consumo_decaimiento_sum


if __name__ == "__main__": 
    excelFolder = r"C:\GitHub\6020.Inecon_ModeloVanMedicion\6020.Inecon_ModeloVanMedicion\Code\Limpio\Excels"
    # excelName = r"error_test.xlsx"
    excelName = r"INPUT ADV 28.05 - prueba.xlsx"
    excelPath = f"{excelFolder}\{excelName}"
    exceldata = pd.read_excel(excelPath, sheet_name=None)
        
    profiler = cProfile.Profile()
    profiler.enable()
    
    salida = mm_calc_run_alg_all_TEST(exceldata)
    
    profiler.disable()
    profiler.dump_stats("profiling_results_main.prof")
    stats = pstats.Stats(profiler).sort_stats('cumulative')
