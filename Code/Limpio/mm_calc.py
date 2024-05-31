import pandas as pd
import numpy as np

##PETORCA
# direccion_excel=r"C:\GitHub\6020.Inecon_ModeloVanMedicion\Code\micromed_input__Datos INPUT modelo MM - PETORCA 28.12.xlsx"
direccion_excel=r"C:\GitHub\6020.Inecon_ModeloVanMedicion\6020.Inecon_ModeloVanMedicion\Code\Limpio\Excels\modelo MM - TALCA 28.12_v3.xlsx"
direccion_excel_salida = "ANY-Processed_MM.xlsx"

import pandas as pd
import numpy as np

def mm_calc_run_alg_all(exceldata, scenario:dict):
    # Retrieve all localities
    localidades = get_localidades_from_excel(exceldata)
    
    # Initialize lists to hold combined results for each sheet
    combined_results = {}
    
    # We need certain sheets to go straight through to the excel for use further
    # down the pipeline.
    
    passthrough_sheets = ['FLUJO E1', 'Datos_Tarifas', 'PARAMETROS GLOBALES']
    
    for special_sheet in passthrough_sheets:
        # Check if the sheet exists in the original exceldata
        if special_sheet in exceldata:
            combined_results[special_sheet] = exceldata[special_sheet]
    
    # Iterate through all localities and process them
    for localidad in localidades:
        
        localidades_necesarias=get_localidades_in_DATOS(exceldata)
        if localidad not in localidades_necesarias:
            continue
        
        # Read parameters for the current locality
        parameters = read_parameters_from_excel(exceldata, localidad)
        
        # Run the main algorithm with the current parameters
        output = mm_calc_run_alg(exceldata, parameters, scenario)
        
        # Concatenate each output into the combined_results dictionary
        for sheet_name, df in output.items():
            # If the sheet name already exists, append the dataframe
            if sheet_name in combined_results:
                combined_results[sheet_name] = pd.concat([combined_results[sheet_name], df], ignore_index=True)
            else:
                # Otherwise, create a new entry in the dictionary
                combined_results[sheet_name] = df
    
    return combined_results

def get_localidades_in_DATOS(exceldata):
    data_completa =exceldata["DATOS"]
    return data_completa["LOCALIDAD"].unique().tolist()
    
def mm_calc_run_alg(exceldata, parametros_localidad, scenario:dict):
    
    #load global params from scenario
    try:
        diametro_max = scenario["Diametro Max"]
        error_ultra = scenario["Error Ultra"]
    except:
        raise ValueError("invalid Scenario")
    
    data_completa =exceldata["DATOS"]
    df_clases =exceldata["CLASES"]
    df_edad =exceldata["EDAD"]
    df_caudal =exceldata["CAUDAL MODIFICADO"]
    df_caudal_original =exceldata["CAUDAL ORIGINAL"]
    df_params =exceldata["PARAMETROS"]
    df_autocontrol =exceldata["PROGRAMA AUTOCONTROL"]
    df_porc_caudal_para_pendiente=exceldata["PORC CAUDAL PENDIENTE"]
    df_clase_grupo_decaimiento=exceldata["REFERENCIA CLASES DECAIMIENTO"]

    ## Carga de parámetros de localidad
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

    #parametros fijos
    intervalos_caudal_bajo=['8-16','16-32', '32-66' ]
    intervalos_caudal_bajo_c25=['16-32', '32-66' ]
    medidores_presicion=['R400', 'R800']

    #selecccionar datos
    data = data_completa.loc[(data_completa["LOCALIDAD"]== localidad) & (data_completa["SECTOR AP"].isin(sectores))] 
    autocontrol= seleccionar_autocontrol(df_autocontrol, localidad, tipo_error_CAlto)


    # Calculo decaimiento
    pendiente = calcular_decaimiento(df_edad,
                                    problema_calidad_agua=problema_calidad_agua,
                                    caudal=df_porc_caudal_para_pendiente)
    caudal_ext= df_caudal.merge(pendiente[["Clase","Pendiente"]], on='Clase')

    # Obtener clases corregidas
    data=data.merge(df_clases, left_on="CLASE",right_on="Clase")

    # Obtener variable dicotomica de diametro
    data["Diametro_max"]=data["DIAMETRO_MEDIDOR"]<diametro_max

    #Generar clase del medidor y grupo en el que pertenece
    data[["Clase", "Grupo"]]=data.apply(lambda x:  armar_clase(x, diametro_max, medidores_presicion), axis=1,result_type="expand") 
    
    # Initial setup - calculate current adjusted age and error
    data["Antiguedad ajustada"] = calcular_antiguedad(data["FECHA_MONTAJE"],
                                                      fecha_estudio=fecha_estudio, 
                                                      edad_extra=edad_extra_decaimiento, 
                                                      edad_minima=edad_minima_decaimiento,
                                                      edad_maxima=edad_maxima_decaimiento)
    
    data = armar_curva_proyectada(data)
    
    #----- SNIPPET BEGIN HERE ----
    
    # Save the initial error calculation as error_0
    E = data.merge(df_clase_grupo_decaimiento, on="Grupo", how='left')\
            .merge(caudal_ext, left_on="Clase referencia", right_on="Clase", how='left')\
            .groupby("INSTALACION").apply(lambda x: error(x, autocontrol, autocontrol_diametro_alto,  
                                                          error_ultra, intervalos_caudal_bajo, 
                                                          intervalos_caudal_bajo_c25))
    E = E.reset_index(name='Error')
    data = data.merge(E, on='INSTALACION', how='left')
    
    # Iterate through 1 to 15 years into the future using Clase corregida
    for i in range(1, 16): 
        
        data["Antiguedad ajustada"] = data["FECHA_MONTAJE"].apply(
            lambda fecha_montaje: calcular_antiguedad(
                fecha_montaje,
                fecha_estudio + pd.DateOffset(years=i),
                edad_extra=edad_extra_decaimiento,
                edad_minima=edad_minima_decaimiento,
                edad_maxima=edad_maxima_decaimiento
            )
        )
        
        E = data.merge(df_clase_grupo_decaimiento, on="Grupo", how='left')\
                .merge(caudal_ext, left_on="Clase referencia", right_on="Clase", how='left')\
                .groupby("INSTALACION").apply(lambda x: error(x, autocontrol, autocontrol_diametro_alto,  
                                                              error_ultra, intervalos_caudal_bajo, 
                                                              intervalos_caudal_bajo_c25))
        E = E.reset_index(name=f'Error_{i}')
        data = data.merge(E, on='INSTALACION', how='left')
    
    # Save the original 'Clase corregida' column
    # data['Original Clase Corregida'] = data['Clase corregida']
    
    # Replace 'Clase corregida' with 'Curva proy' to calculate Error_0.5 and subsequent errors
    # data['Clase corregida'] = data['Curva proy']
    
    #"Antiguedad ajustada" partiendo en 0.5 to the current year after all iterations are complete
    data["Antiguedad ajustada"] = calcular_antiguedad(fecha_estudio,
                                                      fecha_estudio=fecha_estudio + pd.DateOffset(months=6), 
                                                      edad_extra=edad_extra_decaimiento, 
                                                      edad_minima=edad_minima_decaimiento,
                                                      edad_maxima=edad_maxima_decaimiento)
        
    data['Original Clase'] = data['Clase']
    data.drop(['Clase'], axis=1, inplace=True)
    data['Clase'] = data['Curva proy']
    
    data['Original Grupo'] = data['Grupo']
    data.drop(['Grupo'], axis=1, inplace=True)
    data['Grupo'] = data['Curva proy']
    
    # Calculate Error_0.5 using 'Curva proy'
    E = data.merge(df_clase_grupo_decaimiento, on="Grupo", how='left')\
            .merge(caudal_ext, left_on="Clase referencia", right_on="Clase", how='left')\
            .groupby("INSTALACION").apply(lambda x: error(x, autocontrol, autocontrol_diametro_alto,  
                                                          error_ultra, intervalos_caudal_bajo, 
                                                          intervalos_caudal_bajo_c25, 1))
    E = E.reset_index(name='Error_0.5')
    data = data.merge(E, on='INSTALACION', how='left')
    
    # Calculate Error_1.5 to Error_15.5 using Curva proy
    for col_sufix in range(1, 16):
        if(col_sufix == 14 ): #excel tiene nota diciendo que se reponen al año 14.
            i=0
        elif (col_sufix == 15):
            i=1
        else:
            i=col_sufix
            
        data["Antiguedad ajustada"] = data["FECHA_MONTAJE"].apply(
            lambda fecha_montaje: calcular_antiguedad(
                fecha_estudio,
                fecha_estudio + pd.DateOffset(years=i) + pd.DateOffset(months=6),
                edad_extra=edad_extra_decaimiento,
                edad_minima=edad_minima_decaimiento,
                edad_maxima=edad_maxima_decaimiento
            )
        )
        
        E = data.merge(df_clase_grupo_decaimiento, on="Grupo", how='left')\
                .merge(caudal_ext, left_on="Clase referencia", right_on="Clase", how='left')\
                .groupby("INSTALACION").apply(lambda x: error(x, autocontrol, autocontrol_diametro_alto,  
                                                              error_ultra, intervalos_caudal_bajo, 
                                                              intervalos_caudal_bajo_c25, 1))
        E = E.reset_index(name=f'Error_{col_sufix + 0.5}')
        data = data.merge(E, on='INSTALACION', how='left')
    
    # Restore the original 'Clase corregida'
    # data['Clase corregida'] = data['Original Clase Corregida']
    data['Clase'] = data['Original Clase']
    data['Grupo'] = data['Original Grupo']
    
    # Optionally, drop the temporary column if no longer needed
    # data.drop(columns=['Original Clase Corregida'], inplace=True)
    data.drop(columns=['Original Clase'], inplace=True)
    data.drop(columns=['Original Grupo'], inplace=True)
    
    # Optionally, reset "Antiguedad ajustada" to the current year after all iterations are complete
    data["Antiguedad ajustada"] = calcular_antiguedad(data["FECHA_MONTAJE"],
                                                      fecha_estudio=fecha_estudio, 
                                                      edad_extra=edad_extra_decaimiento, 
                                                      edad_minima=edad_minima_decaimiento,
                                                      edad_maxima=edad_maxima_decaimiento)
    
    #----- SNIPPET END HERE ----

    #Columnas finales de BBDD
    data["V sub"]= -data["CONSUMO PROMEDIO"]*data["Error"]/(1+data["Error"])
    data["Precision"]= ~data['CLASE'].isin(medidores_presicion)
    data["DN"]= data['DIAMETRO_MEDIDOR'].apply(lambda x: '50+' if x>=50 else str(x))

    # Algunos ajustes de forma para la obtencion del output

    data["Precision"]= data["Precision"].astype(int)
    data["Diametro_max"]= data["Diametro_max"].astype(int)


    inicial_y_final=IyF(data, edad_inicial_iyf,edad_final_iyf, df_clase_grupo_decaimiento, caudal_ext, autocontrol, autocontrol_diametro_alto,  error_ultra, intervalos_caudal_bajo, intervalos_caudal_bajo_c25)

    inicial_y_final["Diametro_max"]=inicial_y_final["Diametro_max"].astype(int)
    inicial_y_final["Precision"]=inicial_y_final["Precision"].astype(int)

    resultados= resultados_loc_gen(data,inicial_y_final, fecha_estudio, edad_final_iyf, edad_inicial_iyf)
    
    data["FECHA_MONTAJE"] = data["FECHA_MONTAJE"].dt.strftime('%d/%m/%Y')

    output={"BBDD - Error Actual":data,
            "Resultados loc":resultados,
            "Inicial y final":inicial_y_final}

    return output

## Definición de funciones principales

#Aquí hay que agregar algo con respecto a si es ultra
#[TODO ULTRA]
def calcular_decaimiento(edad, problema_calidad_agua=False,caudal=None ):
    """Calcula decaimiento mediante regresion lineal entre el rendimiento y ln(años) para los datos de la tabla/pestaña edad.
    Ajusta si es que tienen problemas de calidad de agua.
    """

    out=edad.groupby("Clase").apply(lambda x:np.polyfit(np.log(x["Años"]), x["Rendimiento"], 1)).reset_index(name="Parametros")
    out[['Pendiente','Intercepto']] = pd.DataFrame(out['Parametros'].tolist(), index= out.index)
    if not(problema_calidad_agua):
        out = out.merge(caudal[caudal['Regimen']=="Bajo"], on='Clase')
        out['Pendiente']=out['Pendiente']/out['Porcentaje']
        
        
    #row fija para ULTRA:
    #Clase ULTRA, Parametros [None, None] Regimen Alto, Pendiente 0, Intercepto 0.02, porcentaje 0.07
    ultra_row = pd.DataFrame({
        "Clase": ["ULTRA"],
        "Parametros": [None],  # Or [np.nan, np.nan] to represent missing values
        "Regimen": ["Alto"],
        "Pendiente": [1],
        "Intercepto": [0.02],
        "Porcentaje": [0.07]
    })
    
    out = pd.concat([out,ultra_row], ignore_index=True)
    
    return out[["Clase",'Pendiente','Intercepto']]        


def calcular_antiguedad(fecha_montaje,fecha_estudio=pd.Timestamp.now(), edad_extra=0, edad_minima=0.5,edad_maxima=15):
    """Ajusta la antiguedad al rango entre edad minima y maxima, además agrega edad adicional en el caso que se ingrese el parametro"""
    edad=(fecha_estudio-fecha_montaje)/ (np.timedelta64(1, 'D')*365) #np.timedelta64(1, 'Y')
    return np.clip(edad+edad_extra,edad_minima, edad_maxima)

def calcular_antiguedad_proximos_n_anos(fecha_montaje, fecha_estudio, n, edad_extra=0, edad_minima=0.5, edad_maxima=15):
    """
    Calcula la antigüedad ajustada de un dispositivo para los próximos n años.
    
    Returns:
    - Lista de antigüedades ajustadas para cada uno de los próximos n años.
    """
    antiguedades_futuras = []
    for year in range(1, n + 1):
        # Calcular la fecha de estudio futuro incrementando el año
        fecha_estudio_futuro = fecha_estudio + pd.DateOffset(years=year)
        # Calcular la antigüedad ajustada para el año futuro
        antiguedad_ajustada = calcular_antiguedad(fecha_montaje, fecha_estudio_futuro, edad_extra, edad_minima, edad_maxima)
        antiguedades_futuras.append(antiguedad_ajustada)
    return antiguedades_futuras

def armar_clase(row, diametro_max, medidores_presicion):
    """Arma la clase y tambien el grupo correspondiente para cada medidor. Las clases corresponden a grupos de clases de medidores que se rigen 
    por el mismo rendimiento y decaimiento."""

    diametro_ajustado= str(row["DIAMETRO_MEDIDOR"]) if row["DIAMETRO_MEDIDOR"]<diametro_max else "25"
    if (row["DIAMETRO_MEDIDOR"]>diametro_max)  and (row["CLASE"] in medidores_presicion):
        return "ULTRA", "ULTRA"
    elif row['Clase corregida']!='ULTRA':

        clase_completa=row["Clase corregida"] + '-' + diametro_ajustado
        if clase_completa in ["A-13", "A-19", "B-13", "B-19"]:            
            return clase_completa, "AB-13/19"
        elif clase_completa in ['A-25', 'B-25']:
            return clase_completa, "AB-25"
        else:
            return clase_completa, clase_completa
    else:
        return "ULTRA", "ULTRA"


def seleccionar_autocontrol(df_autocontrol, localidad, tipo_error_CAlto):
    """Selecciona el parametro de autocontrol desde la pestaña correspondiente de excel"""

    row=df_autocontrol[df_autocontrol["LOCALIDAD"]==localidad].to_dict("records")[0]
    if tipo_error_CAlto=="Promedio":
        return row["Promedio Q %"]
    elif tipo_error_CAlto=="Q1":
        return row["Q1 sin outliers %"]
    elif tipo_error_CAlto=="Q2":
        return row["Q2 sin outliers %"]
    else: 
        raise Exception("Revisar")
        
def error(group,autocontrol, autocontrol_diametro_alto,  error_ultra, intervalos_caudal_bajo, intervalos_caudal_bajo_c25, flag=1):
    """Calcula el error de cada medidor tomando todas las consideraciones correspondientes."""

    diametro=group["Diametro_max"].tolist()[0]
    if flag == 1:    
        grupo = group["Grupo"].tolist()[0]
    else:
        grupo = group["Curva proy"].tolist()[0]
        
    if grupo=='ULTRA':
        return error_ultra
        
    if grupo =='C-25':
        intervalos_cb= intervalos_caudal_bajo_c25
    else:
        intervalos_cb= intervalos_caudal_bajo
        
    if diametro: 
        group['Decaimiento'] = group.apply(lambda x:x["Año 0"]+ x["Pendiente"]*np.log(x["Antiguedad ajustada"])/100  if x["Intervalo (l/h)"] in intervalos_cb else x["Año 0"] ,axis=1)
        return autocontrol*(1 - group['% Consumo'].sum())+ sum(group['% Consumo']*group['Decaimiento'])
    else:
        group['Decaimiento'] = group.apply(lambda x:x["Año 0"]+ x["Pendiente"]*np.log(x["Antiguedad ajustada"])/100  if x["Intervalo (l/h)"] in intervalos_cb else x["Año 0"] ,axis=1)
        return autocontrol_diametro_alto*(1 - group['% Consumo'].sum())+ sum(group['% Consumo']*group['Decaimiento'])

def IyF(data,
        edad_inicial_iyf,
        edad_final_iyf, 
        df_clase_grupo_decaimiento, 
        caudal_ext, 
        autocontrol, 
        autocontrol_diametro_alto,  
        error_ultra, 
        intervalos_caudal_bajo, 
        intervalos_caudal_bajo_c25):
    """Calcula la pestaña Inicial y Final, que corresponde a un analisis por tipo de medidor y sector del error inicial y final para cada configuracion."""

    # Variables generales (resumen agrupado de BBDD) 
    inicial_y_final = data.groupby(['LOCALIDAD', "SECTOR AP","Precision", 'DN', "Clase corregida", 'Grupo', 'Diametro_max']).agg(Cantidad=("INSTALACION", 'count'),
                                                                                                                Volumen=("CONSUMO PROMEDIO", 'sum')).reset_index()

    #Calculo de error inicial
    inicial_y_final['Antiguedad ajustada']=edad_inicial_iyf
    ei=inicial_y_final.merge(df_clase_grupo_decaimiento, on="Grupo", how='left').merge(caudal_ext,left_on="Clase referencia", right_on="Clase", how='left').groupby(['LOCALIDAD', "SECTOR AP","Precision", 'DN', "Clase corregida", 'Grupo', 'Diametro_max']).apply(lambda x: error(x,autocontrol, autocontrol_diametro_alto,  error_ultra, intervalos_caudal_bajo, intervalos_caudal_bajo_c25)).reset_index(name="Error inicial")
    inicial_y_final=inicial_y_final.rename(columns={'Antiguedad ajustada': 'Edad inicial'})

    #Calculo de error final
    inicial_y_final['Antiguedad ajustada']=edad_final_iyf
    ef=inicial_y_final.merge(df_clase_grupo_decaimiento, on="Grupo", how='left').merge(caudal_ext,left_on="Clase referencia", right_on="Clase", how='left').groupby(['LOCALIDAD', "SECTOR AP","Precision", 'DN', "Clase corregida", 'Grupo', 'Diametro_max']).apply(lambda x: error(x,autocontrol, autocontrol_diametro_alto,  error_ultra, intervalos_caudal_bajo, intervalos_caudal_bajo_c25)).reset_index(name="Error final")
    inicial_y_final=inicial_y_final.rename(columns={'Antiguedad ajustada': 'Edad final'})

    #Mezcla de todo en la misma tabla
    inicial_y_final=inicial_y_final.merge(ei,on= ['LOCALIDAD', "SECTOR AP","Precision", 'DN', "Clase corregida", 'Grupo', 'Diametro_max'])
    inicial_y_final=inicial_y_final.merge(ef,on= ['LOCALIDAD', "SECTOR AP","Precision", 'DN', "Clase corregida", 'Grupo', 'Diametro_max'])

    #Obtencion de volumen subestimado
    inicial_y_final["V_sub i"]= -inicial_y_final["Volumen"]*inicial_y_final["Error inicial"]/(1+inicial_y_final["Error inicial"])
    inicial_y_final["V_sub f"]= -inicial_y_final["Volumen"]*inicial_y_final["Error final"]/(1+inicial_y_final["Error final"])

    return inicial_y_final

def resultados_loc_gen(data,inicial_y_final, fecha_estudio, edad_final_iyf, edad_inicial_iyf):

    """Calcula pestaña Resultados Loc, que corresponde a un analisis de las caracteristicas principales agrupadas por sector. Tambien se encuentran
    los resultados agregados por localidad al final de la tabla."""
    
    # Calculo de datos por localidad y sector (resumen de BBDD)
    resultados_loc=data.groupby(["LOCALIDAD",'SECTOR AP']).agg(N_medidores=("INSTALACION", 'count'),
                                                            Edad_media=("FECHA_MONTAJE", lambda x: np.mean(fecha_estudio-pd.to_datetime(x))/ (np.timedelta64(1, 'D')*365)), #np.timedelta64(1, 'Y')),
                                                            Total_consumo_medio=("CONSUMO PROMEDIO", 'sum'),
                                                            V_sub_act=("V sub", "sum")).reset_index()

    # Calculo de datos por localidad y sector (resumen de Inicial y Final)
    resultados_loc=resultados_loc.merge(inicial_y_final.groupby(["LOCALIDAD",'SECTOR AP']).agg(V_sub_0=("V_sub i", "sum"),
                                                        V_sub_final=("V_sub f", "sum")).reset_index(), on= ["LOCALIDAD",'SECTOR AP'])

    # Calculo de errores
    resultados_loc["Error actual"]= -resultados_loc["V_sub_act"]/(resultados_loc["Total_consumo_medio"]+resultados_loc["V_sub_act"])
    resultados_loc["Error inicial"]= -resultados_loc["V_sub_0"]/(resultados_loc["Total_consumo_medio"]+resultados_loc["V_sub_0"])
    resultados_loc["Error final"]= -resultados_loc["V_sub_final"]/(resultados_loc["Total_consumo_medio"]+resultados_loc["V_sub_final"])
    #Calculo de tasa de decaimiento
    resultados_loc["Tasa decaimiento"]= resultados_loc["Error inicial"] - resultados_loc["Error final"]

    # Calculo de datos por localidad (lo mismo que por sector pero sin agrupar por localidad)

    resultados_gen=data.groupby(["LOCALIDAD"]).agg(N_medidores=("INSTALACION", 'count'),
                                                            Total_consumo_medio=("CONSUMO PROMEDIO", 'sum'),
                                                            Edad_media=("FECHA_MONTAJE", lambda x: np.mean(fecha_estudio-x)/ (np.timedelta64(1, 'D')*365) ), #np.timedelta64(1, 'Y')),
                                                                V_sub_act=("V sub", "sum")).reset_index()
    resultados_gen=resultados_gen.merge(inicial_y_final.groupby(["LOCALIDAD"]).agg(V_sub_0=("V_sub i", "sum"),
                                                        V_sub_final=("V_sub f", "sum")).reset_index(), on= ["LOCALIDAD"])

    resultados_gen["Error actual"]= -resultados_gen["V_sub_act"]/(resultados_gen["Total_consumo_medio"]+resultados_gen["V_sub_act"])
    resultados_gen["Error inicial"]= -resultados_gen["V_sub_0"]/(resultados_gen["Total_consumo_medio"]+resultados_gen["V_sub_0"])
    resultados_gen["Error final"]= -resultados_gen["V_sub_final"]/(resultados_gen["Total_consumo_medio"]+resultados_gen["V_sub_final"])

    resultados_gen["Tasa decaimiento"]= (resultados_gen["Error inicial"] - resultados_gen["Error final"])/(edad_final_iyf- edad_inicial_iyf)


    # obtencion de resultados en una sola tabla (como se encuentra actualmente en datos originales)
    resultados=pd.concat([resultados_loc,resultados_gen])

    return resultados

def armar_curva_proyectada(df):
    '''
    requerimientos:
        columna 'DIAMETRO'.
        columna 'CLASE CORREGIDA'
    
    # Usage example
    # Assuming you have a DataFrame 'df' with the necessary columns already in it
    # df = pd.read_excel('path_to_your_excel_file.xlsx')  # Loading data for example
    # df = armar_curva_proyectada(df)  # Applying the function
    # print(df[['DIAMETRO', 'CLASE CORREGIDA', 'new_column']])
    '''
    col_diametro = "DIAMETRO_MEDIDOR"
    col_clase_corregida = "Clase"
    col_salida = 'Curva proy'
    
    # Define conditions
    conditions = [
        (df[col_clase_corregida] == "ULTRA"),
        df[col_diametro] == 25,
        df[col_diametro] < 25,
        df[col_diametro] > 25,
    ]
    
    # Define corresponding choices
    choices = [
        "ULTRA",
        "ULTRA",  
        "C-" + df[col_diametro].astype(str),
        # df["Grupo"].astype(str) #este sería ultra?
        "ULTRA",
    ]
    
    # Apply the select function
    df[col_salida] = np.select(conditions, choices, default="")
    
    return df

# if __name__ == "__main__": 
#     exceldata=pd.read_excel(direccion_excel, sheet_name=None)
#     salida = mm_calc_run_alg(exceldata)
    
#     #Grabar la salida
#     with pd.ExcelWriter(direccion_excel_salida, engine='xlsxwriter') as writer:
#         for sheet_name, df in salida.items():
#             df.to_excel(writer, sheet_name=sheet_name, index=False)

# Helper function to ensure worksheet names are 31 characters or fewer
def shorten_sheet_name(name, max_length=31):
    if len(name) > max_length:
        return name[:max_length - 3] + "..."  # Truncate and add '...' for indication
    return name

def read_parameters_from_excel_old(exceldata, sheet_name='PARAMETROS'):
    df_params = exceldata[sheet_name]
    
    parametros_localidad = {
        "fecha_estudio": pd.to_datetime(df_params.loc[0, 'Valor']) if not pd.isna(df_params.loc[0, 'Valor']) else pd.Timestamp.now(),
        "localidad": df_params.loc[1, 'Valor'],
        "sectores": df_params['Sectores'].dropna().tolist(),
        "tipo_error_CAlto": df_params.loc[4, 'Valor'],
        "edad_minima_decaimiento": df_params.loc[5, 'Valor'],
        "edad_maxima_decaimiento": df_params.loc[6, 'Valor'],
        "edad_extra_decaimiento": df_params.loc[7, 'Valor'],
        "problema_calidad_agua": (df_params.loc[8, 'Valor']).upper() == 'SI',
        "autocontrol_diametro_alto": (df_params.loc[2, 'Valor'] + df_params.loc[3, 'Valor']) / 2,
        "edad_inicial_iyf": df_params.loc[9, 'Valor'],
        "edad_final_iyf": df_params.loc[10, 'Valor'],
    }
    
    return parametros_localidad

def read_parameters_from_excel(exceldata, localidad, sheet_name='PARAMETROS POR LOCALIDAD'):
    # Get the data from the specified sheet
    df_params = exceldata[sheet_name]
    
    # Find the row that corresponds to the specified 'Localidad'
    localidad_row = df_params[df_params['Localidad'] == localidad]
    
    if localidad_row.empty:
        raise ValueError(f"Localidad '{localidad}' not found in {sheet_name}.")

    # Extract the parameters from the identified row
    parametros_localidad = {
        "fecha_estudio": pd.to_datetime(localidad_row['Fecha estudio'].values[0]),
        "localidad": localidad,
        "sectores": df_params[df_params['Localidad.1'] == localidad]['Sector'].dropna().tolist(),
        "tipo_error_CAlto": localidad_row['Error caudal alto < 38mm'].values[0],
        "edad_minima_decaimiento": localidad_row['Edad minima decaimiento'].values[0],
        "edad_maxima_decaimiento": localidad_row['Edad máxima decaimiento'].values[0],
        "edad_extra_decaimiento": localidad_row['Edad extra decaimiento'].values[0],
        "problema_calidad_agua": (localidad_row['Problema Calidad de Agua'].values[0]).upper() == 'SI',
        "autocontrol_diametro_alto": (localidad_row['Q1 >=38 mm'].values[0] + localidad_row['Q2 >=38 mm'].values[0]) / 2,
        "edad_inicial_iyf": localidad_row['Edad inicial (estudio inicial y final)'].values[0],
        "edad_final_iyf": localidad_row['Edad final (estudio inicial y final)'].values[0],
    }
    
    return parametros_localidad

def get_localidades_from_excel(exceldata, sheet_name='PARAMETROS POR LOCALIDAD'):
    # Get the data from the specified sheet
    df_params = exceldata[sheet_name]
    
    # Extract the unique 'Localidad' values and return them as a list
    localidades = df_params['Localidad'].dropna().unique().tolist()
    
    return localidades


# if __name__ == "__main__": 
    
#     # localidad = "CURICO"
#     exceldata=pd.read_excel(direccion_excel, sheet_name=None)
#     salida = mm_calc_run_alg_all(exceldata)
    
#     # localidad_parameters = read_parameters_from_excel(exceldata, localidad)
#     # salida = mm_calc_run_alg(exceldata, localidad_parameters)
    
#     #Grabar la salida
#     with pd.ExcelWriter(direccion_excel_salida, engine='xlsxwriter') as writer:
#         for sheet_name, df in salida.items():
#             df.to_excel(writer, sheet_name=sheet_name, index=False)



