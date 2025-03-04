import os
import pickle
import pandas as pd
import numpy as np
import calendar
from holidays import Spain
from sklearn.preprocessing import MinMaxScaler

# Definir rutas de los datos
ruta_datos = "../data/processed"
ruta_datos_escalados = "../data/data_scaled"
ruta_scalers = os.path.join(ruta_datos_escalados, "scalers")

# Crear carpetas si no existen
def crear_carpetas():
    os.makedirs(ruta_datos_escalados, exist_ok=True)
    os.makedirs(ruta_scalers, exist_ok=True)

# Función para cargar y filtrar datos
def cargar_y_filtrar_datos(df_demanda):
    #ruta_archivo = os.path.join(ruta_datos, nombre_archivo)
    #df_demanda = pd.read_csv(ruta_archivo)

    # Filtrar solo la zona nacional y la demanda
    df = df_demanda[(df_demanda['zona'] == 'nacional') & (df_demanda['titulo'] == 'Demanda')]
    df = df[['valor_(MWh)', 'fecha', 'año', 'mes', 'dia', 'dia_semana']].reset_index(drop=True)

    # Convertir MWh a GWh
    df['valor_(GWh)'] = df['valor_(MWh)'] * 0.001
    df.drop(columns=['valor_(MWh)'], inplace=True)

    return df

# Función para aplicar codificación circular a los días de la semana
def codificar_dia_semana(df):
    df["dia_semana"] = df["dia_semana"].map({
        "lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3, "viernes": 4, "sábado": 5, "domingo": 6
    })
    df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)
    df.drop(columns=["dia_semana"], inplace=True)

    return df

# Función para codificar el mes en formato circular
def codificar_mes(df):
    df["mes_sin"] = np.sin(2 * np.pi * (df["mes"] - 1) / 12)
    df["mes_cos"] = np.cos(2 * np.pi * (df["mes"] - 1) / 12)
    df.drop(columns=["mes"], inplace=True)

    return df

# Función para codificar el día del mes de forma circular considerando la cantidad de días del mes
def codificar_dia_mes(df):
    def normalizar_dia(row):
        # Convertir a enteros para evitar problemas con calendar
        year = int(row["año"])
        month = int(row["mes"])
        dias_del_mes = calendar.monthrange(year, month)[1]  # Obtener el número de días en el mes
        return row["dia"] / dias_del_mes

    df["dia_normalizado"] = df.apply(normalizar_dia, axis=1)
    df["dia_sin"] = np.sin(2 * np.pi * df["dia_normalizado"])
    df["dia_cos"] = np.cos(2 * np.pi * df["dia_normalizado"])
    
    df.drop(columns=["dia", "dia_normalizado"], inplace=True)

    return df

# Función para escalar solo el consumo de energía y el año
def escalar_consumo_anio(df, nombre_scaler):
    minmax_scaler = MinMaxScaler()
    df[["valor_(GWh)", "año"]] = minmax_scaler.fit_transform(df[["valor_(GWh)", "año"]])

    # Guardar el escalador
    ruta_scaler_minmax = os.path.join(ruta_scalers, f"{nombre_scaler}.pkl")
    with open(ruta_scaler_minmax, "wb") as f:
        pickle.dump(minmax_scaler, f)

    return df

# Función principal de procesamiento
def procesar_datos(dataframe):
    #crear_carpetas()
    
    #nombre_archivo = "DF_DEMANDA_10_25_LIMPIO.csv"
    df = cargar_y_filtrar_datos(dataframe)

    festivos = Spain(years=df["año"].unique())

    df["fecha"] = df["fecha"].astype("datetime64[ns]")
    
    # Verificar si la fecha está en festivos
    df["es_festivo"] = df["fecha"].apply(lambda x: 1 if x in festivos else 0)

    print("Columnas antes de procesar:", df.columns)  # Verifica que 'dia' está presente

    df = codificar_dia_semana(df)

    # Guardar "mes" y "dia" antes de eliminarlos
    df["mes_original"] = df["mes"]
    df["dia_original"] = df["dia"]

    df = codificar_dia_mes(df)  # Se ejecuta antes de eliminar "dia"
    df = codificar_mes(df)      # Se ejecuta después de haber usado "mes"

    #  Construcción de la fecha usando valores originales antes del escalado
    df["fecha"] = pd.to_datetime(df["año"].astype(str) + "-" +
                                 df["mes_original"].astype(str).str.zfill(2) + "-" +
                                 df["dia_original"].astype(str).str.zfill(2))
    
    #  Ahora sí escalamos el consumo y el año
    df = escalar_consumo_anio(df, "scaler_consumo_anio_DF_DEMANDA")

    #  Ya no necesitamos mes_original y dia_original
    df.drop(columns=["mes_original", "dia_original"], inplace=True)

    # Guardar datos procesados
    ruta_archivo_final = os.path.join(ruta_datos_escalados, "DF_DEMANDA_10_25_PROCESADO.csv")
    df.to_csv(ruta_archivo_final, index=False)

    print("Procesamiento completado. Datos preparados y guardados.")

    return df

# Ejecutar script
#if __name__ == "__main__":
#    procesar_datos()