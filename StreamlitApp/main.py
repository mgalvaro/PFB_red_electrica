from datetime import datetime
import streamlit as st
import pandas as pd
import os
import sys
from config import PAGE_CONFIG

# Agregar el directorio raíz al sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(BASE_DIR)

from supabase import create_client

from funciones.extraccion_balance import extrae_balance
from funciones.extraccion_demanda import extrae_demanda
from funciones.extraccion_generacion import extrae_generacion
from funciones.extraccion_intercambios import extrae_intercambios

from funciones.limpia_demanda import limpia_demanda
from funciones.limpia_balance import limpia_balance
from funciones.limpia_generacion import limpia_generacion
from funciones.limpia_intercambio import limpia_intercambio

from vis_demanda import vis_demanda
from vis_intercambios_mapa import vis_intercambios
from vis_compare_years import vis_compare
from vis_generacion import vis_generacion

st.set_page_config(**PAGE_CONFIG)

def init_connection() -> tuple:
    url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
    key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

# Obtener la última fecha
def get_latest_fecha(tabla) -> str:
    supabase = init_connection()
    response = (
        supabase.table(tabla)
        .select("fecha")  
        .order("fecha", desc=True)  
        .limit(1)  
        .execute()
    )
    if response.data:
        return response.data[0]["fecha"]  # Devuelve la fecha más reciente
    else:
        return None 

@st.cache_data
def run_query() -> tuple:

    df_demanda = pd.read_csv('data/processed/DF_DEMANDA_10_25_LIMPIO.csv')
    #df_demanda = df_demanda[df_demanda['titulo'] == 'Demanda']
    df_generacion = pd.read_csv('data/processed/DF_GENERACION_10_25_LIMPIO.csv')
    df_intercambios = pd.read_csv('data/processed/DF_INTERCAMBIOS_10_25_LIMPIO.csv')
    df_balance = pd.read_csv('data/processed/DF_BALANCE_10_25_LIMPIO.csv')

    ultimas_fechas = {}
    #supabase = init_connection()
    tablas = ["demanda", "generacion", "intercambios", "balance"]
    
    for tabla in tablas:

        ultimas_fechas[tabla] = get_latest_fecha(tabla)

    fechas_faltantes = {}

    df_dict = {
        "demanda": df_demanda,
        "generacion": df_generacion,
        "intercambios": df_intercambios,
        "balance": df_balance,
    }

    for tabla, ultima_fecha in ultimas_fechas.items():
        if ultima_fecha:
            ultima_fecha_csv = pd.to_datetime(df_dict[tabla]["fecha"]).max()
            if pd.to_datetime(ultima_fecha) > ultima_fecha_csv:
                fechas_faltantes[tabla] = ultima_fecha_csv + pd.Timedelta(days=1)

    return df_demanda, df_generacion, df_intercambios, df_balance, fechas_faltantes

def agregar_datos_supabase(tabla, datos) -> dict:

    supabase = init_connection()
    registros = datos.to_dict(orient="records")  # Convierte el DataFrame a una lista de diccionarios
    
    try:
        response = supabase.table(tabla).insert(registros).execute()
        if response.status_code == 201:
            st.success(f"Datos agregados correctamente a la tabla '{tabla}'.")
        else:
            st.error(f"Error al agregar datos a '{tabla}': {response.status_code} - {response.json()}")
        return response
    except Exception as e:
        st.error(f"Error al agregar datos a '{tabla}': {str(e)}")
        return None

    ''' # Ejecuta las consultas
    actualiza_demanda = supabase.table("demanda").select("*").eq("titulo", "Demanda").execute()
    actualiza_generacion = supabase.table("generacion").select("*").execute()
    actualiza_intercambios = supabase.table("intercambios").select("*").execute()
    actualiza_balance = supabase.table("balance").select("*").execute()
    
    # Accede a los datos en el atributo `.data`
    demanda_data = demanda.data if demanda and demanda.data else None
    generacion_data = generacion.data if generacion and generacion.data else None
    intercambios_data = intercambios.data if intercambios and intercambios.data else None
    balance_data = balance.data if balance and balance.data else None

    if demanda_data and generacion_data and intercambios_data and balance_data:
        return demanda_data, generacion_data, intercambios_data, balance_data
    else:
        return None, None, None, None'''


def main():

    hoy = datetime.now().strftime("%Y-%m-%D")

    #supabase = init_connection()

    df_demanda, df_generacion, df_intercambios, df_balance, fechas_faltantes = run_query()

    # Proceso de extracción y actualización de datos
    if fechas_faltantes:
        st.write("Procesando fechas faltantes...")

        for tabla, fecha in fechas_faltantes.items():
            st.write(f"Extrayendo datos para la tabla '{tabla}' desde la fecha: {fecha}")

            if tabla == "demanda":
                nuevos_datos_demanda = extrae_demanda(fecha, hoy)
                nuevos_datos_demanda = limpia_demanda(nuevos_datos_demanda)

                if len(nuevos_datos_demanda) > 0:
                    st.write(f"Agregando {len(nuevos_datos_demanda)} registros a la tabla '{tabla}'...")
                    agregar_datos_supabase(tabla, nuevos_datos_demanda)
                    df_demanda = pd.concat([df_demanda, nuevos_datos_demanda]).drop_duplicates()
                else:
                    st.write(f"No se encontraron nuevos datos para la tabla '{tabla}'.")
            elif tabla == "generacion":
                nuevos_datos_generacion = extrae_generacion(fecha, hoy)
                nuevos_datos_generacion = limpia_generacion(nuevos_datos_generacion)

                if len(nuevos_datos_generacion) > 0:
                    st.write(f"Agregando {len(nuevos_datos_generacion)} registros a la tabla '{tabla}'...")
                    agregar_datos_supabase(tabla, nuevos_datos_generacion)
                    df_generacion = pd.concat([df_generacion, nuevos_datos_generacion]).drop_duplicates()
                else:
                    st.write(f"No se encontraron nuevos datos para la tabla '{tabla}'.")
            elif tabla == "intercambios":
                nuevos_datos_intercambios = extrae_intercambios(fecha, hoy)
                nuevos_datos_intercambios = limpia_intercambio(nuevos_datos_intercambios)

                if len(nuevos_datos_intercambios) > 0:
                    st.write(f"Agregando {len(nuevos_datos_intercambios)} registros a la tabla '{tabla}'...")
                    agregar_datos_supabase(tabla, nuevos_datos_intercambios)
                    df_intercambios = pd.concat([df_intercambios, nuevos_datos_intercambios]).drop_duplicates()
                else:
                    st.write(f"No se encontraron nuevos datos para la tabla '{tabla}'.")
            elif tabla == "balance":
                nuevos_datos_balance = extrae_balance(fecha, hoy)
                nuevos_datos_balance = limpia_balance(nuevos_datos_balance)

                if len(nuevos_datos_balance) > 0:
                    st.write(f"Agregando {len(nuevos_datos_balance)} registros a la tabla '{tabla}'...")
                    agregar_datos_supabase(tabla, nuevos_datos_balance)
                    df_balance = pd.concat([nuevos_datos_balance, df_balance]).drop_duplicates()
                else:
                    st.write(f"No se encontraron nuevos datos para la tabla '{tabla}'.")
    else:
        st.write("No hay datos faltantes para procesar.")
    
    df_demanda = df_demanda[df_demanda['titulo'] == 'Demanda']

    menu = ['Inicio', 'Serie Temporal Demanda', 'Mapa de Intercambios', 'Comparación Anual', 'Generación por tecnología']

    choice = st.sidebar.selectbox(label='Menu', options=menu, index=0)

    if choice == 'Inicio':
        st.markdown('### :zap: Bienvenido a la App de datos de la REE :zap:')
        if len(df_demanda) and len(df_generacion) and len(df_intercambios) and len(df_balance) > 0:
            st.success(":heavy_check_mark: Datos cargados con éxito")
        else:
            st.error(":exclamation: Error al cargar los datos")

    elif choice == 'Serie Temporal Demanda':
        vis_demanda(df_demanda)

    elif choice == 'Mapa de Intercambios':
        vis_intercambios(df_intercambios)

    elif choice == 'Comparación Anual':
        vis_compare(df_demanda, df_generacion, df_intercambios)

    elif choice == 'Generación por tecnología':
        vis_generacion(df_generacion)
    
    else:
        pass

if __name__ == '__main__':
    main()