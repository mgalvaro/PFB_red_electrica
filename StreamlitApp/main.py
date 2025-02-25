from datetime import datetime
import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

from config import PAGE_CONFIG

from supabase import create_client

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.funciones.extraccion_balance import extrae_balance
from src.funciones.extraccion_demanda import extrae_demanda
from src.funciones.extraccion_generacion import extrae_generacion
from src.funciones.extraccion_intercambios import extrae_intercambios

from src.funciones.divide_fecha import divide_fecha
from src.funciones.limpia_columnas import limpia_columnas

from src.funciones.limpia_demanda import limpia_demanda
from src.funciones.limpia_balance import limpia_balance
from src.funciones.limpia_generacion import limpia_generacion
from src.funciones.limpia_intercambio import limpia_intercambio

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
def get_ultima_extraccion(tabla) -> str:

    supabase = init_connection()
    response = (
        supabase.table(tabla)
        .select("fecha_extraccion")  
        .order("fecha_extraccion", desc=True)  
        .limit(1)  
        .execute()
    )
    if response.data:
        return response.data[0]["fecha_extraccion"]  # Devuelve la fecha más reciente
    else:
        return None 

@st.cache_data
def run_query() -> tuple:

    df_demanda = pd.read_csv('data/processed/DF_DEMANDA_10_25_LIMPIO.csv')
    #df_demanda = df_demanda[df_demanda['titulo'] == 'Demanda']
    df_generacion = pd.read_csv('data/processed/DF_GENERACION_10_25_LIMPIO.csv')
    df_intercambios = pd.read_csv('data/processed/DF_INTERCAMBIOS_10_25_LIMPIO.csv')
    df_balance = pd.read_csv('data/processed/DF_BALANCE_10_25_LIMPIO.csv')

    hoy = datetime.now()

    ultimas_extracciones = {}
    #supabase = init_connection()
    tablas = ["demanda", "generacion", "intercambios", "balance"]
    
    for tabla in tablas:

        ultimas_extracciones[tabla] = get_ultima_extraccion(tabla)

    st.write(f"Ultimas fechas: {ultimas_extracciones}")

    fechas_faltantes = {}

    df_dict = {
        "demanda": df_demanda,
        "generacion": df_generacion,
        "intercambios": df_intercambios,
        "balance": df_balance,
    }

    for tabla, ultima_extraccion in ultimas_extracciones.items():
        dias_faltantes = []
        st.write(f"tabla: {tabla}, ultima extraccion: {ultima_extraccion}")
        
        ultima_extraccion = pd.to_datetime(df_dict[tabla]["fecha_extraccion"]).max()
        st.write(f"ultima extraccion: {ultima_extraccion}")
        while ultima_extraccion < hoy:
            ultima_extraccion += pd.Timedelta(days=1)
            if ultima_extraccion <= hoy:
                dias_faltantes.append(ultima_extraccion.strftime('%Y-%m-%d'))
            
        fechas_faltantes[tabla] = dias_faltantes
        
    st.write(f"fechas faltantes: {fechas_faltantes}")
            
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

def main():

    st.write("main")
    hoy = datetime.now().strftime("%Y-%m-%d")

    df_demanda, df_generacion, df_intercambios, df_balance, fechas_faltantes = run_query()

    st.write(f"{fechas_faltantes}")
    # Proceso de extracción y actualización de datos
    if fechas_faltantes:
        st.write("Procesando fechas faltantes...")

        for tabla, fechas in fechas_faltantes.items():
            for fecha in fechas:
                st.write(f"Extrayendo datos para la tabla '{tabla}' desde la fecha: {fecha}")

                if tabla == "demanda":
                    nuevos_datos_demanda = extrae_demanda(fecha, hoy)
                    nuevos_datos_demanda = divide_fecha(nuevos_datos_demanda)
                    nuevos_datos_demanda = limpia_columnas(nuevos_datos_demanda)
                    nuevos_datos_demanda = limpia_demanda(nuevos_datos_demanda)

                    if len(nuevos_datos_demanda) > 0:
                        st.write(f"Agregando {len(nuevos_datos_demanda)} registros a la tabla '{tabla}'...")
                        agregar_datos_supabase(tabla, nuevos_datos_demanda)
                        df_demanda = pd.concat([df_demanda, nuevos_datos_demanda]).drop_duplicates()
                    else:
                        st.write(f"No se encontraron nuevos datos para la tabla '{tabla}'.")

                elif tabla == "generacion":
                    nuevos_datos_generacion = extrae_generacion(fecha, hoy)
                    nuevos_datos_generacion = divide_fecha(nuevos_datos_generacion)
                    nuevos_datos_generacion = limpia_columnas(nuevos_datos_generacion)
                    nuevos_datos_generacion = limpia_generacion(nuevos_datos_generacion)

                    if len(nuevos_datos_generacion) > 0:
                        st.write(f"Agregando {len(nuevos_datos_generacion)} registros a la tabla '{tabla}'...")
                        agregar_datos_supabase(tabla, nuevos_datos_generacion)
                        df_generacion = pd.concat([df_generacion, nuevos_datos_generacion]).drop_duplicates()
                    else:
                        st.write(f"No se encontraron nuevos datos para la tabla '{tabla}'.")

                elif tabla == "intercambios":
                    nuevos_datos_intercambios = extrae_intercambios(fecha, hoy)
                    nuevos_datos_intercambios = divide_fecha(nuevos_datos_intercambios)
                    nuevos_datos_intercambios = limpia_columnas(nuevos_datos_intercambios)
                    nuevos_datos_intercambios = limpia_intercambio(nuevos_datos_intercambios)

                    if len(nuevos_datos_intercambios) > 0:
                        st.write(f"Agregando {len(nuevos_datos_intercambios)} registros a la tabla '{tabla}'...")
                        agregar_datos_supabase(tabla, nuevos_datos_intercambios)
                        df_intercambios = pd.concat([df_intercambios, nuevos_datos_intercambios]).drop_duplicates()
                    else:
                        st.write(f"No se encontraron nuevos datos para la tabla '{tabla}'.")

                elif tabla == "balance":
                    nuevos_datos_balance = extrae_balance(fecha, hoy)
                    nuevos_datos_balance = divide_fecha(nuevos_datos_balance)
                    nuevos_datos_balance = limpia_columnas(nuevos_datos_balance)
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