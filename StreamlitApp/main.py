import streamlit as st
import pandas as pd
import os

from config import PAGE_CONFIG

from vis_demanda import vis_demanda
from vis_intercambios_mapa import vis_intercambios
from vis_compare_years import vis_compare
from vis_generacion import vis_generacion

st.set_page_config(**PAGE_CONFIG)

def init_connection():
    from supabase import create_client
    url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
    key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

@st.cache_data
def run_query():
    supabase = init_connection()
    
    # Ejecuta las consultas
    demanda = supabase.table("demanda").select("*").eq("titulo", "Demanda").execute()
    generacion = supabase.table("generacion").select("*").execute()
    intercambios = supabase.table("intercambios").select("*").execute()
    balance = supabase.table("balance").select("*").execute()
    
    # Accede a los datos en el atributo `.data`
    demanda_data = demanda.data if demanda and demanda.data else None
    generacion_data = generacion.data if generacion and generacion.data else None
    intercambios_data = intercambios.data if intercambios and intercambios.data else None
    balance_data = balance.data if balance and balance.data else None

    if demanda_data and generacion_data and intercambios_data and balance_data:
        return demanda_data, generacion_data, intercambios_data, balance_data
    else:
        return None, None, None, None


def main():

    # st.write("Directorio actual:", os.getcwd())
    # st.write("Archivos en el directorio actual:", os.listdir("."))
    # st.write("Archivos en '../data/processed':", os.listdir("../data/processed") if os.path.exists("../data/processed") else "No encontrado")
    
    '''df_demanda = pd.read_csv('data/processed/DF_DEMANDA_10_25_LIMPIO.csv')
    df_demanda = df_demanda[df_demanda['titulo'] == 'Demanda']
    df_generacion = pd.read_csv('data/processed/DF_GENERACION_10_25_LIMPIO.csv')
    df_intercambios = pd.read_csv('data/processed/DF_INTERCAMBIOS_10_25_LIMPIO.csv')'''

    df_demanda, df_generacion, df_intercambios, df_balance = run_query()

    st.write(len(df_demanda))
    st.write(len(df_generacion))
    st.write(len(df_intercambios))
    st.write(len(df_balance))


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