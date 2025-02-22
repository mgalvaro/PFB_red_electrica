import streamlit as st
import pandas as pd
from config import PAGE_CONFIG
from vis_demanda import vis_demanda
from vis_intercambios_mapa import vis_intercambios
from vis_compare_years import vis_compare
from vis_generacion import vis_generacion

st.set_page_config(**PAGE_CONFIG)

def main():

    df_demanda = pd.read_csv('../data/processed/DF_DEMANDA_10_25_LIMPIO.csv')
    df_demanda = df_demanda[df_demanda['titulo'] == 'Demanda']
    df_generacion = pd.read_csv('../data/processed/DF_GENERACION_10_25_LIMPIO.csv')
    df_intercambios = pd.read_csv('../data/processed/DF_INTERCAMBIOS_10_25_LIMPIO.csv')

    menu = ['Inicio', 'Serie Temporal Demanda', 'Mapa de Intercambios', 'Comparación Anual', 'Generación por tecnología']

    choice = st.sidebar.selectbox(label='Menu', options=menu, index=0)

    if choice == 'Inicio':
        st.markdown('### :zap: Bienvenido a la App de datos de la REE :zap:')
        if not df_demanda.empty and not df_generacion.empty and not df_intercambios.empty:
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