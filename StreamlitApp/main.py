from datetime import datetime
import streamlit as st
import pandas as pd

from config import PAGE_CONFIG

from vis_demanda import vis_demanda
from vis_intercambios_mapa import vis_intercambios
from vis_compare_years import vis_compare
from vis_generacion import vis_generacion
from functions.carga_dataframes import *

from passwords import pw

def main():
    st.set_page_config(page_title="APP de datos REE", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
    tab1, tab2, tab3 = st.tabs(["INICIO", "EDA", "ML"])

    with tab1:
        st.markdown('### :zap: Bienvenido a la App de datos de la REE :zap:')
            
    with tab2:

        try:
            df_balance, df_demanda, df_generacion, df_intercambios = carga_dataframes(pw["host"], pw["user"], pw["password"], pw["database"])
            df_demanda = df_demanda[df_demanda['titulo'] == 'Demanda']
            if not df_demanda.empty and not df_generacion.empty and not df_intercambios.empty:
                    st.success(":heavy_check_mark: Datos cargados con éxito")
        except Exception:
            st.error(":exclamation: Error al cargar los datos")
    
        menu = ['Serie Temporal Demanda', 'Mapa de Intercambios', 'Comparación Anual', 'Generación por tecnología']

        choice = st.selectbox(label='Menu', options=menu, index=0)

        if choice == 'Serie Temporal Demanda':
            vis_demanda(df_demanda)

        elif choice == 'Mapa de Intercambios':
            vis_intercambios(df_intercambios)

        elif choice == 'Comparación Anual':
            vis_compare(df_demanda, df_generacion, df_intercambios)

        elif choice == 'Generación por tecnología':
            vis_generacion(df_generacion)
        
        else:
            pass
    
    with tab3:
        st.title("EN CONSTRUCCIÓN...")
        
if __name__ == '__main__':
    main()