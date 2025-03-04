from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import PAGE_CONFIG

from vis_demanda import vis_demanda
from vis_intercambios_mapa import vis_intercambios
from vis_compare_years import vis_compare
from vis_generacion import vis_generacion
from functions.carga_dataframes import *

from passwords import pw

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from ML.gru_rnn import *

def main():
    st.set_page_config(page_title="APP de datos REE", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
    tab1, tab2, tab3 = st.tabs(["INICIO", "EDA", "ML"])

    try:
        df_balance, df_demanda, df_generacion, df_intercambios = carga_dataframes(pw["host"], pw["user"], pw["password"], pw["database"])
    except Exception:
        st.error(":exclamation: Error al cargar los datos")

    with tab1:
        st.markdown('### :zap: Bienvenido a la App de datos de la REE :zap:')
            
    with tab2:

        df_demanda = df_demanda[df_demanda['titulo'] == 'Demanda']
            
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
        
        menu = ['Deep Learning', 'Gated Recurrent Unit (GRU)', 'Facebook Propeth']

        choice = st.selectbox(label='Modelos de predicción de demanda', options=menu, index=None, placeholder="Seleccione modelo ML")

        if choice == 'Deep Learning':
            st.header("En construcción...")

        elif choice == 'Gated Recurrent Unit (GRU)':
            vis_gru(df_demanda)

        elif choice == 'Facebook Propeth':
            st.header("En construcción...")
        
        else:
            pass
        
if __name__ == '__main__':
    main()