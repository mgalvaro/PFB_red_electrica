import streamlit as st

import os

import pandas as pd
import plotly.express as px

from functions.filtros_visualizaciones import *

#_____________________


def vis_demanda(df):

    # 1ª gráfica: Serie temporal de Demanda
    # st.write(os.getcwd())

    periodos_dict = {

        "Últimos 7 días": 7,
        "Últimos 30 días": 30,
        "Últimos 365 días": 365,
        "Histórico": -1 
    }

    periodo = st.radio(label = "Selecciona el período",
                       options = list(periodos_dict.keys()),
                       index = 0,
                       disabled = False,
                       horizontal = True,)
    
    periodo_seleccionado = periodos_dict[periodo]

    df_filtered = p7_30_365_hist(df, periodo_seleccionado)[0]
    periodo_seleccionado = p7_30_365_hist(df, periodo_seleccionado)[1]

    p_selec = 'días' if periodo_seleccionado > 0 else 'años'
    

    with st.expander(label = f"DataFrame Filtrado para {periodo_seleccionado} {p_selec}", expanded = False):
        st.dataframe(df_filtered)
        
    fig = px.line(df_filtered, 
                  x = "fecha", 
                  y = "valor_(MWh)", 
                  color='zona', 
                  title = f"Demanda eléctrica en España de los últimos {periodo_seleccionado} días",
                  labels={'fecha': "Fecha", 'valor_(MWh)': "Demanda (MWh)", 'zona': "Zona"}
                  )
    
    st.plotly_chart(fig)

    # _________________________________________________________________________________________________________________________

    
    


if __name__ == "__main__":
    vis_demanda()