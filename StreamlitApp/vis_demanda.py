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

        "Últimas 24 horas": 1,
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

    df_filtered = p7_30_365_hist(df, periodo_seleccionado)
    
    # with st.expander(label = f"DataFrame Filtrado", expanded = False):
    #     st.dataframe(df_filtered)
            
    if periodo_seleccionado != -1:

        if periodo_seleccionado == 365:
            p_selec = f'último año (media semanal)'
            df_filtered = p_365_mas(df_filtered, 'W')
            # df_filtered['fecha'] = df_filtered['fecha'].apply(lambda x : x[:10])
            dtick = 5

        elif periodo_seleccionado == 1:
            p_selec = 'últimas 24 horas' 
            dtick = "1D"

        elif periodo_seleccionado == 7:
            p_selec = f'última semana'
            dtick = "1D"

        else:
            p_selec = 'último mes'
            dtick = None

        
    else:
        p_selec = 'histórico (media trimestral)'
        df_filtered = p_365_mas(df_filtered, 'Q')
        # df_filtered['fecha'] = df_filtered['fecha'].apply(lambda x : x[:4] + "-Q1")
        dtick = 4

    

    with st.expander(label = f"DataFrame Filtrado: {p_selec}", expanded = False):
        st.dataframe(df_filtered)
        
    fig = px.line(df_filtered, 
                  x = "fecha", 
                  y = "valor_(GWh)", 
                  color='zona', 
                  title = f"Demanda eléctrica en España: {p_selec}",
                  labels={'fecha': "Fecha", 'valor_(GWh)': "Demanda (GWh)", 'zona': "Zona"}
                  )
    
    fig.update_xaxes(
        tickformat="%d-%b-%Y",
        tickangle=-45,
        ticklabelposition="outside bottom",
        dtick=dtick
        )
    
    st.plotly_chart(fig)

    # _________________________________________________________________________________________________________________________

    
    


if __name__ == "__main__":
    vis_demanda()