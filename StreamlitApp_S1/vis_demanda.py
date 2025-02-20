import streamlit as st

import pandas as pd
import plotly.express as px

from filtros_visualizaciones import *

#_____________________


def vis_demanda():

    # 1ª gráfica: Serie temporal de Demanda

    periodos_dict = {

        "Últimos 7 días": 7,
        "Últimos 30 días": 30,
        "Últimos 365 días": 365,
        "Histórico": -1 
    }

    st.markdown("#### Demanda Eléctrica en España")
    with st.expander(label = "Dataset: Demanda", expanded = False):
        df = pd.read_csv('../SPRINT1PRUEBAS/Data/DF DEMANDA_20_25_LIMPIO_V1.csv')
        df = df[df['titulo'] == 'Demanda']
        st.dataframe(df) 

    # aquí meter código para cuando estén definitivamente las funciones de extracción y de limpieza

    periodo = st.radio(label = "Selecciona el período",
                       options = list(periodos_dict.keys()),
                       index = 0,
                       disabled = False,
                       horizontal = True,)
    
    periodo_seleccionado = periodos_dict[periodo]

    df_filtered = p7_30_365_hist(df, periodo_seleccionado)[0]
    periodo_seleccionado = p7_30_365_hist(df, periodo_seleccionado)[1]
    

    with st.expander(label = f"DataFrame Filtrado para {periodo_seleccionado} días", expanded = False):
        st.dataframe(df_filtered)
        
    fig = px.line(df_filtered, 
                  x = "fecha", 
                  y = "valor", 
                  color='zona', 
                  title = f"Demanda eléctrica en España de los últimos {periodo_seleccionado} días",
                  labels={'fecha': "Fecha", 'valor': "Demanda (MW)", 'zona': "Zona"}
                  )
    
    st.plotly_chart(fig)

    # _________________________________________________________________________________________________________________________

    
    


if __name__ == "__main__":
    vis_demanda()