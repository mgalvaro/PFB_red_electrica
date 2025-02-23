import streamlit as st

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import seaborn as sns
import plotly.express as px

from functions.filtros_visualizaciones import *


def vis_generacion(df):

    # 4ª gráfica: comparación de la generación por tecnologías
    st.markdown("### :bar_chart: Generación de energía por tecnología")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Fecha de inicio").strftime('%Y-%m-%d')
        
    with col2:
        end_date = st.date_input("Fecha de fin").strftime('%Y-%m-%d')


    zonas = df['zona'].unique().tolist()

    zona = st.multiselect(label='Áreas de generación',
                                      options=zonas,
                                      default=None,
                                      placeholder='Seleccionar la zona'
                           )
    
    if len(zona) != 1:
        st.warning('Escoger una zona')
    
    else:
        df = df[(df['fecha'] >= start_date) & (df['fecha'] <= end_date)]
        df = df[df['zona'] == zona[0]]
        df['valor_(GWh)'] = df['valor_(MWh)'].apply(lambda x: x * 0.001) 

        fig = px.line(df, 
                    x = "fecha", 
                    y = "valor_(GWh)", 
                    color='titulo', 
                    title = f"Generación de energía por tecnolgía entre {start_date} y {end_date}",
                    labels={'fecha': "Fecha", 'valor_(GWh)': "Generación (GWh)", 'titulo': "Tecnología"}
                    )
    
        st.plotly_chart(fig)

        st.divider()

        c1, c2 = st.columns(2)

        with c1:

            fig_pie = px.pie(data_frame = df,
                        names      = "titulo",
                        values     = "valor_(GWh)",
                        title      = f"Generación de energía por tecnología ({zona})")
            st.plotly_chart(figure_or_data = fig_pie, use_container_width = True)

        with c2:

            fig_pie2 = px.pie(data_frame = df.groupby('tipo').sum('valor_(GWh)').reset_index(drop=False),
                        names      = "tipo",
                        values     = "valor_(GWh)",
                        title      = f"Generación de energía por tecnología (zona: {zona[0]})")
            st.plotly_chart(figure_or_data = fig_pie2, use_container_width = True)


if __name__ == "__main__":
    vis_generacion()