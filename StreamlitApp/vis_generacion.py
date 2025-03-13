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
        st.markdown("#### **¿Qué es la generación?** ")
        st.markdown("La generación eléctrica es el proceso mediante el cual se produce la electricidad que usamos diariamente. Puede provenir de fuentes renovables, como el sol y el viento, o de fuentes tradicionales, como el gas natural o el carbón. Cada tipo de fuente tiene un papel importante en garantizar un suministro constante.")
        st.markdown("**Gráfico 1:** Representa el aporte de generación de cada tipo de energía en la zona e intervalo seleccionados.")
        st.markdown("**Gráfico 2:** Representa la misma información pero en un gráfico de pastel.")
        st.markdown("**Gráfico 3:** Representa la misma información pero en un gráfico de pastel agrupado por renovable y no renovable.")
        start_date = st.date_input("Fecha de inicio", key="start_generacion")
        end_date = st.date_input("Fecha de fin",key="end_generacion")

        zonas = ["Nacional", "Peninsular", "Ceuta", "Melilla", "Canarias", "Baleares"]
        zona = st.selectbox(label='Áreas de generación',
                                        options=zonas,
                                        index=None,
                                        placeholder='Seleccionar la zona'
                            )

    with col2:
        try:
            df = df[(df['fecha'] >= start_date) & (df['fecha'] <= end_date)]
            df = df[df['zona'] == zona.lower()]
            df['valor_(GWh)'] = df['valor_(MWh)'].apply(lambda x: x * 0.001) 

            fig = px.line(df, 
                        x = "fecha", 
                        y = "valor_(GWh)", 
                        color='titulo', 
                        title = f"Generación de energía por tecnología entre {start_date} y {end_date}",
                        labels={'fecha': "Fecha", 'valor_(GWh)': "Generación (GWh)", 'titulo': "Tecnología"}
                        )
        
            fig.update_xaxes(tickformat="%d-%b-%Y")
            st.plotly_chart(fig)

            st.divider()

            c1, c2 = st.columns(2)

            with c1:
                fig_pie = px.pie(data_frame = df,
                            names      = "titulo",
                            values     = "valor_(GWh)",
                            title      = f"Generación de energía por tecnología ({zona.lower()})")
                st.plotly_chart(figure_or_data = fig_pie, use_container_width = True)

            with c2:

                fig_pie2 = px.pie(data_frame = df.groupby('tipo').sum('valor_(GWh)').reset_index(drop=False),
                            names      = "tipo",
                            values     = "valor_(GWh)",
                            title      = f"Generación de energía por tecnología (zona: {zona.lower()})")
                st.plotly_chart(figure_or_data = fig_pie2, use_container_width = True)
        
        except AttributeError:
            st.info(":warning: No hay datos que mostrar")


if __name__ == "__main__":
    vis_generacion()