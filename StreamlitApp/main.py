from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import PAGE_CONFIG
from PIL import Image

from vis_demanda import vis_demanda
from vis_intercambios_mapa import vis_intercambios
from vis_compare_years import vis_compare
from vis_generacion import vis_generacion
from functions.carga_dataframes import *
from social_icons import social_icons

from ML.gru_rnn import *

from passwords import pw

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from ML.gru_rnn import *

def main():
    st.set_page_config(page_title="APP de datos REE", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
    tab1, tab2, tab3, tab4 = st.tabs(["HOME", "EDA", "ML", "ABOUT US"])

    try:
        df_balance, df_demanda, df_generacion, df_intercambios = carga_dataframes(pw["host"], pw["user"], pw["password"], pw["database"])
    except Exception:
        st.error(":exclamation: Error al cargar los datos")

    with tab1:
        st.markdown('### :zap: Bienvenid@ a la App de datos de la REE :zap:')

        st.markdown("""
                #### Este proyecto ofrece una interfaz simple e intuitiva para cualquier persona interesada en el sector eléctrico de España. 

                Gracias a [REData API](https://www.ree.es/es/datos/apidatos), un servicio informativo gratuito puesto a disposición por Grupo Red Eléctrica (grupo empresarial multinacional de origen español que actúa en el mercado eléctrico internacional como operador del sistema eléctrico), recopilamos toda la información relativa a **demanda, generación, intercambios y balance** energéticos desde el año 2010 hasta el presente. 

                La app se distribuye en 3 pestañas:

                1. **EDA**
                    
                    En esta pestaña podrás encontrar visualizaciones y datos sobre la **demanda**, **generacion**, **balance** e **intercambios** energéticos.

                    Además, también podrás consultar **predicciones de demanda** con 3 modelos de Machine Learning diferentes!

                2. **ML**

                    En esta pestaña se detalla cada uno de los 3 modelos de Machine Learning usados para las predicciones de demanda.

                3. **ABOUT US**
                    
                    Para que nos conozcas un poco más...

                    """)

    with tab2:

        df_demanda = df_demanda[df_demanda['titulo'] == 'Demanda']
            
        menu = ['Serie Temporal Demanda', 'Mapa de Intercambios', 'Comparación Anual', 'Generación por tecnología']

        choice = st.selectbox(label='Menu', options=menu, index=None, placeholder="Seleccione datos a visualizar")

        if choice == 'Serie Temporal Demanda':
            vis_gru(df_demanda)

        elif choice == 'Mapa de Intercambios':
            vis_intercambios(df_intercambios)

        elif choice == 'Comparación Anual':
            vis_compare(df_demanda, df_generacion, df_intercambios)

        elif choice == 'Generación por tecnología':
            vis_generacion(df_generacion)
        
        else:
            pass
    
    with tab3:
        st.header("En construcción...")
        menu = ['Recurrent Neural Network (RNN)', 'Long Short-Term Memory (LSTM)', 'Gated Recurrent Unit (GRU)', 'Facebook Propeth']

        choice = st.selectbox(label='Modelos de predicción de demanda', options=menu, index=None, placeholder="Seleccione modelo ML")

        if choice == 'Recurrent Neural Network (RNN)':
            st.header("Recurrent Neural Network (RNN)")
            try:
                df_metricas = pd.read_csv("../ML/MODELS/GRU/MetricasRNN.csv")
                st.header("Métricas obtenidas")
                st.dataframe(df_metricas)                
            except FileNotFoundError:
                st.error("No se ha encontrado ningún archivo relativo a la RNN GRU en el sistema, por favor ejecute la pestaña EDA")

        elif choice == 'Long Short-Term Memory (LSTM)':
            st.header("Long Short-Term Memory (LSTM)")
            try:
                df_metricas = pd.read_csv("../ML/MODELS/GRU/MetricasGRU.csv")
                st.header("Métricas obtenidas")
                st.dataframe(df_metricas)                
            except FileNotFoundError:
                st.error("No se ha encontrado ningún archivo relativo a la RNN GRU en el sistema, por favor ejecute la pestaña EDA")

        elif choice == 'Gated Recurrent Unit (GRU)':
            st.markdown("""
                         Una red neuronal es un modelo de computación inspirado en cómo funciona el cerebro humano. Está formada por capas de "neuronas" o unidades que procesan la información, ayudando a resolver tareas como el reconocimiento de imágenes o el análisis de texto. 
 
                         Una red neuronal recurrente es un tipo especial de red donde las conexiones entre las neuronas pueden formar ciclos, lo que permite que la red "recuerde" información de lo que ocurrió antes y la utilice para tomar decisiones en el futuro. Esto es útil para tareas como la traducción automática o la predicción de series temporales, ya que la información anterior tiene importancia. 
 
                         Las redes GRU (Gated Recurrent Unit) son un tipo de red neuronal recurrente que utiliza un mecanismo para controlar mejor cómo se "recuerda" o "olvida" la información. Se diferencian de otras redes recurrentes, como las LSTM, porque son más simples y requieren menos recursos computacionales, manteniendo un rendimiento similar en muchas aplicaciones.
                         """)
            with st.expander("Métricas obtenidas"):
     
                df_metricas = pd.read_csv("../ML/MODELS/GRU/MetricasGRU.csv")
                st.dataframe(df_metricas)                

            with st.expander("Representación LOSS-MSE en entrenamientos previos"):
                st.markdown("#### Función de pérdida-mse prediciendo")
                st.image("../ML/MODELS/GRU/GRU_MAE.png")

            
        elif choice == 'Facebook Propeth':
            st.header("En construcción...")
        
        else:
            pass
    
    with tab4:
        st.markdown("### :zap: ¿Quiénes formamos parte de este proyecto? :zap:")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### **Álvaro Mejía**:")
            social_icons("alvaro-mejia-garcia", "mgalvaro")
            st.markdown("Ingeniero aeroespacial de formación y actualmente trabajando en la industria farmacéutica en Polonia. "
            "Desde hace más de un año, me he estado formando en ciencia de datos para especializarme en este campo y aportar mi experiencia y conocimiento a proyectos innovadores.")
        with col2:
            img_alvaro = Image.open("imagenes/amg.png")
            st.image(img_alvaro, width=300)
        st.markdown("---")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### **Ignacio Barba**:")
            social_icons("josé-ignacio-barba-quezada-621975b0", "NachoMijo")
            st.markdown("rellenar info")
        with col2:
            st.markdown("insertar foto")
        st.markdown("---")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### **Javier Corrales**:")
            social_icons("javiercorralesfdez", "javcorrfer")
            st.markdown("""Optometrista con más de 15 años de experiencia y apasionado por la formación continua, me he embarcado en este proyecto para poder especializar mi perfil en el sector.""")
        with col2:
            img_javi = Image.open("imagenes/jcf.png")
            st.image(img_javi, width=300)

      
if __name__ == '__main__':
    main()