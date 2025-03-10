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

from ML.rnn_lstm import *
from ML.gru_rnn import *

from passwords import pw

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

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

        menu = ['Simple Recurrent Neural Network (Simple RNN)', 'Long Short-Term Memory (LSTM)', 'Gated Recurrent Unit (GRU)', 'Facebook Prophet']

        choice = st.selectbox(label='Modelos de predicción de demanda', options=menu, index=None, placeholder="Seleccione modelo ML")

        if choice == 'Simple Recurrent Neural Network (Simple RNN)':
            st.markdown("""
                        :brain: Una **red neuronal** es un modelo de computación inspirado en cómo funciona el cerebro humano. Está formada por capas de "neuronas" o unidades que procesan la información, ayudando a resolver tareas como el reconocimiento de imágenes o el análisis de texto. 
 
                        Una **red neuronal** recurrente es un tipo especial de red donde las conexiones entre las neuronas pueden formar ciclos, lo que permite que la red "recuerde" información de lo que ocurrió antes y la utilice para tomar decisiones en el futuro. Esto es útil para tareas como la traducción automática o la predicción de series temporales, ya que la información anterior tiene importancia. 
 
                        Una **RNN simple** es, por tanto, un tipo de red neuronal que aprende de datos que cambian con el tiempo, como es la demanda eléctrica. 
                        
                        :large_green_circle: A diferencia de otras redes, recuerda información pasada para hacer mejores predicciones en el futuro.
                        
                        :large_red_square: Sin embargo, las RNN simples pueden tener problemas para recordar información a largo plazo, pudiendo perder información importante àra detectar patrones.
                         """)
            try:
                df_metricas = pd.read_csv("../ML/MODELS/RNN_LSTM/metricas_rnn.csv")
                st.header("Métricas obtenidas")
                st.dataframe(df_metricas) 

                with open('../ML/MODELS/RNN_LSTM/history_rnn.pkl', 'rb') as f:
                    history_rnn = pkl.load(f)

                fig_mae_rnn = plot_mae(history_rnn)
                st.plotly_chart(fig_mae_rnn)

            except FileNotFoundError:
                st.error("No se ha encontrado ningún archivo relativo a la RNN simple en el sistema, por favor ejecute la pestaña EDA")

        elif choice == 'Long Short-Term Memory (LSTM)':
            st.markdown("""
                        :brain: Una **red neuronal** es un modelo de computación inspirado en cómo funciona el cerebro humano. Está formada por capas de "neuronas" o unidades que procesan la información, ayudando a resolver tareas como el reconocimiento de imágenes o el análisis de texto. 
 
                        Una **red neuronal recurrente (RNN)** es un tipo especial de red donde las conexiones entre las neuronas pueden formar ciclos, lo que permite que la red "recuerde" información de lo que ocurrió antes y la utilice para tomar decisiones en el futuro. Esto es útil para tareas como la traducción automática o la predicción de series temporales, ya que la información anterior tiene importancia. 
                         	
                        :large_green_circle: A diferencia de una RNN simple, la **LSTM** recuerda información importante durante más tiempo, lo que la hace mejor para detectar patrones a largo plazo.
                        
                        :large_red_square: Sin embargo, las LSTM son más complejas y requieren más recursos computacionales que las RNN simples. Necesitan más datos para ser entrenadas y es más difícil de interpretar cómo toman decisiones.

                         """)
            try:
                df_metricas = pd.read_csv("../ML/MODELS/RNN_LSTM/metricas_lstm.csv")
                st.header("Métricas obtenidas")
                st.dataframe(df_metricas)

                with open('../ML/MODELS/RNN_LSTM/history_lstm.pkl', 'rb') as f:
                    history_lstm = pkl.load(f)

                fig_mae_lstm = plot_mae(history_lstm)
                st.plotly_chart(fig_mae_lstm)

                with st.expander("Representación LOSS-MSE en entrenamientos previos"):
                    st.markdown("#### Función de pérdida-mse")
                    st.image("../ML/MODELS/GRU/GRU_MAE.png") 

            except FileNotFoundError:
                st.error("No se ha encontrado ningún archivo relativo a la LSTM en el sistema, por favor ejecute la pestaña EDA")

        elif choice == 'Gated Recurrent Unit (GRU)':
            st.markdown("""
                        :brain: Una **red neuronal** es un modelo de computación inspirado en cómo funciona el cerebro humano. Está formada por capas de "neuronas" o unidades que procesan la información, ayudando a resolver tareas como el reconocimiento de imágenes o el análisis de texto. 
 
                        Una **red neuronal** recurrente es un tipo especial de red donde las conexiones entre las neuronas pueden formar ciclos, lo que permite que la red "recuerde" información de lo que ocurrió antes y la utilice para tomar decisiones en el futuro. Esto es útil para tareas como la traducción automática o la predicción de series temporales, ya que la información anterior tiene importancia. 
 
                        Las **redes GRU (Gated Recurrent Unit)** son un tipo de red neuronal recurrente que utiliza un mecanismo para controlar mejor cómo se "recuerda" o "olvida" la información. 
                        
                        :large_green_circle: Se diferencian de otras redes recurrentes, como las LSTM, porque son más simples y requieren menos recursos computacionales, manteniendo un rendimiento similar en muchas aplicaciones.
                        
                        :large_red_square: Sin embargo, no es tan buena como las LSTM para recordar patrones a largo plazo.
                         
                         """)
            
            with st.expander("Métricas obtenidas"):
     
                df_metricas = pd.read_csv("../ML/MODELS/GRU/MetricasGRU.csv")
                st.dataframe(df_metricas)                

            with st.expander("Representación LOSS-MSE en entrenamientos previos"):
                st.markdown("#### Función de pérdida-mse")
                st.image("../ML/MODELS/GRU/GRU_MAE.png")

            
        elif choice == 'Facebook Prophet':
            st.markdown("""
                        :brain: **Facebook Prophet** es un modelo de predicción de series temporales desarrollado por el equipo de investigación de Facebook. Está diseñado para predecir tendencias futuras utilizando un enfoque basado en regresión aditiva con componentes estacionales, lo que permite modelar patrones complejos de manera automática.  

                        Prophet es especialmente útil para trabajar con datos que presentan tendencias no lineales, efectos estacionales y eventos especiales como festivos o cambios estructurales en la serie temporal.  

                        Una de las principales ventajas de Prophet es su capacidad para manejar valores atípicos y datos faltantes sin necesidad de preprocesamiento exhaustivo, lo que lo convierte en una herramienta flexible y fácil de usar.  

                        :large_green_circle: A diferencia de otros modelos tradicionales de series temporales, Prophet no requiere que el usuario ajuste manualmente los parámetros de tendencia o estacionalidad, ya que el modelo los aprende automáticamente a partir de los datos históricos.  

                        :large_red_square: Sin embargo, Prophet puede no ser la mejor opción cuando se trata de datos con patrones altamente irregulares o cambios abruptos que no siguen una estructura definida, ya que su enfoque basado en tendencias suaves puede generar predicciones menos precisas en estos casos. 
                         
                         """)

            with st.expander("Métricas obtenidas"):
                df_metricas = pd.read_csv("../ML/MODELS/PROPHET/evaluacion_modelo_train.csv")
                st.dataframe(df_metricas)      

                df_metricas_comparadas = pd.read_csv("../ML/MODELS/PROPHET/predicciones_comparacion_prophet.csv")
                st.dataframe(df_metricas_comparadas)
                

            with st.expander("Representación de las diferentes granularidades del modelo Prophet"):
                st.markdown("##### Día a día")
                st.image("../ML/MODELS/PROPHET/prophet_vis_Día a día.png")

                st.markdown("##### Mes a mes")
                st.image("../ML/MODELS/PROPHET/prophet_vis_Mes a mes.png")

                st.markdown("##### Semana a semana")
                st.image("../ML/MODELS/PROPHET/prophet_vis_Semana a semana.png")

                st.markdown("##### Trimestre a trimestre")
                st.image("../ML/MODELS/PROPHET/prophet_vis_Trimestre a trimestre.png")
        
        else:
            pass
    
    with tab4:
        st.markdown("### :zap: ¿Quiénes formamos parte de este proyecto? :zap:")
        st.markdown("")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### **Álvaro Mejía**:")
            social_icons("alvaro-mejia-garcia", "mgalvaro")
            st.markdown("")
            st.markdown("Ingeniero aeroespacial de formación y actualmente trabajando en la industria médica en Polonia. "
            "Desde hace más de un año, me he estado formando en ciencia de datos para especializarme en este campo y aportar mi experiencia y conocimiento a proyectos innovadores.")
        with col2:
            img_alvaro = Image.open("imagenes/amg.png")
            st.image(img_alvaro, width=500)
        st.markdown("---")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### **Ignacio Barba**:")
            social_icons("josé-ignacio-barba-quezada-621975b0", "NachoMijo")
            st.markdown("Comunicador y comercial de profesión, aficionado al mundo tech.")
            st.markdown("Explorando el mundo de la ciencia de datos como un reto.")
        with col2:
            img_nacho = Image.open("imagenes/ib.png")
            st.image(img_nacho, width=510)
        st.markdown("---")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### **Javier Corrales**:")
            social_icons("javiercorralesfdez", "javcorrfer")
            st.markdown("")
            st.markdown("""Optometrista con más de 15 años de experiencia y apasionado por la formación continua, me he embarcado en este proyecto para poder especializar mi perfil en el sector.""")
        with col2:
            img_javi = Image.open("imagenes/jcf.png")
            st.image(img_javi, width=520)

      
if __name__ == '__main__':
    main()