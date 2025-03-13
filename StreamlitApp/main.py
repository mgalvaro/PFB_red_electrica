from datetime import datetime
import streamlit as st
import pandas as pd
#import plotly.graph_objects as go

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
from ML.visualiza_predicciones import *

from passwords import pw

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

def main():
    st.set_page_config(**PAGE_CONFIG)

    tab1, tab2, tab3, tab4 = st.tabs([":house: INICIO", ":bar_chart: ANALISIS DE DATOS", ":robot_face: MACHINE LEARNING", ":busts_in_silhouette: QUIENES SOMOS"])

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

                1. **ANALISIS DE DATOS**
                    
                    En esta pestaña podrás encontrar visualizaciones y datos sobre la **demanda**, **generacion**, **balance** e **intercambios** energéticos.

                    Además, también podrás consultar **predicciones de demanda** con varios modelos de Machine Learning diferentes.

                2. **MACHINE LEARNING**

                    En esta pestaña se detalla cada uno de los modelos de Machine Learning usados para las predicciones de demanda.

                3. **QUIENES SOMOS**
                    
                    Para que nos conozcas un poco más...

                    """)

    with tab2:
        df_demanda = df_demanda[df_demanda['titulo'] == 'Demanda']
        subtab1, subtab2, subtab3, subtab4 = st.tabs(['Visualiza y predice demanda', 'Mapa de intercambios', 'Comparador anual', 'Visualiza generación por tecnología'])

        with subtab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(" ### :chart_with_upwards_trend: Visualiza y predice demanda")
                st.markdown("#### **¿Qué es la demanda?** ")
                st.markdown("La demanda eléctrica es la cantidad de energía que necesitan hogares, empresas e industrias en un momento dado. Este dato es fundamental para garantizar que haya suficiente suministro de electricidad para satisfacer las necesidades de todos sin interrupciones.")
                periodos_dict = {

                    "Últimos 7 días": 7,
                    "Últimos 30 días": 30,
                    "Últimos 365 días": 365,
                    "Histórico": -1 
                }

                periodo = st.radio(label = "Selecciona el período",
                                key="periodo_demanda",
                                options = list(periodos_dict.keys()),
                                index = 0,
                                disabled = False,
                                horizontal = True
                                )
                
                periodo_seleccionado = periodos_dict[periodo]
            
                demanda, periodo = vis_demanda(df_demanda, periodo_seleccionado)

                ventana_seleccionada = None
                
                if periodo != 365 and periodo != -1:
                    ver_predicciones = st.checkbox(label="Ver predicciones de demanda", value=False)
                else:
                    ver_predicciones = False 

                if ver_predicciones:
                    demanda = demanda[(demanda['zona'] == 'nacional')] 
                    st.info("Las predicciones sólo están disponible para territorio nacional")
                    ventanas_dict = {

                        "Próximos 7 días": 7,
                        "Próximos 15 días": 15,
                        "Próximos 30 días": 30,
                    }

                    ventana_input = st.selectbox(label = "Selecciona el período",
                                    options = list(ventanas_dict.keys()),
                                    placeholder="Seleccione intervalo de predicción",
                                    index = 0)    
                    ventana_seleccionada = ventanas_dict[ventana_input]

                    menu_modelos = ['Recurrent Neural Network (RNN)', 'Long Short-Term Memory (LSTM)', 'Gated Recurrent Unit (GRU)', 'Facebook Prophet']
                    modelo_input = st.selectbox(label="Seleccione modelo de predicción",
                    options = menu_modelos,
                    placeholder="Seleccione Modelo ML",
                    index=None)

            with col2:
                
                if ver_predicciones == False:
                    visualiza_demanda(demanda)
                
                elif ver_predicciones == True:
                    visualiza_predicciones(df_demanda, demanda, ventana_seleccionada, modelo_input)

        with subtab2:
            vis_intercambios(df_intercambios)

        with subtab3:
            vis_compare(df_demanda, df_generacion, df_intercambios)

        with subtab4:
            vis_generacion(df_generacion) 

    with tab3:
        
        st.markdown("#### **¿Qué son los modelos de machine learning en energía?** ")
        st.markdown("El machine learning es una rama de la inteligencia artificial que permite a las máquinas aprender de los datos y mejorar su desempeño sin necesidad de ser programadas explícitamente. En el sector energético, se utiliza para analizar grandes volúmenes de información y realizar tareas como predecir la demanda eléctrica, optimizar la generación de energía o detectar fallos en la red, contribuyendo a una gestión más eficiente y sostenible.")  
        st.markdown("""Dentro del machine learning, las redes neuronales destacan por estar inspiradas en el funcionamiento del cerebro humano. Estas están formadas por capas de "neuronas" que procesan la información, lo que las hace especialmente útiles para resolver problemas complejos como el análisis de series temporales en energía, ayudando a predecir tendencias y tomar decisiones más acertadas para garantizar la estabilidad del sistema eléctrico.
                      En este proyecto hemos usado 4 tipos diferentes que puedes explorar con el selector abajo:""")
        
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            
            menu = ['Simple Recurrent Neural Network (Simple RNN)', 'Long Short-Term Memory (LSTM)', 'Gated Recurrent Unit (GRU)', 'Facebook Prophet']

            choice = st.selectbox(label="Modelos ML", options=menu, index=None, placeholder="Seleccione modelo ML")

            if choice == 'Simple Recurrent Neural Network (Simple RNN)':
                st.markdown("""
                            :brain: Una **RNN simple** es un tipo de red neuronal que aprende de datos que cambian con el tiempo, como es la demanda eléctrica. 
                            
                            :large_green_circle: A diferencia de otras redes, recuerda información pasada para hacer mejores predicciones en el futuro.
                            
                            :large_red_square: Sin embargo, las RNN simples pueden tener problemas para recordar información a largo plazo, pudiendo perder información importante para detectar patrones.
                            """)
                with subcol2:
                    try:
                        df_metricas = pd.read_csv("../ML/MODELS/RNN_LSTM/metricas_rnn.csv")
                        st.markdown("#### Métricas obtenidas")
                        st.dataframe(df_metricas) 

                        with open('../ML/MODELS/RNN_LSTM/history_rnn.pkl', 'rb') as f:
                            history_rnn = pkl.load(f)

                        st.markdown("#### Función de pérdida-mse")
                        fig_mae_rnn = plot_mae(history_rnn)
                        st.plotly_chart(fig_mae_rnn)

                    except FileNotFoundError:
                        st.error("No se ha encontrado ningún archivo relativo a la RNN simple en el sistema, por favor ejecute la pestaña EDA")

            elif choice == 'Long Short-Term Memory (LSTM)':
                st.markdown("""
                            :brain: Una **red neuronal LSTM** es un tipo especial de RNN con una estructura interna más compleja. 
                                
                            :large_green_circle: A diferencia de una RNN simple, la **LSTM** recuerda información importante durante más tiempo, lo que la hace mejor para detectar patrones a largo plazo.
                            
                            :large_red_square: Sin embargo, al ser más complejas, requieren más recursos computacionales que las RNN simples. Necesitan más datos para ser entrenadas y es más difícil de interpretar cómo toman decisiones.

                            """)
                with subcol2:
                    try:
                        df_metricas = pd.read_csv("../ML/MODELS/RNN_LSTM/metricas_lstm.csv")
                        st.markdown("#### Métricas obtenidas")
                        st.dataframe(df_metricas)

                        with open('../ML/MODELS/RNN_LSTM/history_lstm.pkl', 'rb') as f:
                            history_lstm = pkl.load(f)

                        st.markdown("#### Función de pérdida-mse")
                        fig_mae_lstm = plot_mae(history_lstm)
                        st.plotly_chart(fig_mae_lstm)

                        #with st.expander("Representación LOSS-MSE en entrenamientos previos"):
                        #    st.markdown("#### Función de pérdida-mse")
                        #    st.image("../ML/MODELS/GRU/GRU_MAE.png") 

                    except FileNotFoundError:
                        st.error("No se ha encontrado ningún archivo relativo a la LSTM en el sistema, por favor ejecute la pestaña EDA")

            elif choice == 'Gated Recurrent Unit (GRU)':
                st.markdown("""
                            :brain: Una **red neuronal GRU** es un tipo especial de RNN muy parecida a las LSTM, pero menos compleja.
                            
                            :large_green_circle: Se diferencian de otras redes recurrentes, como las LSTM, porque son más simples y requieren menos recursos computacionales, manteniendo un rendimiento similar en muchas aplicaciones.
                            
                            :large_red_square: Sin embargo, no es tan buena como las LSTM para recordar patrones a largo plazo.
                            
                            """)
                with subcol2:
                    df_metricas = pd.read_csv("../ML/MODELS/GRU/MetricasGRU.csv")
                    st.markdown("#### Métricas obtenidas")
                    st.dataframe(df_metricas)                

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
                with subcol2:
                    st.markdown("#### Métricas obtenidas")
                    df_metricas = pd.read_csv("../ML/MODELS/PROPHET/evaluacion_modelo_train.csv")
                    st.dataframe(df_metricas)      

                    df_metricas_comparadas = pd.read_csv("../ML/MODELS/PROPHET/predicciones_comparacion_prophet.csv")
                    st.dataframe(df_metricas_comparadas)
                        
                    st.markdown("#### Representación de las diferentes granularidades del modelo Prophet")
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

        subcol1, subcol2 = st.columns([2, 1])
        with subcol1:
            st.markdown("#### **Álvaro Mejía**:")
            social_icons("alvaro-mejia-garcia", "mgalvaro")
            st.markdown("")
            st.markdown("Ingeniero aeroespacial de formación y actualmente trabajando en la industria médica en Polonia. "
            "Desde hace más de un año, me he estado formando en ciencia de datos para especializarme en este campo y aportar mi experiencia y conocimiento a proyectos innovadores.")
        with subcol2:
            img_alvaro = Image.open("imagenes/amg.png")
            st.image(img_alvaro, width=250)
        st.markdown("---")

        subcol1, subcol2 = st.columns([2, 1])
        with subcol1:
            st.markdown("#### **Ignacio Barba**:")
            social_icons("josé-ignacio-barba-quezada-621975b0", "NachoMijo")
            st.markdown("Comunicador y comercial de profesión, aficionado al mundo tech.")
            st.markdown("Explorando el mundo de la ciencia de datos como un reto.")
        with subcol2:
            img_nacho = Image.open("imagenes/ib.png")
            st.image(img_nacho, width=250)
        st.markdown("---")

        subcol1, subcol2 = st.columns([2, 1])
        with subcol1:
            st.markdown("#### **Javier Corrales**:")
            social_icons("javiercorralesfdez", "javcorrfer")
            st.markdown("")
            st.markdown("""Optometrista con más de 15 años de experiencia y apasionado por la formación continua, me he embarcado en este proyecto para poder especializar mi perfil en el sector.""")
        with subcol2:
            img_javi = Image.open("imagenes/jcf.png")
            st.image(img_javi, width=250)

      
if __name__ == '__main__':
    main()