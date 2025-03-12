import streamlit as st

import numpy as np
import pandas as pd
from datetime import datetime, timedelta 

import pickle as pkl
import plotly.express as px
import plotly.graph_objects as go

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from StreamlitApp.functions.carga_dataframes import *
from StreamlitApp.vis_demanda import *
from ML.escalado_datos import *
from StreamlitApp.passwords import pw
from ML.rnn_lstm import *
from ML.gru_rnn import *
from ML.prophet import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import GRU, Dense, Input, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

def visualiza_demanda(df_filtrado):

    zonas = df_filtrado["zona"].unique()
    fig = go.Figure()

    for zona in zonas:
        df_historial = df_filtrado[df_filtrado["zona"] == zona] 
        fig.add_trace(go.Scatter(
            x=df_historial["fecha"],
            y=df_historial["valor_(GWh)"],
            mode='lines',
            name=zona,
            line=dict(width=2),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="Histórico Demanda",
            title_x=0.5,
            xaxis_title="Fechas",
            yaxis_title="Demanda (GWh)",
            template="plotly_white",
            hovermode="x",
            xaxis=dict(
                tickformat="%Y-%m-%d", 
                tickangle=-45,
                tickmode='array'
            )
        )

    st.plotly_chart(fig)

    with st.expander(label = f"DataFrame filtrado para período seleccionado", expanded = False):
        st.dataframe(df_filtrado)
        
    return None

def visualiza_predicciones(dataframe, df_filtrado, ventana, modelo_input):    
    df = procesar_datos(dataframe)
    df = df.drop(columns="fecha")
    TARGET = df["valor_(GWh)"]
    n_features = len([col for col in df.columns if col != TARGET.name])
    X, y = crea_secuencias(df, TARGET.name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    fechas = [(dataframe["fecha"].max() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(ventana)]
    
    if modelo_input == 'Recurrent Neural Network (RNN)':
        
        with open(f"../ML/MODELS/RNN_LSTM/rnn.pkl", "br") as file:
                rnn = pkl.load(file)

        pred_1step = prediction_1step_rnn(X, y_test, scaler, rnn, n_future=ventana)
        pred_multistep = prediction_multistep_rnn(X, y_test, scaler, rnn, n_future=ventana)

        st.plotly_chart(grafica_predicciones(df_filtrado[-len(y_test):], pred_1step, pred_multistep))

    elif modelo_input == 'Long Short-Term Memory (LSTM)':

        with open(f"../ML/MODELS/RNN_LSTM/lstm.pkl", "br") as file:
                rnn = pkl.load(file)

        pred_1step = prediction_1step_rnn(X, y_test, scaler, rnn, n_future=ventana)
        pred_multistep = prediction_multistep_rnn(X, y_test, scaler, rnn, n_future=ventana)

        st.plotly_chart(grafica_predicciones(df_filtrado[-len(y_test):], pred_1step, pred_multistep))


    elif modelo_input == 'Gated Recurrent Unit (GRU)':

        with open(f"../ML/MODELS/GRU/gru_model.pkl", "br") as file:
                gru_model = pkl.load(file)

        pred_1step = predice_1step(gru_model, X_test, scaler, num_dias=ventana)
        pred_multistep = predice_multistep(gru_model, X_test, scaler, num_dias=ventana)

        st.plotly_chart(grafica_predicciones(df_filtrado[-len(y_test):], pred_1step, pred_multistep))

    elif modelo_input == 'Facebook Prophet':
        with open(f"../ML/MODELS/PROPHET/modelo_prophet_con.pkl", "br") as file: 
            prophet_model = pkl.load(file)
        prophet_model = prophet_model["modelo"]

        pred_1step = predice_prophet(prophet_model, dataframe)
        pred_1step = pred_1step.rename(columns={"ds":"fecha", "yhat":"valor_(GWh)"})
        pred_1step = pred_1step["valor_(GWh)"][:ventana]

        st.plotly_chart(grafica_predicciones(df_filtrado[-len(y_test):], pred_1step))

    if (modelo_input !=None) & (modelo_input != 'Facebook Prophet'):
        with st.expander(label = f"Predicciones inmediatas", expanded = False):
            pred_1step = pd.DataFrame({"valor_(GWh)":pred_1step})
            pred_1step["Fecha"] = fechas
            st.dataframe(pred_1step)
        
        with st.expander(label = f"Predicciones varios pasos", expanded = False):
            pred_multistep = pd.DataFrame({"valor_(GWh)":pred_multistep})
            pred_multistep["Fecha"] = fechas
            st.dataframe(pred_multistep)

        return pred_1step, pred_multistep

    elif (modelo_input !=None) & (modelo_input == 'Facebook Prophet'):
        with st.expander(label = f"Predicciones", expanded = False):
            pred_1step = pd.DataFrame({"valor_(GWh)":pred_1step})
            pred_1step["Fecha"] = fechas
            st.dataframe(pred_1step)

        return pred_1step

    