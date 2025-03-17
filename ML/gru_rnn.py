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
from ML.prophet import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import GRU, Dense, Input, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# Cargar el scaler
with open("../data/data_scaled/scalers/scaler_consumo_anio_DF_DEMANDA.pkl", "br") as file:
    scaler = pkl.load(file)

# Función para crear secuencias
def crea_secuencias(dataframe, target_col, len_secuencia=15) -> tuple:
    X, y = [], []
    for i in range(len(dataframe) - len_secuencia):
        X.append(dataframe.iloc[i:i+len_secuencia].drop(columns=[target_col]).values) 
        y.append(dataframe.iloc[i+len_secuencia][target_col]) 
    return np.array(X), np.array(y)

def crea_gru(input_shape, xtrain, xtest, ytrain, ytest, len_secuencia=15, epochs=1500) -> tuple: #--  agregar metricas en un df
    # Yo probaría algo similar a esto (ignoren los números de neuronas, eso está sujeto a ensayo y error):
    model = Sequential([
        Input(shape=(len_secuencia, input_shape)),
        GRU(128, activation='tanh'), # Capa de procesamiento recurrente, responsable de interpretar la secuencialidad de los datos
        Dense(64, activation='relu'), # Capas de procesamiento MLP, el "cerebro" de la red neuronal
        Dropout(0.3), # Funcinarán unas 45 neuronas
        Dense(32, activation='relu'),
        Dropout(0.2), # Funcionarán unas 25 neuronas
        Dense(16, activation='relu'),
        Dropout(0.1), # Funcionarán unas 14 neuronas
        Dense(1, activation='linear') # No sería mala idea probar con ReLU en esta última capa, ya que nuestra respuesta no puede ser negativa
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae']) # Sería buena idea meter más métricas como rmse o mae para mejor interpretabilidad
    history = model.fit(x=xtrain, y=ytrain, validation_data=(xtest, ytest), epochs=epochs, verbose=0)

    with open(f"MODELS/GRU/gru_model.pkl", "bw") as file:
        pkl.dump(model,file)
    with open(f"MODELS/GRU/gru_model_history.pkl", "bw") as file:
        pkl.dump(history,file)
        
        return model, history

# Función para graficar loss y mae 
def grafica_loss_mae(historial) -> None:
    fig = px.line(data_frame=historial.history, 
                  y=['loss', 'val_loss'], 
                  title='Función de pérdida (loss) basada en el MSE',
                  labels={'index': 'Época', 'value': 'Pérdida'})
    
    fig.update_layout(title_x=0, 
                      legend_title_text="Variables")
    
    fig.for_each_trace(lambda x: x.update(name="Pérdida Entrenamiento" if x.name == "loss" else "Pérdida Validación"))

    fig.to_image("../ML/MODELS/GRU/GRU_MAE.webp")
    fig.to_image("../ML/MODELS/GRU/GRU_MAE.png")
    return fig

# Función para predecir 1-step   
def predice_1step(model, data, scaler, num_dias, len_secuencia=15) -> np.array:  
    validation_predictions = []
    
    i = -num_dias
    while len(validation_predictions) < num_dias:
        p = model.predict(data[i].reshape(1, len_secuencia, data.shape[2]))[0, 0]
        validation_predictions.append(p)
        i += 1

    # Desescalado
    dummy_features = np.zeros((len(validation_predictions), 1))
    predictions_with_dummy = np.hstack([np.array(validation_predictions).reshape(-1, 1), dummy_features])
    predictions_desescalado = scaler.inverse_transform(predictions_with_dummy)[:, 0]

    return predictions_desescalado

# Función para predecir multi-step
def predice_multistep(model, data, scaler, num_dias, len_secuencia=15) -> np.array:
    predictions = []
    input_seq = data[-1].reshape(1, len_secuencia, data.shape[2])
    
    for _ in range(num_dias):
        pred = model.predict(input_seq)[0, 0]
        predictions.append(pred)
        input_seq = np.roll(input_seq, shift=-1, axis=1)  
        input_seq[0, -1, -1] = pred  
    
    # Desescalado
    dummy_features = np.zeros((len(predictions), 1))
    predictions_with_dummy = np.hstack([np.array(predictions).reshape(-1, 1), dummy_features])
    predictions_desescalado = scaler.inverse_transform(predictions_with_dummy)[:, 0]

    return predictions_desescalado

def muestra_metricas(dataframe, ventana, preds) -> pd.DataFrame:

    df_validacion = dataframe[dataframe["zona"] == "nacional"][-ventana:]["valor_(MWh)"]*0.001
    dict_metricas = {
        "r2" : r2_score(df_validacion, preds),
        "mae_GWh" : mean_absolute_error(df_validacion, preds),
        "rmse_GWh" : np.sqrt(mean_squared_error(df_validacion, preds))
    }
    return dict_metricas

def grafica_predicciones(real, pred_1step, pred_multistep=None) -> go.Figure:
    
    real = real[(real['zona'] == 'nacional')]
    ultima_fecha = real['fecha'].iloc[-1]
    fechas_futuras = [(ultima_fecha + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(pred_1step))]
    fechas_pasadas = [fecha.strftime("%Y-%m-%d") for fecha in real['fecha']]
    pred_1step = np.concatenate([[real['valor_(GWh)'].iloc[-1]], pred_1step])

    if pred_multistep is not None:
        pred_multistep = np.concatenate([[real['valor_(GWh)'].iloc[-1]], pred_multistep])
        
        fig = go.Figure()

        # Valores reales (pasado)
        valores_pasado = real['valor_(GWh)']    
        fig.add_trace(go.Scatter(
            x=fechas_pasadas,
            y=valores_pasado,
            mode='lines',
            name='Valores Reales (Pasado)',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))

        # Predicciones 1-step
        fig.add_trace(go.Scatter(
            x=fechas_futuras,
            y=pred_1step,
            mode='lines',
            name='Predicciones inmediatas',
            line=dict(color='red', width=2, dash='dot'),
            marker=dict(size=6)
        ))

        # Predicciones multi-step
        fig.add_trace(go.Scatter(
            x=fechas_futuras,
            y=pred_multistep,
            mode='lines',
            name='Predicciones en varios pasos',
            line=dict(color='green', width=2, dash='dot'),
            marker=dict(size=6)
        ))

        # Configuración de la gráfica
        fig.update_layout(
            title="Predicciones vs Valores Reales",
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
        return fig
    else:
        fig = go.Figure()

        # Valores reales (pasado)
        valores_pasado = real['valor_(GWh)']    
        fig.add_trace(go.Scatter(
            x=fechas_pasadas,
            y=valores_pasado,
            mode='lines',
            name='Valores Reales (Pasado)',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))

        # Predicciones 1-step
        fig.add_trace(go.Scatter(
            x=fechas_futuras,
            y=pred_1step,
            mode='lines',
            name='Predicciones',
            line=dict(color='red', width=2, dash='dot'),
            marker=dict(size=6)
        ))

        # Configuración de la gráfica
        fig.update_layout(
            title="Predicciones vs Valores Reales",
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
        return fig