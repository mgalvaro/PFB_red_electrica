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
from ML.escalado_datos import *
from StreamlitApp.passwords import pw

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import GRU, Dense, Input, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore



# Cargar el scaler
with open("../data/data_scaled/scalers/scaler_consumo_anio_DF_DEMANDA.pkl", "br") as file:
    scaler = pkl.load(file)

# Función para crear secuencias
def crea_secuencias(dataframe, target_col, len_secuencia) -> tuple:
    X, y = [], []
    for i in range(len(dataframe) - len_secuencia):
        X.append(dataframe.iloc[i:i+len_secuencia].drop(columns=[target_col]).values) 
        y.append(dataframe.iloc[i+len_secuencia][target_col]) 
    return np.array(X), np.array(y)

# Función para redimensionar las secuencias
def redimensiona(xtrain, xtest, ventana, n_features) -> str:
    xtrain = xtrain.reshape((xtrain.shape[0], ventana, n_features))  
    xtest = xtest.reshape((xtest.shape[0], ventana, n_features)) 
    return f"X_train shape: {xtrain.shape}, X_test shape: {xtest.shape}"

# Construcción del modelo GRU con más capas. Este modelo me parece más realista
def crea_gru(input_shape, len_secuencia, xtrain, xtest, ytrain, ytest) -> tuple:
    model = Sequential([
        Input(shape=(len_secuencia, input_shape)),
        GRU(64, activation='tanh', return_sequences=True),  
        Dropout(0.3), 
        GRU(64, activation='tanh', return_sequences=False),  
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear') 
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(x=xtrain, y=ytrain, validation_data=(xtest, ytest), epochs=50, verbose=0, callbacks=[early_stopping])
    return model, history

# Función para graficar loss y mae
def grafica_loss_mae(historial) -> None:
    fig = px.line(data_frame=historial.history, 
                  y=['loss', 'val_loss'], 
                  title='Función de pérdida (loss) basada en el MSE',
                  labels={'index': 'Época', 'value': 'Pérdida'})
    
    fig.update_layout(title_x=0.5, 
                      template="plotly_white",
                      legend_title_text="Variables")
    
    fig.for_each_trace(lambda x: x.update(name="Pérdida Entrenamiento" if x.name == "loss" else "Pérdida Validación"))

    return fig

# Función para predecir 1-step
def predice_1step(model, data, y, scaler, len_secuencia, num_dias) -> tuple:
    validation_target = y[-num_dias:]
    validation_predictions = []
    
    i = -num_dias
    while len(validation_predictions) < len(validation_target):
        p = model.predict(data[i].reshape(1, len_secuencia, data.shape[2]))[0, 0]
        validation_predictions.append(p)
        i += 1

    # Crear arrays para revertir el escalado
    dummy_features = np.zeros((len(validation_predictions), 1))
    predictions_with_dummy = np.hstack([np.array(validation_predictions).reshape(-1, 1), dummy_features])
    validation_target_dummy = np.hstack([validation_target.reshape(-1, 1), dummy_features])

    # Desescalar
    predictions_desescalado = scaler.inverse_transform(predictions_with_dummy)[:, 0]
    validation_desescalado = scaler.inverse_transform(validation_target_dummy)[:, 0]

    return predictions_desescalado, validation_desescalado

# Función para predecir multi-step
def predice_multistep(model, data, scaler, len_secuencia, num_dias_multi) -> np.array:
    predictions = []
    input_seq = data[-1].reshape(1, len_secuencia, data.shape[2])
    
    for _ in range(num_dias_multi):
        pred = model.predict(input_seq)[0, 0]
        predictions.append(pred)
        input_seq = np.roll(input_seq, shift=-1, axis=1)  # Desplazar la secuencia
        input_seq[0, -1, -1] = pred  # Actualizar con la predicción
    
    # Desescalar
    dummy_features = np.zeros((len(predictions), 1))
    predictions_with_dummy = np.hstack([np.array(predictions).reshape(-1, 1), dummy_features])
    predictions_desescalado = scaler.inverse_transform(predictions_with_dummy)[:, 0]

    return predictions_desescalado

# Función para predecir x días en el futuro
def predice_futuro(model, data, scaler, len_secuencia, num_dias_futuro) -> np.array:
    predictions = []
    input_seq = data[-1].reshape(1, len_secuencia, data.shape[2])

    for _ in range(num_dias_futuro):
        # Generar predicción
        pred = model.predict(input_seq, verbose=0)[0, 0]
        predictions.append(pred)
        
        # Actualizar la secuencia de entrada con la predicción
        input_seq = np.roll(input_seq, shift=-1, axis=1)
        input_seq[0, -1, -1] = pred

    # Desescalar las predicciones
    dummy_features = np.zeros((len(predictions), 1))
    predictions_with_dummy = np.hstack([np.array(predictions).reshape(-1, 1), dummy_features])
    predictions_desescalado = scaler.inverse_transform(predictions_with_dummy)[:, 0]

    return predictions_desescalado

# Graficar las predicciones 1-step y multi-step
def grafica_predicciones(real, pred_1step, pred_multistep) -> None:
    fig = go.Figure()

    # Valores reales
    fig.add_trace(go.Scatter(
        y=real,
        mode='lines+markers',
        name='Valores Reales',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))

    # Predicciones 1-step
    fig.add_trace(go.Scatter(
        y=pred_1step,
        mode='lines+markers',
        name=f'Predicciones 1-step',
        line=dict(color='red', width=2, dash='dot'),
        marker=dict(size=6)
    ))

    # Predicciones multi-step
    fig.add_trace(go.Scatter(
        y=pred_multistep,
        mode='lines+markers',
        name=f'Predicciones Multi-step',
        line=dict(color='green', width=2, dash='dot'),
        marker=dict(size=6)
    ))

    # Configuración de la gráfica
    fig.update_layout(
        title="Predicciones vs Valores Reales",
        title_x=0.5,
        xaxis_title="Días",
        yaxis_title="Demanda (GWh)",
        template="plotly_white",
        hovermode="x",
        xaxis=dict(
            tickvals=list(range(len(pred_1step) + 1)),  
            ticktext=[str(i) for i in range(1, len(pred_1step) + 1)]  
        ))

    return fig

# Graficar las predicciones futuras
def grafica_predicciones_futuras(pred_futuro) -> None:
    fig = go.Figure()

    # Predicciones futuras
    fig.add_trace(go.Scatter(
        y=pred_futuro,
        mode='lines+markers',
        name=f'Predicciones Futuras',
        line=dict(color='orange', width=2, dash='dot'),
        marker=dict(size=6)
    ))

    # Configuración de la gráfica
    fig.update_layout(
        title="Predicciones para Días Futuros",
        title_x=0.5,
        xaxis_title="Días Futuros",
        yaxis_title="Demanda (GWh)",
        template="plotly_white",
        hovermode="x",
        xaxis=dict(
            tickvals=list(range(len(pred_futuro))),
            ticktext=[str(i + 1) for i in range(len(pred_futuro))]
        ))

    return fig


def vis_gru(dataframe):

    ventanas_dict = {

        "Próximos 7 días": 7,
        "Próximos 15 días": 15,
        "Próximos 30 días": 30,
    }

    ventana_input = st.selectbox(label = "Selecciona el período",
                    options = list(ventanas_dict.keys()),
                    placeholder="Seleccione intervalo de predicción",
                    index = None)

    if ventana_input != None:
        
        ventana_seleccionada = ventanas_dict[ventana_input]
        ventana = ventana_seleccionada

        df = procesar_datos(dataframe)
        df = df.drop(columns="fecha")
        TARGET = df["valor_(GWh)"]
        n_features = len([col for col in df.columns if col != TARGET.name])
        X, y = crea_secuencias(df, TARGET.name, ventana)

        # División en entrenamiento y test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        redimensiona(X_train, X_test, ventana, n_features)

        # Construir y entrenar el modelo
        st.text("...cargando datos, por favor espere...")
        model, history = crea_gru(X.shape[2], ventana, X_train, X_test, y_train, y_test)
        
        # Graficado de loss-mae
        st.plotly_chart(grafica_loss_mae(history))

        # Creamos lista fechas para que las predicciones sean más legibles
        fechas = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(ventanas_dict[ventana_input])]

        # Predicciones 1-step
        pred_1step, real_1step = predice_1step(model, X_test, y_test, scaler, ventana, num_dias=ventana)

        # Predicciones multi-step
        pred_multistep = predice_multistep(model, X_test, scaler, ventana, num_dias_multi=ventana)

        # Graficado de predicciones
        st.plotly_chart(grafica_predicciones(real_1step, pred_1step, pred_multistep))

        # Dataframes de predicciones
        st.header("Predicciones")
        with st.expander(label = f"Predicciones 1-Step", expanded = False):
            pred_1step = pd.DataFrame({"Predicción (GWh)":pred_1step})
            pred_1step["Fecha"] = fechas
            st.dataframe(pred_1step)

        with st.expander(label = f"Predicciones MultiStep", expanded = False):
            pred_multistep = pd.DataFrame({"Predicción (GWh)":pred_multistep})
            pred_multistep["Fecha"] = fechas
            st.dataframe(pred_multistep)

        