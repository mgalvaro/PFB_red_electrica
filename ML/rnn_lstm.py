import numpy as np
import pandas as pd
import sys
import os
import pickle as pkl
import joblib

sys.path.append(os.path.abspath("../"))

import plotly.express as px

from ML.encoding import encoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_error, r2_score
from holidays import Spain
from StreamlitApp.functions.carga_dataframes import *
from StreamlitApp.passwords import pw
from ML.escalado_datos import *

from keras.layers import Input, SimpleRNN, Dense, Dropout, LSTM, GlobalMaxPool1D, GlobalAveragePooling1D # type: ignore
from keras.models import Sequential # type: ignore
from keras.optimizers import Adam, Nadam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

with open("../data/data_scaled/scalers/scaler_consumo_anio_DF_DEMANDA.pkl", "br") as file:
    scaler = pkl.load(file)

#---------------------------------------------------------------------------------------------------------
# sequencias para predecir y train_test

def create_sequences_rnn(df, target_column, lookback=30):
    X, y = [], []
    for i in range(len(df) - lookback):
        X.append(df.iloc[i:i+lookback].drop(columns=[target_column]).to_numpy()) 
        y.append(df.iloc[i+lookback][target_column]) 
    return np.array(X), np.array(y)


def train_test(X, y, f=0.8):    
    train_size = int(len(X) * f)

    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    return X_train, X_val, y_train, y_val


#---------------------------------------------------------------------------------------------------------
# modelos de RNN y LSTM

def get_model_rnn(X, X_train, X_val, y_train, y_val, lookback=30, epochs=100, lr=0.001, loss="mse", metrics=["mae"], batch_size=32):

    model_rnn = Sequential([
        Input(shape=(lookback, X.shape[2])),

        SimpleRNN(128, activation="relu", return_sequences=False),  

        Dense(64, activation="relu"),  
        Dropout(0.1),

        Dense(32, activation="relu"),

        Dense(1)
    ])
    
    model_rnn.compile(optimizer=Adam(learning_rate=lr), loss = loss, metrics = metrics)

    history = model_rnn.fit(x = X_train, 
                             y = y_train, 
                             validation_data = (X_val, y_val), 
                             epochs = epochs,
                             batch_size = batch_size,
                             verbose=1
                             )
    
    joblib.dump(model_rnn, 'MODELS/RNN_LSTM/rnn_new.pkl')
    print("Modelo guardado")

    return model_rnn, history


def get_model_lstm(X, X_train, X_val, y_train, y_val, lookback=30, epochs=100, lr=0.001, loss="mse", metrics=["mae"], batch_size=32):

    model_lstm = Sequential([
        Input(shape=(lookback, X.shape[2])),

        LSTM(128, activation="tanh", return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        LSTM(64, activation="tanh", return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        GlobalAveragePooling1D(),  # Agregamos la capa de pooling para reducir dimensionalidad

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(32, activation="relu"),

        Dense(1)  # Capa de salida
    ])

    model_lstm.compile(optimizer=Nadam(learning_rate=lr), loss=loss, metrics=metrics)

    history = model_lstm.fit(x = X_train, 
                             y = y_train, 
                             validation_data = (X_val, y_val), 
                             epochs = epochs,
                             batch_size = batch_size,
                             verbose=1
                             )
    
    joblib.dump(model_lstm, 'MODELS/RNN_LSTM/lstm_new.pkl')
    print("Modelo guardado")

    return model_lstm, history


def plot_mae(history):
    
    fig = px.line(data_frame=history.history,
        y=['loss', 'val_loss'],
        title='Función de pérdida (loss) basada en el MSE',
        labels={'index': 'Época', 'value': 'Pérdida'},
        )

    fig.update_layout(
        title_x=0,
        legend_title_text="Variables"
    )

    fig.for_each_trace(lambda t: t.update(name="Pérdida Entrenamiento" if t.name == "loss" else "Pérdida Validación"))
    
    return fig


def desescalado(val_target, val_pred, scaler):

    """Toma el set de validación del target, las predicciones y el scaler utilizado para escalar los datos.
    Devuelve los valores reales de las predicciones y del target."""

    # Crear un array con la misma cantidad de columnas que el scaler espera
    dummy_features = np.zeros((len(val_pred), 1))

    # Convertir las predicciones en un array con la forma adecuada
    val_pred = np.array(val_pred).reshape(-1, 1)
    val_target = np.array(val_target).reshape(-1, 1)

    # Unir las predicciones con los valores ficticios
    predictions_with_dummy = np.hstack([val_pred, dummy_features])
    validation_target_dummy = np.hstack([val_target, dummy_features])

    # Aplicar la transformación inversa
    predictions_real = scaler.inverse_transform(predictions_with_dummy)[:, 0]  # Solo tomar la columna de interés
    validation_real = scaler.inverse_transform(validation_target_dummy)[:, 0]

    return validation_real, predictions_real

#---------------------------------------------------------------------------------------------------------

# Funciones para predecir

def predict_1step_rnn(X, y_val, model, scaler):

    validation_target = y_val
    validation_predictions = []

    i = -len(y_val)

    while len(validation_predictions) < len(validation_target):
        
        # Predice el siguiente valor de X[i]
        p = model.predict(X[i].reshape(1, X.shape[1], X.shape[2]))[0, 0]
        i += 1
        
        validation_predictions.append(p)

    pred_real, target_real = desescalado(validation_target, validation_predictions, scaler)

    return pred_real, target_real


def predict_multi_rnn(X, X_val, y_val, model, scaler):

    validation_target_multi = y_val
    validation_predictions_multi = []

    # Usa la primera ventana de validación (X_val[0])
    last_x = X_val[0].copy()  # La primera secuencia de validación

    for i in range(len(validation_target_multi)):
        # Predicción del siguiente paso
        p = model.predict(last_x.reshape(1, X.shape[1], X.shape[2]))[0, 0]
        
        validation_predictions_multi.append(p)
        print(f"Paso {i+1} -> Valor real: {validation_target_multi[i]:.4f} | Predicción: {p:.4f}")

        # Desplaza la ventana hacia atrás e inserta la nueva predicción
        last_x[:-1] = last_x[1:]  # Mueve los datos hacia atrás
        last_x[-1, 0] = p  # Inserta la nueva predicción como último valor

    pred_real_multi, target_real_multi = desescalado(validation_target_multi, validation_predictions_multi, scaler)

    return pred_real_multi, target_real_multi


def predict_noise_rnn(X, X_val, y_val, model, scaler, sigma=0.05):

    validation_target_noise = y_val  # Última parte de los datos reales
    validation_predictions_noise = []

    # Usar la primera ventana de validación
    last_x = X_val[0].copy()

    for i in range(len(validation_target_noise)):  
        # Predicción sin ruido
        p = model.predict(last_x.reshape(1, last_x.shape[0], X_val.shape[2]))[0, 0]
        
        # Agregar ruido gaussiano proporcional
        noise = np.random.normal(loc=0, scale=sigma * abs(p), size=1)[0]  # Proporcional al valor predicho
        p_noisy = p + noise
        
        validation_predictions_noise.append(p_noisy)
        # print(f"Paso {i+1} -> Valor real: {validation_real_noise[i]:.4f} | Predicción con ruido: {p_noisy:.4f}")

        # Desplazar la ventana y actualizar el último valor con la predicción ruidosa
        last_x[:-1] = last_x[1:]  # Mueve los datos hacia atrás
        last_x[-1, 0] = p_noisy  # Inserta la predicción con ruido como último valor

        pred_real_noise, target_real_noise = desescalado(validation_target_noise, validation_predictions_noise, scaler)

#-----------------------------------------------------------------------------------------

# Funciones para plotear y métricas

def plot_validation(validation_target, validation_predictions):

    fig_pred = px.line(
                       y=[validation_target, validation_predictions],
                       title='Validación de las predicciones frente a valores reales',
                       labels={'index': 'Día', 'value': 'Demanda (GWh)'},
                       )

    fig_pred.update_layout(
        title_x=0.5,
        legend_title_text="Variables"
    )

    fig_pred.for_each_trace(lambda t: t.update(name="Demanda real" if t.name == "wide_variable_0" else "Predicción"))

    return fig_pred


def metricas_rnn_lstm(validation_target, validation_predictions):

    mse = round(mean_squared_error(validation_target, validation_predictions),2)
    # print(f"MSE: {mse}")

    mae = round(mean_absolute_error(validation_target, validation_predictions), 2)
    # print(f"MAE: {mae}")

    rmse = round(np.sqrt(mse), 2)
    # print(f"RMSE: {rmse}")

    r2 = round(r2_score(validation_target, validation_predictions), 2)

    dict_metricas = {
        "r2" : [r2],
        "mae_GWh" : [mae],
        "rmse_GWh" : [rmse]
    }

    return dict_metricas


def plot_predictions(validation_target, validation_predictions):

    # Graficar
    fig_pred = px.line(
                        y=[validation_target, validation_predictions],
                        title='Predicción de la demanda en 14 días',
                        labels={'x': 'Día', 'y': 'Demanda (GWh)'}

                        )

    fig_pred.update_layout(
        title_x=0.5,
        legend_title_text="Variables"
    )

    fig_pred.for_each_trace(lambda t: t.update(name="Demanda real" if t.name == "wide_variable_0" else "Predicción"))

    fig_pred.show()


def plot_n_future(predictions):

    fig_one = px.line(
                        y=predictions,
                        title=f'Predicción de la demanda en {len(predictions)} días',
                        labels={'x': 'Día', 'y': 'Demanda (GWh)'}

                        )

    fig_one.update_layout(
        title_x=0.5,
        legend_title_text="Variables"
    )

    fig_one.for_each_trace(lambda t: t.update(name="Demanda real" if t.name == "wide_variable_0" else "Predicción"))

    fig_one.show()


#-----------------------------------------------------------------------------------------
# predicciones 1 paso y multistep para n_future días

def prediction_1step_rnn(X, y_val, scaler, model, n_future):

    predictions = model.predict(X[-n_future:])

    one_real = y_val[-n_future:]

    predictions_real_one = desescalado(one_real, predictions, scaler)[1]

    return predictions_real_one


def prediction_multistep_rnn(X, y_val, scaler, model, n_future):

    multi_real = y_val[-n_future:]  # Últimos valores reales para comparación
    predictions = []
    
    # Última ventana de entrada (asegurar la forma correcta)
    last_x = X[-1].copy()  
    if last_x.ndim == 1:
        last_x = last_x.reshape(-1, 1)  # Convertir a (lookback, 1) si es necesario

    for _ in range(n_future):
        # Predecir el siguiente valor asegurando que la entrada tenga la forma correcta
        p = model.predict(last_x.reshape(1, last_x.shape[0], last_x.shape[1]))[0, 0]
        predictions.append(p)

        # Desplazar la ventana sin alterar su forma
        last_x = np.roll(last_x, -1, axis=0)  # Desplazar una posición
        last_x[-1, 0] = p  # Sustituir el último valor con la predicción

    # Desescalar predicciones
    predictions_real_multi = desescalado(multi_real, predictions, scaler)[1]

    return predictions_real_multi


