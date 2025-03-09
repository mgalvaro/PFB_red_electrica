import pandas as pd
import pickle as pkl
import numpy as np
import sys
import os
from holidays import Spain
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from StreamlitApp.functions.carga_dataframes import *
from ML.escalado_datos import *
from StreamlitApp.passwords import pw

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import GRU, Dense, Input, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# Cargar el scaler
with open("../data/data_scaled/scalers/scaler_consumo_anio_DF_DEMANDA.pkl", "br") as file:
    scaler = pkl.load(file)


def predice_prophet(model, dataframe):

    dataframe = dataframe.rename(columns={"fecha": "ds", "valor_(GWh)": "y"})

    #  Generar fechas futuras
    df_futuro = model.make_future_dataframe(periods=365, freq="D")

    #  Agregar la variable exógena es_festivo
    años_prediccion = df_futuro["ds"].dt.year.unique()
    festivos = Spain(years=años_prediccion)
    df_futuro["es_festivo"] = df_futuro["ds"].apply(lambda x: 1 if x in festivos else 0)

    #  Hacer predicciones con es_festivo
    predicciones = model.predict(df_futuro)[["ds", "yhat"]]

    #   Desescalar valores correctamente usando el MinMaxScaler 
    dummy_feature = np.zeros((predicciones.shape[0], 1))  
    predicciones_scaled = np.hstack([predicciones[["yhat"]].values, dummy_feature])  
    predicciones["yhat"] = scaler.inverse_transform(predicciones_scaled)[:, 0]  
    
    return predicciones