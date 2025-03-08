import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from holidays import Spain

#  Rutas de archivos
ruta_modelo = r"C:\Users\nacho\Desktop\Aprendiendo a programar\PFB\TFB\ML\MODELS\modelo_prophet_con.pkl"
ruta_scaler = r"C:\Users\nacho\Desktop\Aprendiendo a programar\PFB\TFB\data\data_scaled\scalers\scaler_consumo_anio_DF_DEMANDA.pkl"
ruta_datos_historicos = r"C:\Users\nacho\Desktop\Aprendiendo a programar\PFB\TFB\data\data_scaled\DF_DEMANDA_10_25_PROCESADO.csv"


with open(ruta_modelo, "rb") as f:
    modelo_guardado = pickle.load(f)

modelo = modelo_guardado["modelo"]


with open(ruta_scaler, "rb") as f:
    scaler = pickle.load(f)


df_historico = pd.read_csv(ruta_datos_historicos, parse_dates=["fecha"])
df_historico = df_historico.rename(columns={"fecha": "ds", "valor_(GWh)": "y"})

#  Interfaz en Streamlit
st.title("Predicción de Demanda con Prophet")

#  Selección de granularidad en Streamlit
granularidad = st.selectbox(
    "Selecciona la granularidad",
    ["Día a día", "Semana a semana", "Mes a mes", "Trimestre a trimestre"]
)

#  Generar fechas futuras
df_futuro = modelo.make_future_dataframe(periods=365, freq="D")

#  Agregar la variable exógena es_festivo
años_prediccion = df_futuro["ds"].dt.year.unique()
festivos = Spain(years=años_prediccion)
df_futuro["es_festivo"] = df_futuro["ds"].apply(lambda x: 1 if x in festivos else 0)

#  Hacer predicciones con es_festivo
predicciones = modelo.predict(df_futuro)[["ds", "yhat"]]

#   Desescalar valores correctamente usando el MinMaxScaler 
dummy_feature = np.zeros((predicciones.shape[0], 1))  
predicciones_scaled = np.hstack([predicciones[["yhat"]].values, dummy_feature])  
predicciones["yhat"] = scaler.inverse_transform(predicciones_scaled)[:, 0]  

#  ** Desescalar también los datos históricos **
dummy_feature_hist = np.zeros((df_historico.shape[0], 1))  
historico_scaled = np.hstack([df_historico[["y"]].values, dummy_feature_hist])  
df_historico["y"] = scaler.inverse_transform(historico_scaled)[:, 0]  

#  Aplicar granularidad después de predecir
def aplicar_granularidad(df, frecuencia):
    conversion = {"D": "D", "W": "W", "M": "ME", "Q": "QE"}
    return df.groupby(pd.Grouper(key="ds", freq=conversion[frecuencia])).mean().reset_index()

frecuencias = {"Día a día": "D", "Semana a semana": "W", "Mes a mes": "M", "Trimestre a trimestre": "Q"}

#  **Colores optimizados**
colores = {
    "Día a día": "#2ca02c",  # Verde más claro
    "Semana a semana": "#ff7f0e",  # Naranja
    "Mes a mes": "#1f77b4",  # Azul fuerte
    "Trimestre a trimestre": "#d62728"  # Rojo más oscuro
}

predicciones = aplicar_granularidad(predicciones, frecuencias[granularidad])
df_historico = aplicar_granularidad(df_historico, frecuencias[granularidad])


fig = go.Figure()


fig.add_trace(go.Scatter(
    x=df_historico["ds"],
    y=df_historico["y"],
    mode="lines",
    name="Datos Reales",
    line=dict(color='#444444', width=2)  
))


fig.add_trace(go.Scatter(
    x=predicciones["ds"],
    y=predicciones["yhat"],
    mode="lines",
    name=f"Predicción ({granularidad})",
    line=dict(color=colores[granularidad], width=3, dash='dot')
))

fig.update_layout(
    title=f" Predicción de Demanda ({granularidad})",
    xaxis_title="Fecha",
    yaxis_title="Demanda (GWh)",
    template="plotly_white",
    hovermode="x"
)

st.plotly_chart(fig)

st.success("Predicción con es_festivo integrada correctamente")