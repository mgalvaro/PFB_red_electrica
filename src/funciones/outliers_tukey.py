import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualiza_outliers_tukey(dataframe, año, territorio, k=1.5) -> None:

    df_año = dataframe[(dataframe["año"] == año)&(dataframe["zona"] == territorio)].copy()

    q1 = np.quantile(df_año["valor_(MWh)"], 0.25)
    q3 = np.quantile(df_año["valor_(MWh)"], 0.75)
    iqr = q3 - q1
    lim_l = q1 - k * iqr
    lim_r = q3 + k * iqr

    df_año["es_outlier"] = (df_año["valor_(MWh)"] < lim_l) | (df_año["valor_(MWh)"] > lim_r)

    plt.figure(figsize=(15, 2))
    sns.histplot(data=df_año, x="valor_(MWh)", hue="es_outlier", bins=30, palette={False: "blue", True: "red"})
    plt.title(f"Demanda en el año {año} en territorio {territorio}")
    plt.xlabel("Demanda (MWh)")
    plt.ylabel("Frecuencia")
    plt.legend(["Outlier", "Normal"])
    plt.show()

    return None

def aisla_outliers_tukey(dataframe, año, territorio, k=1.5) -> pd.DataFrame:

    df_año = dataframe[(dataframe["año"] == año)\
                        &(dataframe["zona"] == territorio)\
                        &(dataframe["categoria"] == "Evolución de la demanda")].copy()

    q1 = np.quantile(df_año["valor_(MWh)"], 0.25)
    q3 = np.quantile(df_año["valor_(MWh)"], 0.75)
    iqr = q3 - q1
    lim_l = q1 - k * iqr
    lim_r = q3 + k * iqr

    df_año["es_outlier"] = (df_año["valor_(MWh)"] < lim_l) | (df_año["valor_(MWh)"] > lim_r)

    return df_año[df_año["es_outlier"]]

def sustituye_outliers_tukey(array, k=1.5) -> np.array:

    q1 = np.quantile(array, 0.25)
    q3 = np.quantile(array, 0.75)

    ric = q3 - q1
    
    lim_l = q1 - k*ric
    lim_r = q3 + k*ric

    array_modificado = np.clip(array, lim_l, lim_r)
    
    return array_modificado