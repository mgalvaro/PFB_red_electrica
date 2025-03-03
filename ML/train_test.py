import pandas as pd

def train_test(df):
    # Ordenar por año, mes y día respetando el orden temporal
    df = df.sort_values(by=["fecha"])
    df = df.drop(columns="fecha")

    # Dividir en train (80%) y test (20%)
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    x_train = df_train.drop(columns="valor_(GWh)")
    y_train = df_train["valor_(GWh)"]

    x_test = df_test.drop(columns="valor_(GWh)")
    y_test = df_test["valor_(GWh)"]

    return x_train, x_test, y_train, y_test