import pandas as pd

def balance_energetico_total(fecha_inicio, fecha_fin, zona) -> pd.DataFrame:

    df_balance = pd.read_csv("../data/processed/DF_BALANCE_20_25_LIMPIO_V1.csv")

    balance_total = df_balance[((df_balance["fecha"] >= fecha_inicio)&(df_balance["fecha"]<= fecha_fin))&(df_balance["zona"] == zona)].groupby("fecha").agg({"valor_(MWh)":"sum"})
    balance_total = balance_total.rename(columns={"valor_(MWh)":f"total_balance_{zona}_(MWh)"})
    
    return balance_total

def demanda_energetica_total(fecha_inicio, fecha_fin, zona) -> pd.DataFrame:

    demanda_total = pd.DataFrame()

    df_demanda = pd.read_csv("../data/processed/DF DEMANDA_20_25_LIMPIO_V1.csv")
    df_generacion = pd.read_csv("../data/processed/DF GENERACION_20_25_LIMPIO_V1.csv")
    df_intercambio = pd.read_csv("../data/processed/DF INTERCAMBIOS_20_25_LIMPIO_V1.csv")

    generacion = df_generacion[((df_generacion["fecha"] >= fecha_inicio)&(df_generacion["fecha"]<= fecha_fin))&(df_generacion["zona"] == "nacional")].groupby("fecha").agg({"valor_(MWh)":"sum"})
    generacion = generacion.rename(columns={"valor_(MWh)":f"total_generacion_{zona}_(MWh)"})

    perdidas_transporte = df_demanda[((df_demanda["fecha"] >= fecha_inicio)&(df_demanda["fecha"]<= fecha_fin)&(df_demanda["zona"] == zona))&(df_demanda["titulo"] == "Pérdidas de transporte")].groupby("fecha").agg({"valor_(MWh)":"sum"})
    perdidas_transporte = perdidas_transporte.rename(columns={"valor_(MWh)":f"total_perdidas_transporte_{zona}_(MWh)"})

    if zona != "baleares":
        intercambios = df_intercambio[((df_intercambio["fecha"] >= fecha_inicio)&(df_intercambio["fecha"]<= fecha_fin)&(df_intercambio["zona"] == zona))][:-2].groupby("fecha").agg({"valor_(MWh)":"sum"})
        intercambios = intercambios.rename(columns={"valor_(MWh)":f"total_intercambios_{zona}_(MWh)"})
    else:
        intercambios = df_intercambio[((df_intercambio["fecha"] >= fecha_inicio)&(df_intercambio["fecha"]<= fecha_fin)&(df_intercambio["zona"] == zona))].groupby("fecha").agg({"valor_(MWh)":"sum"})
        intercambios = intercambios.rename(columns={"valor_(MWh)":f"total_intercambios_{zona}_(MWh)"})

    demanda_total = pd.concat([generacion, perdidas_transporte, intercambios], axis=1)

    demanda_total[f"demanda_total_{zona}_(MWh)"] = demanda_total.sum(axis=1)
    
    return demanda_total