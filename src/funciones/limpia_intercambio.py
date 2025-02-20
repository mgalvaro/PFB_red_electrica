import pandas as pd

def limpia_intercambio(dataframe) -> pd.DataFrame:

    dataframe = dataframe.copy()

    dataframe["frontera"] = dataframe["frontera"].fillna("Enlace Península-Baleares")
    dataframe = dataframe[dataframe["titulo"] != "saldo"]

    return dataframe