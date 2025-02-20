import pandas as pd

def limpia_generacion(dataframe) -> pd.DataFrame:

    dataframe = dataframe.copy()

    dataframe["tipo"] = dataframe["tipo"].replace("No-Renovable","No renovable")
    dataframe = dataframe[dataframe["titulo"] != "Generación total"]

    return dataframe