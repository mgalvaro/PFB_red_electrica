import pandas as pd

def limpia_demanda(dataframe) -> pd.DataFrame:

    dataframe = dataframe.copy()

    dataframe = dataframe.drop(columns="tipo")

    demanda_duplicada_idxs = dataframe[(dataframe["categoria"] == "Pérdidas de transporte")&(dataframe["titulo"] == "Demanda")].index
    dataframe = dataframe.drop(index=demanda_duplicada_idxs)

    return dataframe