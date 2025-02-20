
def divide_fecha(dataframe):

    dataframe["año"] = dataframe["datetime"].str[:4].astype("int32")
    dataframe["mes"] = dataframe["datetime"].str[5:7].astype("int32")
    dataframe["dia"] = dataframe["datetime"].str[8:10].astype("int32")
    dataframe["hora"] = dataframe["datetime"].str[11:13].astype("int32")

    return dataframe

#_________________________________________________________________________________________________________________________

def limpia_columnas(dataframe):
    
    columnas_renombradas = {
        "datetime": "fecha",
        "description": "descripcion",
        "last-update": "ultima_actualizacion",
        "percentage": "porcentaje",
        "title": "energia",
        "total": "total_zona",
        "type": "tipo",
        "value": "valor_kWh"
    }

    dataframe = dataframe.rename(columns=columnas_renombradas)
    dataframe["ID"] = dataframe["Unnamed: 0"].apply(
        lambda x: str(dataframe["extraccion"].iloc[0]).replace("-", "").replace(":", "").replace("T", "") + str(x)
    ).astype("int64")
    dataframe["composite"] = dataframe["composite"].map({True:1, False:0})
    dataframe = dataframe.drop(columns=["magnitude", "Unnamed: 0", "color", "descripcion"])

    return dataframe
