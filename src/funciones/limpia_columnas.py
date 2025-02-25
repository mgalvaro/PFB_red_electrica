import pandas as pd
import uuid

def limpia_columnas(dataframe) -> pd.DataFrame:

    columnas_renombradas = {
        "datetime": "fecha",
        "percentage": "porcentaje",
        "title": "titulo",
        "type": "tipo",
        "value": "valor_(MWh)",
        "extraccion": "fecha_extraccion",
    }
    
    dias_semana = {
        0: "lunes",
        1: "martes",
        2: "miércoles",
        3: "jueves",
        4: "viernes",
        5: "sábado",
        6: "domingo",
    }
    
    dataframe = dataframe.rename(columns=columnas_renombradas)
   
    dataframe["fecha"] = dataframe["fecha"].apply(lambda x: x[:10].replace("-",""))
    dataframe["fecha_extraccion"] = dataframe["fecha_extraccion"].apply(lambda x: x[:10])
    dataframe["dia_semana"] = pd.to_datetime(dataframe["fecha"]).dt.dayofweek.map(dias_semana)

    dataframe["ID"] = [uuid.uuid4().hex for _ in range(len(dataframe))]
    dataframe["ID"] = dataframe["fecha"] + "-" + dataframe["ID"]

    dataframe["composite"] = dataframe["composite"].map({True: 1, False: 0})
    
    if "total" in dataframe.columns:
        dataframe = dataframe.drop(columns=["magnitude", "color", "description", "last-update", "total"])
    else:
        dataframe = dataframe.drop(columns=["magnitude", "color", "description", "last-update"])

    #dataframe["fecha"] = pd.to_datetime(dataframe["fecha"])
    #dataframe["fecha_extraccion"] = pd.to_datetime(dataframe["fecha_extraccion"])

    return dataframe
