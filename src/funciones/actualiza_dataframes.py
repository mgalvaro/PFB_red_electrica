import pandas as pd
from datetime import datetime
from time import sleep

from funciones.extraccion_balance import *
from funciones.extraccion_demanda import *
from funciones.extraccion_generacion import *
from funciones.extraccion_intercambios import *

from funciones.divide_fecha import *
from funciones.limpia_columnas import *  

from funciones.limpia_intercambio import * 
from funciones.limpia_balance import *
from funciones.limpia_generacion import *
from funciones.limpia_demanda import *

from funciones.outliers_tukey import *

def actualiza_dfs() -> tuple:

    df_balance_historico = pd.read_csv("../data/processed/DF_BALANCE_10_25_LIMPIO.csv")
    df_demanda_historico = pd.read_csv("../data/processed/DF_DEMANDA_10_25_LIMPIO.csv")
    df_generacion_historico = pd.read_csv("../data/processed/DF_GENERACION_10_25_LIMPIO.csv")
    df_intercambios_historico = pd.read_csv("../data/processed/DF_INTERCAMBIOS_10_25_LIMPIO.csv")

    hoy = datetime.now() 
    fin = (hoy -pd.Timedelta(days=1)).strftime("%Y-%m-%dT23:59")

    ultima_extraccion = {}
    fechas_faltantes = {}

    df_dict = {
        "balance": df_balance_historico,
        "demanda": df_demanda_historico,
        "generacion": df_generacion_historico,
        "intercambios": df_intercambios_historico
    }

    for key, value in df_dict.items():
        
        dias_faltantes = []
        
        ultima_extraccion[key] = pd.to_datetime(value["fecha_extraccion"].sort_values(ascending=False)[0])

        while ultima_extraccion[key] < hoy:
            dias_faltantes.append(ultima_extraccion[key].strftime('%Y-%m-%d'))
            ultima_extraccion[key] += pd.Timedelta(days=1)
                
        fechas_faltantes[key] = dias_faltantes

        inicio = sorted(dias_faltantes)[0]

        fechas = {inicio : fin}

    for inicio, fin in fechas.items():

        #balance (extraccion y limpieza)
        df_balance = extrae_balance(inicio, fin)
        df_balance = divide_fecha(df_balance)
        df_balance = limpia_columnas(df_balance)
        df_balance = limpia_balance(df_balance)

        df_balance_historico = pd.concat([df_balance_historico, df_balance])

        df_balance_historico_bis = df_balance_historico[~df_balance_historico.duplicated(subset=["titulo", "zona", "fecha"], keep="last")]
        df_balance_historico_bis = df_balance_historico_bis.reset_index(drop="index")

        df_balance_historico_bis[["fecha_extraccion", "fecha"]] = df_balance_historico_bis[["fecha_extraccion", "fecha"]].astype("datetime64[ns]")
        sleep(1)

        #demanda (extraccion, limpieza y sustitucion outliers - si hay)
        df_demanda = extrae_demanda(inicio, fin)
        df_demanda = divide_fecha(df_demanda)
        df_demanda = limpia_columnas(df_demanda)
        df_demanda = limpia_demanda(df_demanda)
        
        años = set([inicio[:4],int(fin[:4])])
        territorios = [territorio for territorio in df_demanda_historico["zona"].unique()]

        for año in años:
            for territorio in territorios:
                try:
                    df_demanda_historico.loc[(df_demanda_historico["año"] == año) \
                                            & (df_demanda_historico["zona"] == territorio)\
                                            &(df_demanda_historico["categoria"] == "Evolución de la demanda"), "valor_(MWh)"] \
                                            = sustituye_outliers_tukey(df_demanda_historico.loc[(df_demanda_historico["año"] == año) \
                                            & (df_demanda_historico["zona"] == territorio)\
                                            &(df_demanda_historico["categoria"] == "Evolución de la demanda"), "valor_(MWh)"])
                except IndexError:
                    print(f"No hay datos que sobreescribir en zona {territorio} entre las fechas {inicio} y {fin}")
        
        df_demanda_historico = pd.concat([df_demanda_historico, df_demanda])

        df_demanda_historico_bis = df_demanda_historico[~df_demanda_historico.duplicated(subset=["titulo", "zona", "fecha"], keep="last")]
        df_demanda_historico_bis = df_demanda_historico_bis.reset_index(drop="index")

        df_demanda_historico_bis[["fecha_extraccion", "fecha"]] = df_demanda_historico_bis[["fecha_extraccion", "fecha"]].astype("datetime64[ns]")
        sleep(1)

        #generacion (extraccion y limpieza)
        df_generacion = extrae_generacion(inicio, fin)
        df_generacion = divide_fecha(df_generacion)
        df_generacion = limpia_columnas(df_generacion)
        df_generacion = limpia_generacion(df_generacion)

        df_generacion_historico = pd.concat([df_generacion_historico, df_generacion])

        df_generacion_historico_bis = df_generacion_historico[~df_generacion_historico.duplicated(subset=["titulo", "zona", "fecha"], keep="last")]
        df_generacion_historico_bis = df_generacion_historico_bis.reset_index(drop="index")

        df_generacion_historico_bis[["fecha_extraccion", "fecha"]] = df_generacion_historico_bis[["fecha_extraccion", "fecha"]].astype("datetime64[ns]")
        sleep(1)

        #intercambios (extraccion y limpieza)
        df_intercambios = extrae_intercambios(inicio, fin)
        df_intercambios = divide_fecha(df_intercambios)
        df_intercambios = limpia_columnas(df_intercambios)
        df_intercambios = limpia_intercambio(df_intercambios)

        df_intercambios_historico = pd.concat([df_intercambios_historico, df_intercambios])

        df_intercambios_historico_bis = df_intercambios_historico[~df_intercambios_historico.duplicated(subset=["titulo", "zona", "frontera", "fecha"], keep="last")]
        df_intercambios_historico_bis = df_intercambios_historico_bis.reset_index(drop="index")

        df_intercambios_historico_bis[["fecha_extraccion", "fecha"]] = df_intercambios_historico_bis[["fecha_extraccion", "fecha"]].astype("datetime64[ns]")
        sleep(1)
    
    df_balance_historico_bis_2 = df_balance_historico_bis[~df_balance_historico_bis.duplicated(subset=["titulo", "zona", "fecha"], keep="last")]
    df_balance_historico_bis_2 = df_balance_historico_bis_2.reset_index(drop="index")

    df_demanda_historico_bis_2 = df_demanda_historico_bis[~df_demanda_historico_bis.duplicated(subset=["titulo", "zona", "fecha"], keep="last")]
    df_demanda_historico_bis_2 = df_demanda_historico_bis_2.reset_index(drop="index")

    df_generacion_historico_bis_2 = df_generacion_historico_bis[~df_generacion_historico_bis.duplicated(subset=["titulo", "zona", "fecha"], keep="last")]
    df_generacion_historico_bis_2 = df_generacion_historico_bis_2.reset_index(drop="index")

    df_intercambios_historico_bis_2 = df_intercambios_historico_bis[~df_intercambios_historico_bis.duplicated(subset=["titulo", "zona", "frontera", "fecha"], keep="last")]
    df_intercambios_historico_bis_2 = df_intercambios_historico_bis_2.reset_index(drop="index")

    return df_balance_historico_bis_2, df_demanda_historico_bis_2, df_generacion_historico_bis_2, df_intercambios_historico_bis_2