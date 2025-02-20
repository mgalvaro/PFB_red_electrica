import pandas as pd

def limpia_balance(dataframe) -> pd.DataFrame:

    dataframe = dataframe.copy()
    
    titulos_renovables = ['Hidráulica',
        'Hidroeólica',
        'Eólica',
        'Solar fotovoltaica',
        'Solar térmica',
        'Otras renovables',
        'Residuos renovables']
    
    titulos_no_renovables = ['Nuclear',
        'Carbón',
        'Motores diésel',
        'Turbina de gas',
        'Turbina de vapor',
        'Ciclo combinado',
        'Cogeneración',
        'Residuos no renovables']
    
    almacenamiento = ["Turbinación bombeo",
        "Consumos bombeo",
        "Entrega batería",
        "Carga batería"
    ]

    #sobreescribo "tipo" para categorizar las titulos y sacar totales y graficos más fácilmente. Así rellenamos también los NaN que corresponde a Saldos I. Internacionales
    dataframe["tipo"] = dataframe["titulo"].apply(lambda x: "Renovable" if x in titulos_renovables else\
                                                    "No renovable" if x in titulos_no_renovables else \
                                                    "Almacenamiento" if x in almacenamiento else x)

    #descartamos del dataframe las columnas que son cálculos totales
    dataframe = dataframe[(dataframe["titulo"] != "Generación renovable")]
    dataframe = dataframe[(dataframe["titulo"] != "Generación no renovable")]
    dataframe = dataframe[(dataframe["titulo"] != "Saldo almacenamiento")]
    dataframe = dataframe[(dataframe["titulo"] != "Demanda en b.c.")]   

    return dataframe