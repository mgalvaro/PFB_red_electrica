import numpy as np

# definimos los días que tiene cada mes
n_dias = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31
}

def encoder(df_filtrado):
    # Inicialmente, codificamos de forma lineal los días de la semana para luego aplicar la codificación circular
    df_filtrado["dia_semana"] = df_filtrado["dia_semana"].map({"lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3, "viernes": 4, "sábado": 5, "domingo": 6})

    df_filtrado["dia_semana_sin"] = np.sin(2 * np.pi * df_filtrado["dia_semana"] / 7)
    df_filtrado["dia_semana_cos"] = np.cos(2 * np.pi * df_filtrado["dia_semana"] / 7)

    # codificamos los meses
    df_filtrado["mes_sin"] = np.sin(2 * np.pi * df_filtrado["mes"] / 12)
    df_filtrado["mes_cos"] = np.cos(2 * np.pi * df_filtrado["mes"] / 12)

    # codificamos los días del mes en función de cada mes y teniendo en cuenta los bisiestos
    dia_sin = []
    dia_cos = []

    for i, row in df_filtrado.iterrows():

        if row['mes'] != 2 or row['año'] % 4 != 0:  # tiene en cuenta meses distintos a febrero, o febrero en un año no bisiesto
            dia_sin.append(np.sin(2 * np.pi * row['dia'] / n_dias[row['mes']]))
            dia_cos.append(np.cos(2 * np.pi * row['dia'] / n_dias[row['mes']]))
        else:  # febrero en año bisiesto
            dia_sin.append(np.sin(2 * np.pi * row['dia'] / 29))
            dia_cos.append(np.cos(2 * np.pi * row['dia'] / 29))

    df_filtrado['dia_mes_sin'] = dia_sin
    df_filtrado['dia_mes_cos'] = dia_cos

    df_filtrado.drop(columns=["dia_semana", "mes", "dia"], inplace=True)

    return df_filtrado