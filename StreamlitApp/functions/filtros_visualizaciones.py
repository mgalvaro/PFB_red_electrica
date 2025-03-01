from datetime import datetime, timedelta
import pandas as pd

variables = {

        'Demanda': 1,
        'Balance': 2,
        'Generación': 3,
        'Intercambios': 4
        
    }

renombrar = {
    'mean': 'Media (GWh)',
    'median': 'Mediana (GWh)',
    'max': 'Máximo (GWh)',
    'min': 'Mínimo (GWh)'
}

def p7_30_365_hist(df, p):

    """
    Filtra el dataframe según el período temporal deseado

    Parámetros:
    df (pandas.Dataframe): el dataframe completo que se quiere filtrar
    p (int): el período de tiempo especificado

    Retorna:
    df: el dataframe filtrado
    p: si se pide el histórico, devuelve el total de días de los que hay registros
    """

    if p > 0:
        # today = datetime.strptime(df['fecha'].max(), "%Y-%m-%d")
        today = df['fecha'].max()
        delta = timedelta(days=p)
        start_date = (today - delta)#.strftime("%Y-%m-%d")
        end_date = today#.strftime("%Y-%m-%d")

        df = df[(start_date <= df['fecha']) & (df['fecha']<= end_date)].reset_index(drop=True)
        
    else:
        
        df = df
        p = round(len(df)/365, 1)

    df['valor_(MWh)'] = df['valor_(MWh)'].apply(lambda x : x * 0.001)
    df = df.rename(columns={'valor_(MWh)': 'valor_(GWh)'})

    return df


def p_365_mas(df, p):

    df['fecha'] = pd.to_datetime(df['fecha'])
    df['fecha'] = df['fecha'].dt.to_period(p)
    df = df.groupby(['fecha', 'zona']).agg({'valor_(GWh)': 'mean'}).reset_index(drop=False)
    df['fecha'] = df['fecha'].astype(str)

    return df



def filtro_intercambios(df, year1, year2):

    """
    Filtra el df de los intercambios energéticos entre países 
    para quedarse con los datos de los 2 años elegidos

    Parámetros:
    df (pandas.Dataframe): el dataframe completo que se quiere filtrar
    year1, year2 (int): los años a comparar

    Retorna:
    df filtrado
    """

    df = df[df['año'].isin([year1, year2])]
    df = df[df['frontera'].isin(['Francia', 'Portugal', 'Marruecos', 'Andorra'])].reset_index(drop=True)
    df = df.groupby(['año', 'mes', 'dia', 'tipo'], as_index=False).agg({'valor_(MWh)': 'sum'})
    df['fecha_sin_year'] = pd.to_datetime(df[['mes', 'dia']].rename(columns={'mes': 'month', 'dia': 'day'}).assign(year=2000))
    df['valor_(MWh)'] = df['valor_(MWh)'].apply(lambda x: abs(x))
    df['valor_(GWh)'] = df['valor_(MWh)'].apply(lambda x: x*0.001)

    df_estadisticas = df.groupby(['año', 'tipo'])['valor_(GWh)'].agg(['mean', 'median', 'max', 'min']).reset_index(drop=False).rename(columns=renombrar)
    cols_to_round = df_estadisticas.columns.difference(['año'])
    df_estadisticas[cols_to_round] = df_estadisticas[cols_to_round].round(1)

    # df['fecha_sin_year'] = df['fecha_sin_year'].dt.to_period('W')
    # df = df.groupby(['fecha_sin_year', 'año']).agg({'valor_(GWh)': 'mean'}).reset_index(drop=False)
    # df['fecha_sin_year'] = df['fecha_sin_year'].astype(str)
    
    return df, df_estadisticas



def filtro_comparador(df, year1, year2):

    """
    Filtra el df introducido (generación, demanda o balance)
    para quedarse con los datos de los 2 años elegidos

    Parámetros:
    df (pandas.Dataframe): el dataframe completo que se quiere filtrar
    year1, year2 (int): los años a comparar

    Retorna:
    df filtrado
    """
    print(df.columns)
    df = df[df['año'].isin([year1, year2])]
    df = df.groupby(['año', 'mes', 'dia'], as_index=False).agg({'valor_(MWh)': 'sum'})
    df['fecha_sin_year'] = pd.to_datetime(df[['mes', 'dia']].rename(columns={'mes': 'month', 'dia': 'day'}).assign(year=2000))
    df['valor_(GWh)'] = df['valor_(MWh)'].apply(lambda x: x*0.001)

    df_estadisticas = df.groupby('año')['valor_(GWh)'].agg(['mean', 'median', 'max', 'min']).reset_index(drop=False).rename(columns=renombrar)
    cols_to_round = df_estadisticas.columns.difference(['año'])
    df_estadisticas[cols_to_round] = df_estadisticas[cols_to_round].round(1)

    # df['fecha_sin_year'] = df['fecha_sin_year'].dt.to_period('W')
    # df = df.groupby(['fecha_sin_year']).agg({'valor_(GWh)': 'mean'}).reset_index(drop=False)
    # df['fecha_sin_year'] = df['fecha_sin_year'].astype(str)


    return df, df_estadisticas

def calculo_balance(df1, df2, year1, year2):

    df1 = df1[(df1['año'] == year1) | (df1['año'] == year2)].reset_index(drop=True)
    df2 = df2[(df2['año'] == year1) | (df2['año'] == year2)].reset_index(drop=True)

    df1 = df1[df1['zona'] == 'nacional']
    df2 = df2[df2['zona'] == 'nacional']

    df1 = df1.groupby('fecha').sum('valor_(MWh)').reset_index(drop=False)
    df2 = df2.groupby('fecha').sum('valor_(MWh)').reset_index(drop=False)

    df = pd.merge(df1, df2, on='fecha', how='left')

    df['valor_(GWh)'] = (df['valor_(MWh)_y'] - df['valor_(MWh)_x']) * 0.001
    df = df[['fecha', 'año_x', 'mes_x', 'dia_x', 'valor_(GWh)']].rename(columns={'año_x': 'año', 'mes_x': 'mes', 'dia_x': 'dia'})

    df['fecha_sin_year'] = pd.to_datetime(df[['mes', 'dia']].rename(columns={'mes': 'month', 'dia': 'day'}).assign(year=2000))

    df_estadisticas = df.groupby('año')['valor_(GWh)'].agg(['mean', 'median', 'max', 'min']).reset_index(drop=False).rename(columns=renombrar)
    cols_to_round = df_estadisticas.columns.difference(['año'])
    df_estadisticas[cols_to_round] = df_estadisticas[cols_to_round].round(1)

    # df['fecha_sin_year'] = df['fecha_sin_year'].dt.to_period('W')
    # df = df.groupby(['fecha_sin_year']).agg({'valor_(GWh)': 'mean'}).reset_index(drop=False)
    # df['fecha_sin_year'] = df['fecha_sin_year'].astype(str)

    return df, df_estadisticas


