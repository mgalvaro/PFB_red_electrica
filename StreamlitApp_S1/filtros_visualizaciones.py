from datetime import datetime, timedelta

def p7_30_365_hist(df, p):

    """
    Filtra el dataframe según el período temporal deseado

    Parámetros:
    df (pandas.Dataframe): el dataframe completo que se quiere filtrar
    p (int): el período de tiempo especificado

    Retorna:
    df_filtered: el dataframe filtrado
    p: si se pide el histórico, devuelve el total de días de los que hay registros
    """

    if p > 0:
        today = datetime.strptime(df['fecha'].max(), "%Y-%m-%d")
        delta = timedelta(days=p)
        start_date = (today - delta).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

        df_filtered = df[(start_date <= df['fecha']) & (df['fecha']<= end_date)].reset_index(drop=True)

    else:
        
        df_filtered = df
        p = len(df_filtered)

    return df_filtered, p

def filtro_intercambios(df, year1, year2):

    df = df[(df['año'] == year1) | (df['año'] == year2)]
    df = df[df['frontera'].isin(['Francia', 'Portugal', 'Marruecos', 'Andorra'])].reset_index(drop=True)
    df = df.groupby(['mes', 'tipo', 'año']).sum('valor_(MWh)').reset_index()
    df['valor_(MWh)'] = df['valor_(MWh)'].apply(lambda x: abs(x))
    
    return df

