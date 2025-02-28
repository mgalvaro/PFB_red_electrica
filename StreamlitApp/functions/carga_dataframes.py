import pandas as pd
import mysql.connector

def carga_dataframes(host, user, password, database):

    df_balance = pd.DataFrame()
    df_demanda = pd.DataFrame()
    df_generacion = pd.DataFrame()
    df_intercambios = pd.DataFrame()

    use_bbdd_query = f"USE {database};"

    db = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

    cursor = db.cursor()
    cursor.execute(use_bbdd_query)

    # Diccionario con las tablas y los dataframes
    df_dict = {
        "balance": None,
        "demanda": None,
        "generacion": None,
        "intercambios": None
    }

    # Borrar datos de todas las tablas
    for tabla in df_dict.keys():

        select_from_bbdd = f"SELECT * FROM {tabla.upper()};"
        cursor.execute(select_from_bbdd)
        df_dict[tabla] = pd.DataFrame(cursor.fetchall())

        select_columns = f"SELECT column_name FROM information_schema.COLUMNS WHERE table_name = '{tabla.upper()}'"
        cursor.execute(select_columns)
        columns = [col[0] for col in cursor.fetchall()]

        df_dict[tabla].columns = columns

    cursor.close()
    db.close()

    df_balance = df_dict["balance"]
    df_demanda = df_dict["demanda"]
    df_generacion = df_dict["generacion"]
    df_intercambios = df_dict["intercambios"]

    return df_balance, df_demanda, df_generacion, df_intercambios