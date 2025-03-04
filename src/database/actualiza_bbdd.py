import pandas as pd
import sqlalchemy
import mysql.connector
from sqlalchemy import create_engine, text
# from passwords import pw


# Configuración de conexión
host = "localhost"
user = "root"
pw = 'Merlinku16/'
database = "red_electrica"

def actualiza_bbdd(df_balance_historico, df_demanda_historico, df_generacion_historico, df_intercambios_historico) -> None:
    
    # Conexión a la base de datos con mysql.connector para garantizar que conecta con la bbdd
    db = mysql.connector.connect(
        host=host,
        user=user,
        password=pw,
        database=database
    )
    cursor = db.cursor()

    # Seleccionar la base de datos
    cursor.execute(f"USE {database};")

    # Diccionario con las tablas y los dataframes
    df_dict = {
        "balance": df_balance_historico,
        "demanda": df_demanda_historico,
        "generacion": df_generacion_historico,
        "intercambios": df_intercambios_historico
    }

    # Borrar datos de todas las tablas
    for tabla in df_dict.keys():
        cursor.execute(f"DELETE FROM {tabla.upper()};")
        db.commit()  

    cursor.close()
    db.close()

    # Conexión con SQLAlchemy para usar to_sql y simplificar el proceso
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@{host}/{database}")

    # Insertar los datos con to_sql
    for tabla, dataframe in df_dict.items():
        # Convertir fechas a string para evitar problemas con tipos de datos
        if "fecha_extraccion" in dataframe.columns:
            dataframe["fecha_extraccion"] = dataframe["fecha_extraccion"].astype(str)
        if "fecha" in dataframe.columns:
            dataframe["fecha"] = dataframe["fecha"].astype(str)
        
        # Insertar datos
        dataframe.to_sql(
            name=tabla.upper(),
            con=engine,
            if_exists="append",  
            index=False,         
            chunksize=1000       
        )

    engine.dispose() 

    print("Datos actualizados con éxito.")

    return None