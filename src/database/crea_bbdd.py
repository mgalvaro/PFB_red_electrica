import pandas as pd
import mysql.connector
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.types import Float, BigInteger


host = "localhost"
user = "root"
password = "root"
database = "red_electrica"

def crea_bbdd() -> str:

    drop_db_query = f"DROP DATABASE IF EXISTS {database};"

    create_db_query = f"CREATE DATABASE {database};"

    usr_db_query = f"USE {database};"

    db_query = """-- Tabla DEMANDA
        CREATE TABLE DEMANDA (
            ID CHAR(42) NOT NULL PRIMARY KEY,
            titulo VARCHAR(255),
            composite INT,
            fecha_extraccion DATETIME,
            categoria VARCHAR(255),
            zona VARCHAR(255),
            valor_(MWh) FLOAT,
            porcentaje FLOAT,
            fecha DATE NOT NULL,
            año INT,
            mes INT,
            dia INT,
            dia_semana VARCHAR(50)
        );

        -- Tabla GENERACION
        CREATE TABLE GENERACION (
            ID CHAR(42) NOT NULL PRIMARY KEY,
            titulo VARCHAR(255),
            tipo VARCHAR(255),
            composite INT,
            fecha_extraccion DATETIME,
            categoria VARCHAR(255),
            zona VARCHAR(255),
            valor_(MWh) FLOAT,
            porcentaje FLOAT,
            fecha DATE NOT NULL,
            año INT,
            mes INT,
            dia INT,
            dia_semana VARCHAR(50)
        );

        -- Tabla INTERCAMBIOS
        CREATE TABLE INTERCAMBIOS (
            ID CHAR(42) NOT NULL PRIMARY KEY,
            titulo VARCHAR(255),
            tipo VARCHAR(255),
            composite INT,
            fecha_extraccion DATETIME,
            categoria VARCHAR(255),
            frontera VARCHAR(255),
            zona VARCHAR(255),
            valor_(MWh) FLOAT,
            porcentaje FLOAT,
            fecha DATE NOT NULL,
            año INT,
            mes INT,
            dia INT,
            dia_semana VARCHAR(50)
        );

        -- Tabla BALANCE
        CREATE TABLE BALANCE (
            ID CHAR(42) NOT NULL PRIMARY KEY,
            titulo VARCHAR(255),
            tipo VARCHAR(255),
            composite INT,
            fecha_extraccion DATETIME,
            categoria VARCHAR(255),
            zona VARCHAR(255),
            valor_(MWh) FLOAT,
            porcentaje FLOAT,
            fecha DATE NOT NULL,
            año INT,
            mes INT,
            dia INT,
            dia_semana VARCHAR(50)
        );"""

    db = mysql.connector.connect(host     = host,
                                user     = user,
                                password = password,
                                database = database)

    cursor = db.cursor()
    
    cursor.execute(drop_db_query)
    cursor.fetchall()

    cursor.execute(create_db_query)
    cursor.fetchall()

    cursor.execute(usr_db_query)
    cursor.fetchall()

    cursor.execute(db_query, multi=True)
    cursor.fetchall()
    
    cursor.close()
    db.close()

    return f"{database} creada con éxito!"


def crea_tablas() -> str:

    conexion_str = f"mysql+pymysql://{user}:{password}@{host}/{database}"
    engine = sqlalchemy.create_engine(conexion_str)

    file_paths = {
    "BALANCE": "../../data/processed/DF_BALANCE_10_25_LIMPIO.csv",
    "DEMANDA": "../../data/processed/DF_DEMANDA_10_25_LIMPIO.csv", 
    "GENERACION": "../../data/processed/DF_GENERACION_10_25_LIMPIO.csv", 
    "INTERCAMBIOS": "../../data/processed/DF_INTERCAMBIOS_10_25_LIMPIO.csv"
    }

    for tabla, file_path in file_paths.items():
        print(f"Insertando datos en la tabla {tabla} desde {file_path}...")
        
        dataframe = pd.read_csv(file_path)

        dataframe.columns = [col.replace(" ", "_") for col in dataframe.columns]
        
        dataframe.to_sql(name=tabla, con=engine, if_exists='append', index=False)

        print(f"✔ {len(dataframe)} Datos insertados en {tabla} correctamente.")

    return f"Tablas creadas con éxito!"