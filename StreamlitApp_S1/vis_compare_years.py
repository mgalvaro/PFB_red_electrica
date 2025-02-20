import streamlit as st

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import seaborn as sns
import plotly.express as px

from filtros_visualizaciones import *
# from data_cleaning import divide_fecha, limpia_columnas
# from parametros import variables


#_____________________


def vis_compare():

    # 3ª gráfica: Comparación de dos años

    df_demanda = pd.read_csv('../data/processed/DF_ DEMANDA_10_25_LIMPIO_V1.csv')
    df_generacion = pd.read_csv('../data/processed/DF_GENERACION_10_25_LIMPIO_V1.csv')
    df_intercambios = pd.read_csv('../data/processed/DF_INTERCAMBIOS_10_25_LIMPIO_V1.csv')
    df_balance = pd.read_csv('../data/processed/DF_BALANCE_10_25_LIMPIO_V1.csv')

    variables = {

        'Demanda': 1,
        'Balance': 2,
        'Generación': 3,
        'Intercambios': 4
        
    }

    variable = st.radio(label='Seleccionar la variable a comparar',
                        options=variables.keys(),
                        index=0,
                        disabled=False,
                        horizontal=True
                        )

    years = list(range(2010,2026))
    years_to_compare = st.multiselect(label='Comparación por años',
                                      options=years,
                                      default=None,
                                      placeholder='Seleccionar 2 años para comparar'
                           )
    
    
    
    if len(years_to_compare) != 2:
        st.warning('Por favor, elige 2 años para comparar')

    else:

        year1 = years_to_compare[0]
        year2 = years_to_compare[1]

        color = 'año'
        ldash = None

        if variables[variable] == 1:
            df = df_demanda[df_demanda['titulo'] == 'Demanda']
            df = df[(df['año'] == year1) | (df['año'] == year2)]
            df = df.groupby(['mes', 'año']).sum('valor_(MWh)').reset_index()

        elif variables[variable] == 2:
            df = df_balance
            df = df[df['zona'] == 'nacional']
            df = df[(df['año'] == year1) | (df['año'] == year2)]
            df = df.groupby(['mes', 'año']).sum('valor_(MWh)').reset_index()

        elif variables[variable] == 3:
            df = df_generacion
            df = df[df['zona'] == 'nacional']
            df = df[(df['año'] == year1) | (df['año'] == year2)]
            df = df.groupby(['mes', 'año']).sum('valor_(MWh)').reset_index()

        else:
            df = df_intercambios#[df_intercambios['zona'].isin(['Francia', 'Portugal', 'Marruecos', 'Andorra'])].groupby(by=['fecha'])
            df = filtro_intercambios(df, year1, year2)
            color = 'tipo'
            ldash='año'

        df1 = df[df['año'] == year1]
        df2 = df[df['año'] == year2]
        df = pd.concat([df1, df2], ignore_index=True)

        media = df['valor_(MWh)'].mean()
        mediana = df['valor_(MWh)'].median()
        maximo = df['valor_(MWh)'].max()
        minimo = df['valor_(MWh)'].min()

        fig1 = px.line(data_frame=df,
                       x='mes',
                       y='valor_(MWh)',
                       color=color,
                       line_dash=ldash,
                       title = f"{variable} en los años {year1} y {year2}",
                       labels={'mes': "Mes", 'valor_(MWh)': f"{variable} (MWh)", 'zona': "Zona"}
                       )
        st.plotly_chart(fig1)

        estadisticas = pd.DataFrame({
                                    'Estadística': ['Media', 'Mediana', 'Máximo', 'Mínimo'],
                                    'valor_': [media, mediana, maximo, minimo]
                                })

        st.table(estadisticas)

    

if __name__ == "__main__":
    vis_compare()