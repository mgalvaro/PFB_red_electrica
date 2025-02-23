import streamlit as st

import plotly.express as px

from functions.filtros_visualizaciones import *


def vis_compare(df_demanda, df_generacion, df_intercambios):

    # 3ª gráfica: Comparación de dos años

    variables = {

        'Demanda': 1,
        'Balance': 2,
        'Generación': 3,
        'Intercambio': 4
        
    }

    st.markdown("### :calendar: Comparador por años")

    variable = st.radio(label='Seleccionar la variable a comparar',
                        options=variables.keys(),
                        index=0,
                        disabled=False,
                        horizontal=True
                        )
    
    years = list(df_demanda['año'].unique())
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

        ldash = None

        if variables[variable] == 1:
            df = filtro_comparador((df_demanda[df_demanda['zona'] == 'nacional']), year1, year2)

        elif variables[variable] == 2:
            df1 = df_demanda
            df2 = df_generacion
            df = calculo_balance(df1, df2, year1, year2)  # saca una tupla de dos df

        elif variables[variable] == 3:
            df = filtro_comparador(df_generacion[df_generacion['zona'] == 'nacional'], year1, year2)

        else:
            df = filtro_intercambios(df_intercambios, year1, year2)
            ldash='tipo'


        fig = px.line(
            data_frame=df[0],
            x='fecha_sin_year',
            y='valor_(GWh)',
            color='año',
            title=f"<b>{variable.title()}</b> nacional de energía a lo largo del año para {year1} y {year2}",
            labels={'fecha_sin_year': 'Fecha', 'valor_(GWh)': 'Generación (GWh)', 'año': 'Año'},
            line_dash=ldash,
            color_discrete_sequence=["#32CD32", "#8B00FF"]
            )

        fig.update_xaxes(tickformat="%m-%b")

        st.plotly_chart(fig)

        st.table(df[1])

        if variables[variable] != 4:
            df_stats = df[1].melt(id_vars=['año'], var_name='Métrica', value_name='Valor')
            fc = None
        else:
            df_stats = df[1].melt(id_vars=['año', 'tipo'], var_name='Métrica', value_name='Valor')
            fc='tipo'

        fig_stats = px.bar(df_stats,
                           
                            x='año', 
                            y='Valor', 
                            color='Métrica',
                            facet_col=fc, 
                            barmode='group', 
                            title=f"Estadísticas de {variable} de energía por Año",
                            labels={'Valor': 'GWh', 'año': 'Año', 'tipo': 'Tipo'},
                            color_discrete_sequence=['#FF5733', '#33FF57', '#3357FF', '#FF33A1']

                            )
        
        st.plotly_chart(fig_stats)

    

if __name__ == "__main__":
    vis_compare()