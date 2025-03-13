import streamlit as st

import plotly.express as px

from functions.filtros_visualizaciones import *


def vis_compare(df_demanda, df_generacion, df_intercambios):

    # 3ª gráfica: Comparación de dos años
    col1, col2 = st.columns(2)

    with col1:
        variables = {

            'Demanda': 1,
            'Balance': 2,
            'Generación': 3,
            'Intercambio': 4
            
        }

        st.markdown("### :calendar: Comparador por años")
        st.markdown("Aquí podrás ver las diferencias que hay entre cualquiera de las 4 variables energéticas durante los 2 años seleccionados.")
        st.markdown("**Gráfico 1:** Representa la media semanal de la variable seleccionada a lo largo de los años seleccionados.")
        st.markdown("**Gráfico 2:** Representa información estadística más detallada a través de un boxplot.")

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
            st.warning('Por favor, elige 2 años para comparar.')

        else:
            year1 = years_to_compare[0]
            year2 = years_to_compare[1]

            # parámetros necesarios para graficar ya que el df de intercambios no se procesa igual por la columna 'tipo'
            ldash = None
            x_var = 'año'
            filtro_groupby = ['fecha_sin_year', 'año']

            if variables[variable] == 1:
                df, df_stats = filtro_comparador((df_demanda[df_demanda['zona'] == 'nacional']), year1, year2)

            elif variables[variable] == 2:
                df1 = df_demanda
                df2 = df_generacion
                df, df_stats = calculo_balance(df1, df2, year1, year2)  # saca una tupla de dos df

            elif variables[variable] == 3:
                df, df_stats = filtro_comparador(df_generacion[df_generacion['zona'] == 'nacional'], year1, year2)

            else:
                df, df_stats = filtro_intercambios(df_intercambios, year1, year2)
                ldash='tipo'
                x_var = 'tipo'
                filtro_groupby = ['fecha_sin_year', 'año', 'tipo']


            df['fecha_sin_year'] = pd.to_datetime(df['fecha_sin_year'])
            df['fecha_sin_year'] = df['fecha_sin_year'].dt.to_period('W')

            df_0 = df.groupby(filtro_groupby).agg({'valor_(GWh)': 'mean'}).reset_index(drop=False)
            df_0['fecha_sin_year'] = df_0['fecha_sin_year'].astype(str)
            df_0['fecha_sin_year'] = df_0['fecha_sin_year'].apply(lambda x : x[:10])

        with col2:
            try:
                fig = px.line(
                    data_frame=df_0,
                    x='fecha_sin_year',
                    y='valor_(GWh)',
                    color='año',
                    title=f"<b>{variable.title()}</b> nacional de energía a lo largo del año para {year1} y {year2} (media semanal)",
                    labels={'fecha_sin_year': 'Fecha', 'valor_(GWh)': 'Generación (GWh)', 'año': 'Año'},
                    line_dash=ldash,
                    color_discrete_sequence=["#32CD32", "#8B00FF"]
                    )

                fig.update_xaxes(tickformat="%d-%b")

                st.plotly_chart(fig)

                st.table(df_stats)

                fig_stats = px.box(data_frame=df,
                                x=x_var,
                                y='valor_(GWh)',
                                color='año',
                                    # facet_col=fc, 
                                    # barmode='group', 
                                    title=f"Boxplot de {variable} de energía por Año",
                                    labels={'valor_(GWh)': 'GWh', 'año': 'Año'},
                                    color_discrete_sequence=["#32CD32", "#8B00FF"]
                                    )
                
                st.plotly_chart(fig_stats)
                with st.expander(label = "Dataset de Intercambios Filtrado", expanded = False):
                    st.dataframe(df_0) 

            except UnboundLocalError:
                st.info(":warning: No hay datos para mostrar")

if __name__ == "__main__":
    vis_compare()