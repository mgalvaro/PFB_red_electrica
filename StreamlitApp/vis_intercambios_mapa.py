import streamlit as st

import pandas as pd

import folium
from streamlit_folium import st_folium

import json

from functions.filtros_visualizaciones import *

def vis_intercambios(df):
    coord_paises = {
                    "frontera": ["Intercambios", "Portugal", "Francia", "Andorra", "Marruecos"],
                    "latitud": [40.4168, 38.7169, 44.5, 42.5078, 33.2514],
                    "longitud": [-3.7038, -9.1399, -0, 1.5211, -5.372]
                }

    df_coord = pd.DataFrame(coord_paises)    
    st.markdown("### :world_map: Intercambios de Energía con otros Países")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### **¿Qué son los intercambios?** ")
        st.markdown("Los intercambios eléctricos ocurren cuando un país o región comparte energía con otros a través de conexiones especiales. Esto ayuda a equilibrar el sistema, aprovechando la electricidad disponible y apoyándose mutuamente en momentos de alta demanda o baja generación local.")
        start_date = st.date_input("Fecha de inicio")
        end_date = st.date_input("Fecha de fin")

        try:
            df = df[(df['fecha'] >= start_date) & (df['fecha'] <= end_date)]

            df_intercambio_total = df.groupby(['frontera','tipo'])['valor_(MWh)'].sum().reset_index()
            df_intercambio_total = df_intercambio_total[df_intercambio_total['frontera'] != 'Enlace Península-Baleares']

            exp_imp_total = df_intercambio_total.groupby('tipo').sum('valor_(MWh)')['valor_(MWh)']

            intercambios_nac = {
                'frontera': ['Intercambios', 'Intercambios'],
                'tipo': ['export', 'import'],
                'valor_(MWh)': [exp_imp_total.values[0], exp_imp_total.values[1]]
            }
        
            intercambios_nac = pd.DataFrame(intercambios_nac)
            df_intercambio_total = pd.concat([df_intercambio_total, intercambios_nac])
            df_intercambio_total = df_intercambio_total.merge(df_coord, on='frontera', how='left')
            df_saldo = df_intercambio_total.groupby('frontera').sum().reset_index()
            df_saldo['tipo'] = df_saldo['tipo'].apply(lambda x: 'saldo')
            df_saldo = pd.concat([df_intercambio_total, df_saldo]).reset_index(drop=True)

        except IndexError:
            st.warning(":warning: No hay datos en este intervalo de fechas")

        with col2:
            try:
                geo_json_path = "world_countries.json"
                with open(geo_json_path, "r", encoding="utf-8") as file:
                    geo_json_data = json.load(file)

                spain_map = folium.Map(location=[40.4637, -3.7492], tiles="openstreetmap", zoom_start=5, min_zoom=5, max_zoom=7, max_bounds=True)
                for pais, lat, lon in zip(df_coord['frontera'], df_coord['latitud'], df_coord['longitud']):

                    exp = df_saldo[(df_saldo['tipo'] == 'export') & (df_saldo['frontera'] == pais)]['valor_(MWh)'].values[0]
                    imp = df_saldo[(df_saldo['tipo'] == 'import') & (df_saldo['frontera'] == pais)]['valor_(MWh)'].values[0]
                    saldo = df_saldo[(df_saldo['tipo'] == 'saldo') & (df_saldo['frontera'] == pais)]['valor_(MWh)'].values[0]

                    popup_html = f"""
                        <div style="width: 250px; font-size: 14px;">
                        <b>📍{pais}</b><br>
                        📤 Exportado: {round(exp, 1)} GWh<br>
                        📥 Importado: {round(imp, 1)} GWh<br>
                        ⚖️ Saldo: {round(saldo, 1)} GWh
                        </div>
                        """
                    popup = folium.Popup(popup_html, max_width=300)
                    if saldo < 0:
                        color = 'green'
                    else:
                        color = 'blue'

                    folium.Marker(
                        location=[lat, lon],
                        popup=popup,
                        icon=folium.Icon(color=color, icon="bolt", prefix="fa")
                        ).add_to(spain_map)
                    
                folium.Choropleth(
                    geo_data=geo_json_data,
                    data=df_saldo[df_saldo['tipo'] == 'saldo'],
                    columns=["frontera", "valor_(MWh)"],
                    nan_fill_color="gray",
                    nan_fill_opacity=0.4,
                    key_on="feature.properties.ADMIN",
                    fill_color="GnBu",  # Escala de color de folium
                    fill_opacity=0.7,
                    line_opacity=0.2,
                ).add_to(spain_map)

                st_folium(spain_map, width=800, height=600)
                with st.expander(label = "Dataset de Intercambios Filtrado", expanded = False):
                    st.dataframe(df_intercambio_total) 

            except UnboundLocalError:
                spain_map_vacio = folium.Map(location=[40.4637, -3.7492], tiles="openstreetmap", zoom_start=5, min_zoom=5, max_zoom=7, max_bounds=True)
                st_folium(spain_map_vacio, width=800, height=600)

if __name__ == "__main__":
    vis_intercambios()