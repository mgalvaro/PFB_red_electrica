import streamlit as st

import pandas as pd

import folium
from streamlit_folium import st_folium

import json

from filtros_visualizaciones import *

def vis_intercambios():

    coord_paises = {
                    "frontera": ["Intercambios", "Portugal", "Francia", "Andorra", "Marruecos"],
                    "latitud": [40.4168, 38.7169, 44.5, 42.5078, 33.2514],
                    "longitud": [-3.7038, -9.1399, -0, 1.5211, -5.372]
                }

    df_coord = pd.DataFrame(coord_paises)

    periodo_dict = {
                    "Últimos 7 días": 7,
                    "Últimos 30 días": 30,
                    "Últimos 365 días": 365,
                    "Histórico": -1 
                }
    
    st.markdown("#### Intercambios de Energía con otros Países")
    with st.expander(label = "Dataset: Intercambios", expanded = False):
        df = pd.read_csv('../data/processed/SPRINT1PRUEBAS/Data/DF_INTERCAMBIOS_10_25_LIMPIO_V1.csv')
        df = df[df['frontera'].isin(coord_paises['frontera'])]
        st.dataframe(df)

    periodo = st.radio(label = "Selecciona el período",
                       options = list(periodo_dict.keys()),
                       index = 0,
                       disabled = False,
                       horizontal = True)
    
    periodo_seleccionado = periodo_dict[periodo]

    df_filtered = p7_30_365_hist(df, periodo_seleccionado)[0]
    periodo_seleccionado = p7_30_365_hist(df, periodo_seleccionado)[1]

    df_intercambio_total = df_filtered.groupby(['frontera','tipo'])['valor'].sum().reset_index()

    exportacion_total = df_intercambio_total.groupby('tipo').sum()['valor'].values[0]
    importacion_total = df_intercambio_total.groupby('tipo').sum()['valor'].values[1]

    df_intercambio_total.loc[len(df_intercambio_total)] = ['Intercambios', 'export', exportacion_total]
    df_intercambio_total.loc[len(df_intercambio_total)] = ['Intercambios', 'import', importacion_total]
    with st.expander(label = "Dataset de Intercambios Filtrado", expanded = False):
        st.dataframe(df_intercambio_total)

    df_intercambio_total = df_intercambio_total.merge(df_coord, on='frontera', how='left')

    # with st.expander(label = "Dataset de Intercambios Filtrado", expanded = False):
    #     st.dataframe(df_intercambio_total)

    geo_json_path = "world_countries.json"
    with open(geo_json_path, "r", encoding="utf-8") as file:
        geo_json_data = json.load(file)

    spain_map = folium.Map(location=[40.4637, -3.7492], tiles="openstreetmap", zoom_start=5, min_zoom=5, max_zoom=7, max_bounds=True)


    for pais, lat, lon in zip(df_coord["frontera"], df_coord["latitud"], df_coord["longitud"]):
        
        row = df_intercambio_total[df_intercambio_total['frontera'] == pais].reset_index(drop=True)
        
        if not row.empty:  # Verificamos si el país tiene datos
            exp = row[row['tipo'] == 'export']['valor'].values[0]
            imp = row[row['tipo'] == 'import']['valor'].values[0]
            saldo = exp + imp

            popup_html = f"""
            <div style="width: 250px; font-size: 14px;">
            <b>{pais}</b><br>
            📤 Exportado: {exp:,} MWh<br>
            📥 Importado: {imp:,} MWh<br>
            ⚖️ Saldo: {saldo:,} MWh
            </div>
            """
            popup = folium.Popup(popup_html, max_width=300)
            if saldo < 0:
                color = 'green'
            else:
                color = 'red'

            folium.Marker(
                location=[lat, lon],
                popup=popup,
                icon=folium.Icon(color=color, icon="bolt", prefix="fa")
            ).add_to(spain_map)

    folium.GeoJson(
        geo_json_data,
        name="Intercambios Energéticos",
        style_function=lambda feature: {
            "fillColor": (
                "green" if (
                    feature["properties"]["ADMIN"] in df_intercambio_total["frontera"].values and saldo < 0,
                    # df_intercambio_total.loc[df_intercambio_total["frontera"] == feature["properties"]["ADMIN"], "valor"].values[0] < 0
                ) else 
                "red" if (
                    feature["properties"]["ADMIN"] in df_intercambio_total["frontera"].values and saldo > 0
                    # df_intercambio_total.loc[df_intercambio_total["frontera"] == feature["properties"]["ADMIN"], "valor"].values[0] > 0
                ) else 
                "gray"
            ),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["ADMIN"], aliases=["País:"], sticky=False
        )
    ).add_to(spain_map)





    # folium.GeoJson(
    #     geo_json_data,
    #     name="Intercambios Energéticos",
    #     style_function=lambda feature: {
    #         "fillColor": colormap(df_intercambio_total.loc[
    #             df_intercambio_total["frontera"] == feature["properties"]["ADMIN"], "valor"
    #         ].values[0]) if feature["properties"]["ADMIN"] in df_intercambio_total["frontera"].values else "gray",
    #         "color": "black",
    #         "weight": 1,
    #         "fillOpacity": 0.7
    #     },
    #     tooltip=folium.GeoJsonTooltip(
    #         fields=["ADMIN"], aliases=["País:"], sticky=False
    #     )
    # ).add_to(spain_map)

    # colormap.add_to(spain_map)
    # colormap.caption = "Saldo Energético (MWh)"

    st_folium(spain_map, width=800, height=600)

if __name__ == "__main__":
    vis_intercambios()