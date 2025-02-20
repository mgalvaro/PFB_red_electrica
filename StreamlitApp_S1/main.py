import streamlit as st
from config import PAGE_CONFIG
from vis_demanda import vis_demanda
from vis_intercambios_mapa import vis_intercambios
from vis_compare_years import vis_compare

st.set_page_config(**PAGE_CONFIG)

def main():

    menu = ['Inicio', 'Serie Temporal Demanda', 'Mapa de Intercambios', 'Comparación Anual']

    choice = st.sidebar.selectbox(label='Menu', options=menu, index=0)

    if choice == 'Inicio':
        st.markdown('### :zap: Bienvenido a la App de datos de la REE :zap:')

    elif choice == 'Serie Temporal Demanda':
        vis_demanda()

    elif choice == 'Mapa de Intercambios':
        vis_intercambios()

    elif choice == 'Comparación Anual':
        vis_compare()
    
    else:
        pass



if __name__ == '__main__':
    main()