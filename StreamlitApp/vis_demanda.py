import streamlit as st

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from ML.gru_rnn import *
from functions.filtros_visualizaciones import *


def vis_demanda(df, periodo_seleccionado):

    df_filtered = p7_30_365_hist(df, periodo_seleccionado)
      
    if periodo_seleccionado != -1:

        if periodo_seleccionado == 365:
            p_selec = f'último año (media semanal)'
            df_filtered = p_365_mas(df_filtered, 'W')
            dtick = 5

        elif periodo_seleccionado == 7:
            p_selec = f'última semana'
            dtick = "1D"

        else:
            p_selec = 'último mes'
            dtick = None

    else:
        p_selec = 'histórico (media trimestral)'
        df_filtered = p_365_mas(df_filtered, 'Q')
        dtick = 4
           
    return df_filtered, periodo_seleccionado

if __name__ == "__main__":
    vis_demanda()