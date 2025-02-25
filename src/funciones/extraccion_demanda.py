#from funciones.parametros import *
from parametros import *
import pandas as pd
import requests
from datetime import datetime
from time import sleep

# DEFINICION DE FUNCION PARA EXTRACCION DE DEMANDA

def extrae_demanda(start_date, end_date) -> pd.DataFrame:

    hoy = datetime.now().strftime("%Y-%m-%dT%H:%M")
    df_demanda = pd.DataFrame()
    
    for widget in widgets["demanda"]:

        if (widget == "variacion-componentes") | (widget == "ire-general"):
            
            get_request_url = url + "demanda/" + widget + f"?start_date={start_date}&end_date={end_date}&time_trunc={time_trunc[2]}&systemElectric=nacional"
            #print(f"ENDPOINT: {get_request_url}")
            response = requests.get(get_request_url)
            status_response = response.status_code

            if status_response == 200:
                response = response.json()
                for i in response["included"]:   
                    df_base = pd.DataFrame(i["attributes"])
                    df_base.drop(columns="values", inplace=True)
                    df_base["extraccion"] = hoy
                    df_base["categoria"] = response["data"]["type"]
                    df_base["zona"] = "nacional"
                    df_valores = (pd.json_normalize((i["attributes"]["values"])))
                    df_concatenado = pd.concat([df_base,df_valores], axis=1)
                    df_demanda = pd.concat([df_demanda,df_concatenado], ignore_index=True)

        else:
            
            get_request_url = url + "demanda/" + widget + f"?start_date={start_date}&end_date={end_date}&time_trunc={time_trunc[1]}&systemElectric=nacional"
            #print(f"ENDPOINT: {get_request_url}")
            response = requests.get(get_request_url)
            status_response = response.status_code

            if status_response == 200:
                response = response.json()
                for i in response["included"]:   
                    df_base = pd.DataFrame(i["attributes"])
                    df_base.drop(columns="values", inplace=True)
                    df_base["extraccion"] = hoy
                    df_base["categoria"] = response["data"]["type"]
                    df_base["zona"] = "nacional"
                    df_valores = (pd.json_normalize((i["attributes"]["values"])))
                    df_concatenado = pd.concat([df_base,df_valores], axis=1)
                    df_demanda = pd.concat([df_demanda,df_concatenado], ignore_index=True)

            for zona in geo_limit:
                
                if zona != "ccaa":
                    
                    get_request_url = url + "demanda/" + widget + f"?start_date={start_date}&end_date={end_date}&time_trunc={time_trunc[1]}&geo_trunc={geo_trunc[0]}&geo_limit={zona}&geo_ids={geo_ids[zona]}"
                    #print(f"ENDPOINT: {get_request_url}")
                    response = requests.get(get_request_url)
                    status_response = response.status_code

                    if status_response == 200:
                        response = response.json()
                        for i in response["included"]:  
                            df_base = pd.DataFrame(i["attributes"])
                            df_base.drop(columns="values", inplace=True)
                            df_base["extraccion"] = hoy
                            df_base["categoria"] = response["data"]["type"]
                            df_base["zona"] = zona
                            df_valores = (pd.json_normalize((i["attributes"]["values"])))
                            df_concatenado = pd.concat([df_base,df_valores], axis=1)
                            df_demanda = pd.concat([df_demanda,df_concatenado], ignore_index=True)
                                
                else:
                    
                    for ccaa in geo_ids["ccaa"]:
                        
                        if (geo_ids['ccaa'][ccaa] != 8742) & (geo_ids['ccaa'][ccaa] != 8743) & (geo_ids['ccaa'][ccaa] != 8744) & (geo_ids['ccaa'][ccaa] != 8745):
                            
                            get_request_url = url + "demanda/" + widget + f"?start_date={start_date}&end_date={end_date}&time_trunc={time_trunc[1]}&geo_trunc={geo_trunc[0]}&geo_limit=ccaa&geo_ids={geo_ids['ccaa'][ccaa]}"
                            #print(f"ENDPOINT: {get_request_url}")
                            response = requests.get(get_request_url)
                            status_response = response.status_code

                            # RESPUESTA 500, ASI QUE NO ENTRA AQUI Y NO AÑADE NADA
                            if status_response == 200:
                                response = response.json()
                                for i in response["included"]:   
                                    df_base = pd.DataFrame(i["attributes"])
                                    df_base.drop(columns="values", inplace=True)
                                    df_base["extraccion"] = hoy
                                    df_base["categoria"] = response["data"]["type"]
                                    df_base["zona"] = ccaa
                                    df_valores = (pd.json_normalize((i["attributes"]["values"])))
                                    df_concatenado = pd.concat([df_base,df_valores], axis=1)
                                    df_demanda = pd.concat([df_demanda,df_concatenado], ignore_index=True)   

        sleep(1)

    return df_demanda
