# DEFINICION DE PARAMETROS BASE PARA EXTRACCION

url = "https://apidatos.ree.es/es/datos/"

languages = ["es", "en"]

categories = ["balance", "demanda", "generacion", "intercambios"]

widgets = {

    "balance" : ["balance-electrico"], 

    "demanda" : ["evolucion",
                "perdidas-transporte",
                "ire-general"], 

    "generacion" : ["estructura-generacion"],
    
    "intercambios" : ["francia-frontera",  
                      "portugal-frontera",
                      "marruecos-frontera",
                      "andorra-frontera",
                      "enlace-baleares"]
}

time_trunc = ["hour", "day_2", "month", "year"]

geo_trunc = ["electric_system"]

geo_limit = ["peninsular", "canarias", "baleares", "ceuta", "melilla", "ccaa"]

geo_ids = {

    "peninsular" : 8741, 
    "canarias": 8742, 
    "baleares" : 8743, 
    "ceuta" : 8744, 
    "melilla" : 8745, 
    "ccaa" : {
        "Andalucía" : 4,
        "Aragón" : 5,
        "Cantabria" : 6,
        "Castilla la Mancha" : 7,
        "Castilla y León" : 8,
        "Cataluña" : 9,
        "País Vasco" : 10,
        "Principado de Asturias" : 11,
        "Comunidad de Ceuta" : 8744,
        "Comunidad de Melilla" : 8745,
        "Comunidad de Madrid" : 13,
        "Comunidad de Navarra" : 14,
        "Comunidad Valenciana" : 15,
        "Extremadura" : 16,
        "Galicia" : 17,
        "Islas Baleares" : 8743,
        "Islas Canarias" : 8742,
        "La Rioja" : 20,
        "Región de Murcia" : 21
        }
}