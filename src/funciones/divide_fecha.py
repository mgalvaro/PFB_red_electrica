import pandas as pd 

def divide_fecha(dataframe) -> pd.DataFrame:
 
    dataframe["año"] = dataframe["datetime"].str[:4].astype("int32")
    dataframe["mes"] = dataframe["datetime"].str[5:7].astype("int32")
    dataframe["dia"] = dataframe["datetime"].str[8:10].astype("int32")
    
    return dataframe