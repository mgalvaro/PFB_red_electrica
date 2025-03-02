import pandas as pd

# Cargar el dataset
ruta_archivo = "C:/Users/nacho/Desktop/Aprendiendo a programar/PFB/TFB/data/data_scaled/DF_DEMANDA_10_25_PROCESADO.csv"
df = pd.read_csv(ruta_archivo)

# Ordenar por año, mes y día respetando el orden temporal
df = df.sort_values(by=["fecha"])

# Dividir en train (80%) y test (20%)
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

# Guardar los conjuntos en archivos CSV en el directorio especificado
train_path = "C:/Users/nacho/Desktop/Aprendiendo a programar/PFB/TFB/data/train_test/train_temporal.csv"
test_path = "C:/Users/nacho/Desktop/Aprendiendo a programar/PFB/TFB/data/train_test/test_temporal.csv"

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)

print("Archivos guardados en:")
print(train_path)
print(test_path)