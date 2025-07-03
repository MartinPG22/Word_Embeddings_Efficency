# pip install tensorflow tensorflow-hub

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Cargar los datos de entrenamiento y prueba
df_train = pd.read_csv('Datos/texto_train-True.csv')  # Asegúrate de que esté preprocesado y tenga columnas 'text' y 'label_text'
df_test = pd.read_csv('Datos/texto_test-True.csv')

# Cargar el modelo del Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Función para generar embeddings de una oración
def generate_sentence_embedding(text):
    # Asegurarse de que el texto sea una cadena
    if pd.isna(text):  # Si es NaN, reemplazar con una cadena vacía
        text = ""
    
    text = str(text)  # Convertir el texto a cadena (en caso de que sea float u otro tipo)
    
    # Generar el embedding con USE
    sentence_embedding = embed([text])[0].numpy()  # Genera el embedding y convierte a NumPy
    
    return sentence_embedding

# Generar embeddings para el conjunto de entrenamiento
print("Generando embeddings para el conjunto de entrenamiento...")
df_train['embedding'] = df_train['text'].apply(generate_sentence_embedding)

# Generar embeddings para el conjunto de prueba
print("Generando embeddings para el conjunto de prueba...")
df_test['embedding'] = df_test['text'].apply(generate_sentence_embedding)

# Guardar los embeddings en archivos CSV
print("Guardando los embeddings en CSV...")
df_train['embedding'] = df_train['embedding'].apply(lambda x: x.tolist())  # Convertir los embeddings de lista a formato adecuado para CSV
df_test['embedding'] = df_test['embedding'].apply(lambda x: x.tolist())  # Convertir los embeddings de lista a formato adecuado para CSV

df_train.to_csv('Datos/texto_train_with_embeddings-USE.csv', index=False)
df_test.to_csv('Datos/texto_test_with_embeddings-USE.csv', index=False)

print("Embeddings guardados en 'texto_train_with_embeddings-USE.csv' y 'texto_test_with_embeddings-USE.csv'.")
