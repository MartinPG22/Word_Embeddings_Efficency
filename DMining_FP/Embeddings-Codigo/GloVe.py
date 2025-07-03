import pandas as pd
import numpy as np

# Cargar los datos de entrenamiento y prueba
df_train = pd.read_csv('Datos/texto_train-True.csv')  # Asegúrate de que esté preprocesado y tenga columnas 'text' y 'label_text'
df_test = pd.read_csv('Datos/texto_test-True.csv')

# Cargar los vectores de GloVe
print("Cargando GloVe...")
glove_path = 'GloVe/glove.6B/glove.6B.300d.txt'  # Ruta al archivo GloVe
embedding_dim = 300

# Crear un diccionario de palabras a vectores
word_to_vec_map = {}
with open(glove_path, encoding='utf-8') as f:
    for line in f:
        line_split = line.split()
        word = line_split[0]
        vector = np.array(line_split[1:], dtype=np.float32)
        word_to_vec_map[word] = vector

# Función para generar embeddings promediando los vectores de las palabras
def generate_sentence_embedding(text):
    if pd.isna(text):  # Si es NaN, reemplazar con una cadena vacía
        text = ""
    
    words = str(text).lower().split()  # Tokenizar el texto en palabras
    embeddings = [word_to_vec_map[word] for word in words if word in word_to_vec_map]  # Obtener los vectores
    
    # Si no hay palabras reconocidas, usar un vector de ceros
    if len(embeddings) == 0:
        return np.zeros(embedding_dim)
    
    return np.mean(embeddings, axis=0)  # Promediar los vectores

# Generar embeddings para el conjunto de entrenamiento
print("Generando embeddings para el conjunto de entrenamiento...")
df_train['embedding'] = df_train['text'].apply(generate_sentence_embedding)

# Generar embeddings para el conjunto de prueba
print("Generando embeddings para el conjunto de prueba...")
df_test['embedding'] = df_test['text'].apply(generate_sentence_embedding)

# Guardar los embeddings en archivos CSV
print("Guardando los embeddings en CSV...")
df_train['embedding'] = df_train['embedding'].apply(lambda x: x.tolist())
df_test['embedding'] = df_test['embedding'].apply(lambda x: x.tolist())

df_train.to_csv('Datos/texto_train_with_embeddings-GloVe-True.csv', index=False)
df_test.to_csv('Datos/texto_test_with_embeddings-GloVe-True.csv', index=False)

print("Embeddings guardados en 'texto_train_with_embeddings-GloVe.csv' y 'texto_test_with_embeddings-GloVe.csv'.")
