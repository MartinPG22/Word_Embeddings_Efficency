import pandas as pd
import numpy as np

# Cargar los datos de entrenamiento y prueba
df_train = pd.read_csv('Datos/texto_train-True.csv')  # Asegúrate de que esté preprocesado y tenga columnas 'text' y 'label_text'
df_test = pd.read_csv('Datos/texto_test-True.csv')

# Cargar los vectores preentrenados de ConceptNet Numberbatch
print("Cargando ConceptNet Numberbatch...")
numberbatch_path = 'Conceptnet/numberbatch-en.txt'  # Ruta al archivo de ConceptNet Numberbatch
embedding_dim = 300

# Crear un diccionario de palabras a vectores
word_to_vec_map = {}
with open(numberbatch_path, encoding='utf-8') as f:
    next(f)  # Saltar la primera línea (información de la matriz)
    for line in f:
        values = line.split()
        word = values[0]  # Primera palabra
        vector = np.array(values[1:], dtype=np.float32)  # El resto son valores del vector
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

df_train.to_csv('Datos/texto_train_with_embeddings-ConceptNet.csv', index=False)
df_test.to_csv('Datos/texto_test_with_embeddings-ConceptNet.csv', index=False)

print("Embeddings guardados en 'texto_train_with_embeddings-ConceptNet.csv' y 'texto_test_with_embeddings-ConceptNet.csv'.")
