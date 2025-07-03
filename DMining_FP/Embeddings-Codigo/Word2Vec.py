import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

# Cargar los datos de entrenamiento y prueba
df_train = pd.read_csv('Datos/texto_train-True.csv')  # Asegúrate de que esté preprocesado y tenga columnas 'text' y 'label_text'
df_test = pd.read_csv('Datos/texto_test-True.csv')

# Cargar el modelo preentrenado de Word2Vec
print("Cargando Word2Vec...")
word2vec_path = 'Word2vec/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin'  # Ruta al archivo binario de Word2Vec
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
embedding_dim = word2vec_model.vector_size

# Función para generar embeddings promediando los vectores de las palabras
def generate_sentence_embedding(text):
    if pd.isna(text):  # Si es NaN, reemplazar con una cadena vacía
        text = ""
    
    words = str(text).lower().split()  # Tokenizar el texto en palabras
    embeddings = [word2vec_model[word] for word in words if word in word2vec_model]  # Obtener los vectores
    
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

df_train.to_csv('Datos/texto_train_with_embeddings-Word2Vec.csv', index=False)
df_test.to_csv('Datos/texto_test_with_embeddings-Word2Vec.csv', index=False)

print("Embeddings guardados en 'texto_train_with_embeddings-Word2Vec.csv' y 'texto_test_with_embeddings-Word2Vec.csv'.")
