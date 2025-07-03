import pandas as pd
import numpy as np
import fasttext
import fasttext.util

# Cargar los datos de entrenamiento y prueba
df_train = pd.read_csv('Datos/texto_train.csv')  # Asegúrate de que esté preprocesado y tenga las columnas 'text' y 'label_text'
df_test = pd.read_csv('Datos/texto_test.csv')
ft = fasttext.load_model('cc.en.300.bin')
ft.get_dimension()

# Para reducir la dimensión del modelo a 100 si queremos
# fasttext.util.reduce_model(ft, 100)
# ft.get_dimension()

# Función para generar el embedding promedio de una oración
def get_sentence_embedding(text, model):
    words = text.split()  # Asumiendo que el texto está en formato de palabras separadas por espacios
    word_embeddings = [model.get_word_vector(word) for word in words if word in model]  # Obtener embeddings para cada palabra
    
    # Promediar los embeddings de las palabras
    if len(word_embeddings) > 0:
        sentence_embedding = np.mean(word_embeddings, axis=0)
    else:
        sentence_embedding = np.zeros(model.get_dimension())  # Vector cero si no hay palabras en el vocabulario
    return sentence_embedding

# Generar embeddings para el conjunto de entrenamiento
print("Generando embeddings para el conjunto de entrenamiento...")
df_train['embedding'] = df_train['text'].apply(lambda x: get_sentence_embedding(x, model))

# Generar embeddings para el conjunto de prueba
print("Generando embeddings para el conjunto de prueba...")
df_test['embedding'] = df_test['text'].apply(lambda x: get_sentence_embedding(x, model))

# Guardar los embeddings en archivos CSV
print("Guardando los embeddings en CSV...")
df_train['embedding'] = df_train['embedding'].apply(lambda x: x.tolist())  # Convertir los embeddings de lista a formato adecuado para CSV
df_test['embedding'] = df_test['embedding'].apply(lambda x: x.tolist())  # Convertir los embeddings de lista a form
