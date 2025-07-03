import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertModel
import numpy as np
import joblib

# Cargar los datos de entrenamiento y prueba
df_train = pd.read_csv('Datos/texto_train-True.csv')  
df_test = pd.read_csv('Datos/texto_test-True.csv')

# Inicializar el tokenizador y el modelo BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Función para generar embeddings de una oración
def generate_sentence_embedding(text):
    # Asegurarse de que el texto sea una cadena
    if pd.isna(text):  # Si es NaN, reemplazar con una cadena vacía
        text = ""
    
    text = str(text)  # Convertir el texto a cadena (en caso de que sea float u otro tipo)
    marked_text = "[CLS] " + text + " [SEP]"
    # Tokenización y conversión a tensores
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])
    
    # Obtener embeddings de BERT
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensor)
    
    # Usar el embedding del token [CLS]
    sentence_embedding = outputs.last_hidden_state[0][0].numpy()  # Convierte a NumPy
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

df_train.to_csv('Datos/texto_train_with_embeddings-Bert.csv', index=False)
df_test.to_csv('Datos/texto_test_with_embeddings-Bert.csv', index=False)

print("Embeddings guardados")

