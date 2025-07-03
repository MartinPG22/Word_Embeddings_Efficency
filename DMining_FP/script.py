import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Cargar los datos
data = pd.read_csv("Datos/text_copy.csv")

# Inicializar el tokenizador y el modelo preentrenado
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Función para tokenizar, mapear a índices y extraer embeddings
def tokenize_and_generate_embeddings(sentence):
    marked_text = "[CLS] " + sentence + " [SEP]"
    
    # Tokenización y mapeo a índices
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    
    # Conversión a tensores
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])
    
    # Obtener embeddings del modelo BERT
    with torch.no_grad():  # No calcular gradientes
        outputs = model(tokens_tensor, token_type_ids=segments_tensor)
    
    # Obtener los "últimos estados ocultos" de la salida
    hidden_states = outputs.last_hidden_state  # [batch_size, n_tokens, hidden_size]
    
    # Eliminar la dimensión extra (batch_size)
    hidden_states = torch.squeeze(hidden_states, dim=0)  # [n_tokens, hidden_size]
    
    # Embedding para toda la oración usando el token [CLS]
    sentence_embedding = hidden_states[0]  # [768]
    
    # Devolver tokens, índices y embeddings
    token_index_pairs = list(zip(tokenized_text, indexed_tokens))
    return tokenized_text, indexed_tokens, token_index_pairs, hidden_states, sentence_embedding

# Aplicar el tokenizador y generación de embeddings a cada oración
data['tokenized_data'] = data['text'].apply(tokenize_and_generate_embeddings)

# Mostrar resultados
for idx, row in data.iterrows():
    tokenized_text, indexed_tokens, token_index_pairs, hidden_states, sentence_embedding = row['tokenized_data']
    print(f"Original: {row['text']}")
    print(f"Tokenizado: {tokenized_text}")
    
    print("\nTokens con índices:")
    for token, index in token_index_pairs:
        print('{:<12} {:>6,}'.format(token, index))
    
    print(f"\nTamaño de los embeddings por token: {hidden_states.shape}")  # [n_tokens, 768]
    print(f"Embedding de la oración ([CLS]): {sentence_embedding.shape}")  # [768]
    print("\n" + "="*50 + "\n")
