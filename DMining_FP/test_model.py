import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Definir la misma clase del modelo
class TextClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)


# Cargar los vectores de ConceptNet Numberbatch
def load_conceptnet_vectors(path, embedding_dim=300):
    word_to_vec_map = {}
    print("Cargando ConceptNet Numberbatch...")
    with open(path, encoding='utf-8') as f:
        next(f)  # Saltar la primera línea
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            word_to_vec_map[word] = vector
    return word_to_vec_map

# Función para generar embeddings promediando los vectores de las palabras
def generate_sentence_embedding(text, word_to_vec_map, embedding_dim=300):
    if pd.isna(text):
        text = ""
    words = str(text).lower().split()
    embeddings = [word_to_vec_map[word] for word in words if word in word_to_vec_map]
    
    if len(embeddings) == 0:
        return np.zeros(embedding_dim)
    return np.mean(embeddings, axis=0)

# Configuración del modelo
model_path = "modelo_entrenado_final.pth"
numberbatch_path = "Conceptnet/numberbatch-en.txt"

embedding_dim = 300
num_classes = 6  # Cambiar según tu configuración

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassifier(embedding_dim, num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print(f"Modelo cargado desde {model_path}")

# Cargar vectores de ConceptNet
word_to_vec_map = load_conceptnet_vectors(numberbatch_path, embedding_dim)

# Probar con texto en lenguaje natural
while True:
    input_text = input("Ingresa un texto en lenguaje natural o escribe 'salir' para terminar: ")
    if input_text.lower() == "salir":
        break
    
    try:
        # Obtener embedding y convertir a tensor
        embedding = generate_sentence_embedding(input_text, word_to_vec_map, embedding_dim)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Realizar la predicción
        with torch.no_grad():
            output = model(embedding_tensor)
            _, predicted = torch.max(output, 1)
            print(f"Predicción: Clase {predicted.item()}")
    
    except Exception as e:
        print(f"Error: {e}. Asegúrate de ingresar un texto válido.")
