import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 1. Dataset personalizado para CSV con embeddings en forma de string
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.embeddings = data['embedding'].apply(ast.literal_eval).tolist()  # Convertir string a lista
        self.labels = data['label'].values  # Etiquetas

    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return embedding, label

# 2. Modelo de clasificación
class TextClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

# 3. Función de entrenamiento
def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    for embeddings, labels in train_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 4. Función de evaluación
def evaluate_model(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 5. Cargar datos y ejecutar
if __name__ == "__main__":
    # Archivos CSV
    train_csv = 'Datos/texto_train_with_embeddings-Bert.csv'  # Ruta al CSV de entrenamiento
    test_csv = 'Datos/texto_test_with_embeddings-Bert.csv'    # Ruta al CSV de prueba

    # Crear datasets y dataloaders
    train_dataset = CSVDataset(train_csv)
    test_dataset = CSVDataset(test_csv)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("Dimensiones")
    # Dimensiones
    embedding_dim = len(train_dataset.embeddings[0])  # Dimensión del embedding
    num_classes = len(set(train_dataset.labels))  # Número de clases

    # Modelo, criterio, optimizador
    print("Modelo, criterio, optimizado")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifier(embedding_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print("Entrenamiento")
    # Entrenamiento
    for epoch in range(10):  # Número de épocas
        train_model(train_loader, model, criterion, optimizer, device)
        train_accuracy = evaluate_model(train_loader, model, device)
        test_accuracy = evaluate_model(test_loader, model, device)
        print(f"Epoch {epoch+1}: Train Accuracy = {train_accuracy:.2f}, Test Accuracy = {test_accuracy:.2f}")