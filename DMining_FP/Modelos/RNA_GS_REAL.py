"""
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

# 2. Modelo de clasificación con más capas
class TextClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
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

# 3. Función de entrenamiento con registro de pérdida
def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for embeddings, labels in train_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)  # Pérdida promedio

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
    
    # Dimensiones
    embedding_dim = len(train_dataset.embeddings[0])  # Dimensión del embedding
    num_classes = len(set(train_dataset.labels))  # Número de clases

    # Modelo, criterio, optimizador
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifier(embedding_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Entrenamiento
    for epoch in range(20):  # Número de épocas
        avg_loss = train_model(train_loader, model, criterion, optimizer, device)
        train_accuracy = evaluate_model(train_loader, model, device)
        test_accuracy = evaluate_model(test_loader, model, device)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Train Accuracy = {train_accuracy:.2f}, Test Accuracy = {test_accuracy:.2f}")
"""
import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import numpy as np

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

# 2. Modelo de clasificación con más capas
class TextClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
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

# 3. Función de entrenamiento con registro de pérdida
def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for embeddings, labels in train_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)  # Pérdida promedio

# 4. Función de evaluación con matriz de confusión
def evaluate_model(loader, model, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calcular la matriz de confusión
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
    
    # Calcular métricas para cada clase
    metrics = {}
    for i in range(num_classes):
        tp = cm[i, i]  # Verdaderos positivos
        fp = cm[:, i].sum() - tp  # Falsos positivos
        fn = cm[i, :].sum() - tp  # Falsos negativos
        tn = cm.sum() - (tp + fp + fn)  # Verdaderos negativos
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        metrics[i] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    
    return cm, metrics

# 5. Cargar datos y ejecutar
if __name__ == "__main__":
    # Archivos CSV
    train_csv = 'Datos/texto_train_with_embeddings-Bert.csv'  # Ruta al CSV de entrenamiento
    test_csv = 'Datos/texto_test_with_embeddings-Bert.csv'    # Ruta al CSV de prueba

    # Crear datasets y dataloaders
    train_dataset = CSVDataset(train_csv)
    test_dataset = CSVDataset(test_csv)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Dimensiones
    embedding_dim = len(train_dataset.embeddings[0])  # Dimensión del embedding
    num_classes = len(set(train_dataset.labels))  # Número de clases

    # Modelo, criterio, optimizador
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifier(embedding_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # Entrenamiento
    for epoch in range(40):  # Número de épocas
        avg_loss = train_model(train_loader, model, criterion, optimizer, device)
        train_accuracy = evaluate_model(train_loader, model, device, num_classes)[1]
        test_accuracy = evaluate_model(test_loader, model, device, num_classes)[1]
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Imprimir métricas de la clase
        print("Train Metrics:")
        for i in range(num_classes):
            print(f"Class {i}: {train_accuracy[i]}")
        
        print("Test Metrics:")
        for i in range(num_classes):
            print(f"Class {i}: {test_accuracy[i]}")
