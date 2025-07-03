import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

"""class TextClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(TextClassifier, self).__init__()
        
        # Bloque residual
        def residual_block(in_dim, out_dim, dropout_prob):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(out_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            )
        
        self.initial_layer = nn.Sequential(
            nn.Linear(embedding_dim, 2048),  # Aumentamos el número de neuronas iniciales
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Bloques residuales
        self.residual_block1 = residual_block(2048, 1024, 0.4)
        self.residual_block2 = residual_block(1024, 512, 0.4)
        self.residual_block3 = residual_block(512, 256, 0.3)

        # Proyecciones para las conexiones residuales
        self.projection1 = nn.Linear(2048, 1024)  # Proyección del bloque 1
        self.projection2 = nn.Linear(1024, 512)  # Proyección del bloque 2
        self.projection3 = nn.Linear(512, 256)   # Proyección del bloque 3

        # Bloques finales
        self.final_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.initial_layer(x)
        
        # Residual Block 1
        x1 = self.residual_block1(x)
        x = self.projection1(x) + x1  # Ajustar dimensiones
        
        # Residual Block 2
        x2 = self.residual_block2(x)
        x = self.projection2(x) + x2  # Ajustar dimensiones
        
        # Residual Block 3
        x3 = self.residual_block3(x)
        x = self.projection3(x) + x3  # Ajustar dimensiones
        
        # Final Layers
        x = self.final_layers(x)
        return x
"""
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

# Función actualizada para visualizar la matriz de confusión
def evaluate_and_plot_cm(loader, model, device, num_classes, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    # Realizar predicciones
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcular la matriz de confusión
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))

    # Visualización de la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False, 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Real")
    plt.show()
    
    return cm


"""# 5. Cargar datos y ejecutar
if __name__ == "__main__":
    # Archivos CSV
    train_csv = 'Datos/texto_train_with_embeddings-ConceptNet.csv'  # Ruta al CSV de entrenamiento
    test_csv = 'Datos/texto_test_with_embeddings-ConceptNet.csv'    # Ruta al CSV de prueba

    # Crear datasets y dataloaders
    train_dataset = CSVDataset(train_csv)
    test_dataset = CSVDataset(test_csv)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
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
        train_accuracy = evaluate_model(train_loader, model, device, num_classes)[1]
        test_accuracy = evaluate_model(test_loader, model, device, num_classes)[1]
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Imprimir métricas de la clase
        print("Train Metrics:")
        for i in range(num_classes):
            print(f"Class {i}: {train_accuracy[i]}")
        
        print("Test Metrics:")
        for i in range(num_classes):
            print(f"Class {i}: {test_accuracy[i]}")"""

# 5. Cargar datos y ejecutar
if __name__ == "__main__":
    # Archivos CSV
    train_csv = 'Datos/texto_train_with_embeddings-ConceptNet.csv'
    test_csv = 'Datos/texto_test_with_embeddings-ConceptNet.csv'

    # Crear datasets y dataloaders
    train_dataset = CSVDataset(train_csv)
    test_dataset = CSVDataset(test_csv)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Dimensiones
    embedding_dim = len(train_dataset.embeddings[0])
    num_classes = len(set(train_dataset.labels))

    # Nombres de las clases
    class_names = [f"Clase {i}" for i in range(num_classes)]

    # Modelo, criterio, optimizador
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifier(embedding_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Entrenamiento y evaluación
    num_epochs = 20
    for epoch in range(num_epochs):
        avg_loss = train_model(train_loader, model, criterion, optimizer, device)
        train_accuracy = evaluate_model(train_loader, model, device, num_classes)[1]
        test_accuracy = evaluate_model(test_loader, model, device, num_classes)[1]
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")

         # Imprimir métricas de la clase
        print("Train Metrics:")
        for i in range(num_classes):
            print(f"Class {i}: {train_accuracy[i]}")
        
        print("Test Metrics:")
        for i in range(num_classes):
            print(f"Class {i}: {test_accuracy[i]}")
    
    # Guardar el modelo
    model_path = "modelo_entrenado_final.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")

    # Evaluar solo en el último test de prueba
    print("Evaluando el modelo en el conjunto de prueba final...")
    evaluate_and_plot_cm(test_loader, model, device, num_classes, class_names)

