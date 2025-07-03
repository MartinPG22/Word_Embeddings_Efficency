import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los embeddings guardados desde los archivos CSV
df_train_with_embeddings = pd.read_csv('Datos/texto_train_with_embeddings-ConceptNet.csv')
df_test_with_embeddings = pd.read_csv('Datos/texto_test_with_embeddings-ConceptNet.csv')

# Convertir las columnas 'embedding' de las listas a arrays de NumPy
df_train_with_embeddings['embedding'] = df_train_with_embeddings['embedding'].apply(lambda x: np.array(eval(x)))
df_test_with_embeddings['embedding'] = df_test_with_embeddings['embedding'].apply(lambda x: np.array(eval(x)))

# Ahora puedes usar estos embeddings directamente
X_train = np.array(df_train_with_embeddings['embedding'].tolist())
y_train = df_train_with_embeddings['label'].values
X_test = np.array(df_test_with_embeddings['embedding'].tolist())
y_test = df_test_with_embeddings['label'].values

"""
### ESTO ES EL GRID SEARCH PARA ENCONTRAR LOS MEJORES HIPERPARÁMETROS
# Configurar la búsqueda de hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400, 500],      # Número de árboles
    'max_depth': [None, 10, 20, 30],                    # Profundidad máxima del árbol
    'min_samples_split': [2, 5, 10, 20, 30, 40],        # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 4]                       # Mínimo de muestras en un nodo hoja
}

# Crear el clasificador base
rf = RandomForestClassifier(random_state=42)
# Los mejores son max depth: 20, min samples leaf: 4, min samples split: 10, n estimators: 200, accuracy: 0.34
# Configurar el GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,  # Validación cruzada con 5 particiones
    scoring='accuracy',
    verbose=2,
    n_jobs=-1  # Usar todos los núcleos disponibles
)

# Ejecutar la búsqueda
print("Iniciando Grid Search...")
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo y parámetros
best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"Mejores parámetros encontrados: {best_params}")


# Evaluar el modelo en el conjunto de prueba
print("Evaluando el modelo en el conjunto de prueba...")
y_pred = best_rf.predict(X_test)
print(f"Accuracy en el conjunto de prueba: {accuracy_score(y_test, y_pred)}")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Opcional: Guardar el modelo entrenado
joblib.dump(best_rf, 'random_forest_best.pkl')
print("Modelo entrenado y guardado como 'random_forest_best.pkl'.")"""

# Entrenar el modelo con los mejores hiperparámetros
best_rf = RandomForestClassifier(
    n_estimators=500,      # Número de árboles
    max_depth=20,          # Profundidad máxima
    min_samples_split=20,  # Muestras mínimas para dividir
    min_samples_leaf=2,    # Muestras mínimas en una hoja
    random_state=42
)

print("Entrenando el modelo con los mejores hiperparámetros...")
best_rf.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = best_rf.predict(X_test)

# Evaluación del modelo
print(f"Accuracy en el conjunto de prueba: {accuracy_score(y_test, y_pred)}")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Generar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Real")
plt.show()