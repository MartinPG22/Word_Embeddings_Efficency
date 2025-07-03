import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential

# Cargar los embeddings guardados desde los archivos CSV
df_train_with_embeddings = pd.read_csv('Datos/texto_train_with_embeddings-Bert.csv')
df_test_with_embeddings = pd.read_csv('Datos/texto_test_with_embeddings-Bert.csv')

# Convertir las columnas 'embedding' de las listas a arrays de NumPy
df_train_with_embeddings['embedding'] = df_train_with_embeddings['embedding'].apply(lambda x: np.array(eval(x)))
df_test_with_embeddings['embedding'] = df_test_with_embeddings['embedding'].apply(lambda x: np.array(eval(x)))

# Preparar los datos
X_train = np.array(df_train_with_embeddings['embedding'].tolist())
y_train = df_train_with_embeddings['label'].values
X_test = np.array(df_test_with_embeddings['embedding'].tolist())
y_test = df_test_with_embeddings['label'].values

# Convertir etiquetas a one-hot encoding si es una tarea de clasificación multiclase
num_classes = len(np.unique(y_train))
if num_classes > 2:
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

# Crear el modelo de red neuronal
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))  # Capa oculta con 128 neuronas
model.add(Dropout(0.3))  # Regularización
model.add(Dense(64, activation='relu'))  # Segunda capa oculta con 64 neuronas
model.add(Dropout(0.3))  # Regularización
model.add(Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid'))  # Capa de salida

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
    metrics=['accuracy']
)

# Configurar el EarlyStopping para evitar sobreajuste
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo
print("Entrenando la red neuronal...")
history = model.fit(
    X_train, 
    y_train, 
    validation_split=0.2,  # Usar un 20% del conjunto de entrenamiento para validación
    epochs=50, 
    batch_size=32, 
    callbacks=[early_stopping],
    verbose=1
)

# Evaluar el modelo en el conjunto de prueba
print("Evaluando el modelo en el conjunto de prueba...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Loss en prueba: {test_loss}")
print(f"Accuracy en prueba: {test_accuracy}")

# Generar predicciones
y_pred = model.predict(X_test)
if num_classes == 2:
    y_pred = (y_pred > 0.5).astype(int)  # Para clasificación binaria
else:
    y_pred = np.argmax(y_pred, axis=1)  # Para clasificación multiclase

# Generar el reporte de clasificación
y_test_classes = np.argmax(y_test, axis=1) if num_classes > 2 else y_test
print("Reporte de Clasificación:")
print(classification_report(y_test_classes, y_pred))

# Guardar el modelo entrenado
model.save('neural_network_best.h5')
print("Modelo entrenado y guardado como 'neural_network_best.h5'.")
