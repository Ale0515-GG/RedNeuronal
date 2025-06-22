# Paso 1: Importamos las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding

# Paso 2: Cargar el archivo CSV
print("Cargando datos...")
df = pd.read_csv("comidas.csv", encoding="latin1")  # usamos 'latin1' para evitar errores de acentos

# Paso 3: Visualizamos las primeras filas del dataset
print("Primeras filas del dataset:")
print(df.head())

# Paso 4: Verificamos las columnas disponibles
print("\nColumnas disponibles:")
print(df.columns)

# Paso 5: Convertimos la columna 'tipo_preparacion' (rápido/elaborado) en valores numéricos
le_tipo = LabelEncoder()
df['tipo_preparacion'] = le_tipo.fit_transform(df['tipo_preparacion'])

# Paso 6: Definimos variables predictoras (X) y variable objetivo (y)
# Excluimos columnas de texto irrelevantes para la red neuronal
X = df.drop(columns=["id", "nombre_receta", "ingredientes", "etiqueta", "saludable"])
y = df["saludable"]

# Paso 7: Escalamos los datos para mejorar el rendimiento de la red neuronal
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 8: Dividimos el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Paso 9: Definimos la arquitectura de la red neuronal
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Capa oculta con 64 neuronas
model.add(Dense(32, activation='relu'))  # Otra capa oculta con 32 neuronas
model.add(Dense(1, activation='sigmoid'))  # Capa de salida binaria (0 o 1)

# Paso 10: Compilamos el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Paso 11: Entrenamos el modelo
print("\nEntrenando el modelo...")
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

# Paso 12: Evaluamos el modelo en el conjunto de prueba
print("\nEvaluando en datos de prueba:")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Precisión en datos de prueba: {accuracy:.2f}")

# Paso 13: Graficamos el historial de entrenamiento
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión durante el entrenamiento')
plt.show()

# Paso 14: Hacemos predicciones con los datos de prueba
y_pred_prob = model.predict(X_test)  # Predicciones como probabilidad (entre 0 y 1)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convertimos a 0 o 1 (umbral 0.5)

# Paso 15: Mostramos algunas predicciones reales vs predichas
print("\nEjemplos de predicciones:")
for real, pred in zip(y_test[:10], y_pred[:10]):
    print(f"Real: {real} - Predicho: {int(pred)}")

# Paso 16: Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No saludable", "Saludable"])
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión")
plt.show()

# Paso 17: Reporte de métricas de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["No saludable", "Saludable"]))

# Paso 18: Conclusión automática en base a la precisión
print("\nConclusión automática basada en la precisión:")
if accuracy >= 0.90:
    print("El modelo tiene una precisión **alta**. Puede identificar comidas saludables de forma confiable.")
elif accuracy >= 0.80:
    print("El modelo tiene un buen rendimiento general, aunque puede mejorar con más datos o ajuste de parámetros.")
else:
    print("El modelo necesita mejoras. Puede requerir más datos, limpieza o una red neuronal más profunda.")
