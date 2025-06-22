# Paso 1: Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Paso 2: Cargar el archivo CSV
print("Cargando datos...")
df = pd.read_csv("comidas.csv", encoding="latin1")

# Paso 3: Visualizar las primeras filas del dataset
print("Primeras filas del dataset:")
print(df.head())

# Paso 4: Preprocesar los datos
le_tipo = LabelEncoder()
df['tipo_preparacion'] = le_tipo.fit_transform(df['tipo_preparacion'])  # rápido/elaborado → 0/1

le_etiqueta = LabelEncoder()
df['etiqueta_encoded'] = le_etiqueta.fit_transform(df['etiqueta'])  # texto → número

# Guardar las clases para luego mostrar el nombre correcto de la etiqueta
etiqueta_clases = le_etiqueta.classes_

# Paso 5: Variables para modelo 1 (saludable)
X1 = df.drop(columns=["id", "nombre_receta", "ingredientes", "etiqueta", "saludable", "etiqueta_encoded"])
y1 = df["saludable"]

# Paso 6: Variables para modelo 2 (etiqueta)
X2 = X1.copy()
y2 = df["etiqueta_encoded"]

# Paso 7: Escalado de datos para mejor rendimiento
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)
X2_scaled = scaler.transform(X2)  # usamos el mismo scaler

# Paso 8: División de los datos en entrenamiento y prueba
X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2_scaled, y2, test_size=0.2, random_state=42)

# Paso 9: Crear y entrenar el modelo 1 (Saludabilidad)
model1 = Sequential()
model1.add(Dense(64, input_dim=X1_train.shape[1], activation='relu'))
model1.add(Dense(32, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))  # salida binaria

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\nEntrenando modelo de saludabilidad...")
history1 = model1.fit(X1_train, y1_train, epochs=20, batch_size=16, validation_split=0.2)

# Evaluar modelo 1
loss1, accuracy1 = model1.evaluate(X1_test, y1_test)
print(f"✅ Precisión del modelo de saludabilidad: {accuracy1*100:.2f}%")

# Paso 10: Graficar desempeño del modelo 1
plt.figure(figsize=(8, 4))
plt.plot(history1.history['accuracy'], label='Entrenamiento')
plt.plot(history1.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo de saludabilidad')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Paso 11: Crear y entrenar el modelo 2 (Etiqueta)
model2 = Sequential()
model2.add(Dense(64, input_dim=X2_train.shape[1], activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(len(etiqueta_clases), activation='softmax'))  # salida multiclase

model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\nEntrenando modelo de etiquetas...")
history2 = model2.fit(X2_train, y2_train, epochs=20, batch_size=16, validation_split=0.2)

# Evaluar modelo 2
loss2, accuracy2 = model2.evaluate(X2_test, y2_test)
print(f"✅ Precisión del modelo de etiquetas: {accuracy2*100:.2f}%")

# Paso 12: Graficar precisión de ambos modelos para comparar
plt.figure(figsize=(6, 5))
plt.bar(["Saludable", "Etiqueta"], [accuracy1*100, accuracy2*100], color=["#4CAF50", "#2196F3"])
plt.title("Comparación de precisión de los modelos")
plt.ylabel("Precisión (%)")
plt.ylim(0, 100)
for i, val in enumerate([accuracy1*100, accuracy2*100]):
    plt.text(i, val + 2, f"{val:.2f}%", ha='center', fontsize=12)
plt.tight_layout()
plt.show()

# Paso 13: Ingreso de nueva receta por el usuario
print("\n📥 Ingresa una receta para predecir:")

try:
    calorias = float(input("Calorías: "))
    proteinas = float(input("Proteínas: "))
    grasas = float(input("Grasas: "))
    carbohidratos = float(input("Carbohidratos: "))
    tiempo_min = float(input("Tiempo de preparación (minutos): "))
    tipo_preparacion_txt = input("Tipo de preparación (rápido/elaborado): ").strip().lower()

    # Validar entrada
    if tipo_preparacion_txt not in le_tipo.classes_:
        raise ValueError("Tipo de preparación inválido. Usa 'rápido' o 'elaborado'.")

    tipo_preparacion = le_tipo.transform([tipo_preparacion_txt])[0]

    # Construir arreglo con los datos
    nueva_receta = np.array([[calorias, proteinas, grasas, carbohidratos, tiempo_min, tipo_preparacion]])
    nueva_receta_scaled = scaler.transform(nueva_receta)

    # Modelo 1: ¿Es saludable?
    prob_saludable = model1.predict(nueva_receta_scaled)[0][0]
    resultado_saludable = "Saludable" if prob_saludable > 0.5 else "No saludable"

    # Modelo 2: ¿Qué etiqueta nutricional tiene?
    etiqueta_pred = model2.predict(nueva_receta_scaled)
    etiqueta_index = np.argmax(etiqueta_pred)
    etiqueta_nombre = etiqueta_clases[etiqueta_index]

    # Mostrar resultado
    print(f"\n🔍 Resultado de la predicción:")
    print(f"Probabilidad de ser saludable: {prob_saludable*100:.2f}% → {resultado_saludable}")
    print(f"Etiqueta nutricional predicha: {etiqueta_nombre}")

except Exception as e:
    print(f"❌ Error al ingresar los datos: {e}")
