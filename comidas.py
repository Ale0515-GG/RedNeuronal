import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Paso 1: Cargar datos
print("Cargando datos...")
df = pd.read_csv("comidas.csv", encoding="latin1")

print("Primeras filas del dataset:")
print(df.head())

# Paso 2: Codificar variables categóricas
le_tipo = LabelEncoder()
df['tipo_preparacion'] = le_tipo.fit_transform(df['tipo_preparacion'])

le_etiqueta = LabelEncoder()
df['etiqueta_encoded'] = le_etiqueta.fit_transform(df['etiqueta'])
etiqueta_clases = le_etiqueta.classes_

# Crear un diccionario para saber qué etiquetas son saludables o no
# Aquí asumo que cualquier etiqueta que NO sea "no saludable" es saludable
etiqueta_saludable_map = {et: (et != 'no saludable') for et in etiqueta_clases}

# Paso 3: Preparar variables para modelo multiclasificación (solo etiquetas)
X = df[["calorias", "proteinas", "grasas", "carbohidratos", "tiempo_min", "tipo_preparacion"]]
y = df["etiqueta_encoded"]  # etiquetas nutricionales

# Paso 4: División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Paso 5: Escalado (ajustado solo con entrenamiento)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Paso 6: Modelo multiclasificación para predecir etiqueta
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(len(etiqueta_clases), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\nEntrenando modelo de etiquetas nutricionales...")
history = model.fit(X_train_scaled, y_train, epochs=30, batch_size=16, validation_split=0.2)

# Evaluación modelo
loss, acc = model.evaluate(X_test_scaled, y_test)
print(f"\n✅ Precisión del modelo en datos de prueba: {acc*100:.2f}%")

# Paso 7: Graficar precisión durante el entrenamiento
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.axhline(y=acc, color='r', linestyle='--', label=f'Precisión Test: {acc*100:.2f}%')
plt.title('Precisión del modelo de etiquetas nutricionales')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Paso 8: Entrada manual y predicción
print("\n📥 Ingresa los datos de tu receta para predecir su etiqueta nutricional:")

try:
    calorias = float(input("Calorías: "))
    proteinas = float(input("Proteínas: "))
    grasas = float(input("Grasas: "))
    carbohidratos = float(input("Carbohidratos: "))
    tiempo_min = float(input("Tiempo de preparación (minutos): "))
    tipo_txt = input("Tipo de preparación (rápido/elaborado): ").strip().lower()

    if tipo_txt not in le_tipo.classes_:
        raise ValueError("Tipo inválido. Usa 'rápido' o 'elaborado'.")

    tipo_encoded = le_tipo.transform([tipo_txt])[0]

    # Crear array con la nueva receta
    nueva_receta = np.array([[calorias, proteinas, grasas, carbohidratos, tiempo_min, tipo_encoded]])
    nueva_receta_scaled = scaler.transform(nueva_receta)

    # Predicción etiqueta
    pred_prob = model.predict(nueva_receta_scaled)
    etiqueta_index = np.argmax(pred_prob)
    etiqueta_nombre = etiqueta_clases[etiqueta_index]
    prob_etiqueta = pred_prob[0][etiqueta_index]

    # Saber si saludable o no según etiqueta
    es_saludable = etiqueta_saludable_map[etiqueta_nombre]
    texto_saludable = "✅ Saludable" if es_saludable else "❌ No saludable"

    print("\n🔍 Resultado de la predicción:")
    print(f"🏷️ Etiqueta nutricional predicha: {etiqueta_nombre} (confianza: {prob_etiqueta*100:.2f}%)")
    print(f"🧠 Clasificación general: {texto_saludable}")

except Exception as e:
    print(f"❌ Error al ingresar datos: {e}")
