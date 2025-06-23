import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from scipy.sparse import hstack

# Paso 1: Cargar datos
print("Cargando datos...")
df = pd.read_csv("comidas.csv", encoding="latin1")

# Paso 2: Codificar variables categóricas
le_tipo = LabelEncoder()
df['tipo_preparacion'] = le_tipo.fit_transform(df['tipo_preparacion'])

le_etiqueta = LabelEncoder()
df['etiqueta_encoded'] = le_etiqueta.fit_transform(df['etiqueta'])
etiqueta_clases = le_etiqueta.classes_

print("Clases encontradas:", etiqueta_clases)
print("Cantidad de clases:", len(etiqueta_clases))
print("\nDistribución de clases:")
print(df['etiqueta'].value_counts(normalize=True))

# Paso 3: Preparar variables
X_num = df[["calorias", "proteinas", "grasas", "carbohidratos", "tiempo_min", "tipo_preparacion"]]
y = df["etiqueta_encoded"]

# TF-IDF sobre ingredientes
vectorizer = TfidfVectorizer()
X_ing = vectorizer.fit_transform(df["ingredientes"])

# Concatenar numéricos + TF-IDF (sparse matrix)
X_full = hstack([X_num, X_ing])

# Paso 4: División de datos
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# Paso 5: Escalado solo para variables numéricas
scaler = StandardScaler(with_mean=False)  # with_mean=False porque estamos usando sparse matrices
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Paso 6: Modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(len(etiqueta_clases), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\nEntrenando modelo de etiquetas nutricionales...")
history = model.fit(X_train.toarray(), y_train, epochs=10, batch_size=16, validation_split=0.2)

# Evaluación
loss, acc = model.evaluate(X_test.toarray(), y_test)
print(f"\n Precisión del modelo en test: {acc*100:.2f}%")

# Reporte de clasificación
from sklearn.metrics import classification_report
y_pred = np.argmax(model.predict(X_test.toarray()), axis=1)
print("\n  Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=etiqueta_clases))

# Gráfico
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

# Entrada manual
print("\n Ingresa los datos de tu receta para predecir su etiqueta nutricional:")

try:
    calorias = float(input("Calorías: "))
    proteinas = float(input("Proteínas: "))
    grasas = float(input("Grasas: "))
    carbohidratos = float(input("Carbohidratos: "))
    tiempo_min = float(input("Tiempo de preparación (minutos): "))
    tipo_txt = input("Tipo de preparación (rápido/elaborado): ").strip().lower()
    ingredientes_txt = input("Lista de ingredientes (separados por coma): ")

    if tipo_txt not in le_tipo.classes_:
        raise ValueError("Tipo inválido. Usa 'rápido' o 'elaborado'.")

    tipo_encoded = le_tipo.transform([tipo_txt])[0]

    # Datos numéricos
    datos_numericos = np.array([[calorias, proteinas, grasas, carbohidratos, tiempo_min, tipo_encoded]])

    # Procesar ingredientes
    datos_ingredientes = vectorizer.transform([ingredientes_txt])

    # Concatenar
    entrada_completa = hstack([datos_numericos, datos_ingredientes])
    entrada_completa = scaler.transform(entrada_completa)

    # Predicción
    pred_prob = model.predict(entrada_completa.toarray())
    etiqueta_index = np.argmax(pred_prob)
    etiqueta_nombre = etiqueta_clases[etiqueta_index]
    prob_etiqueta = pred_prob[0][etiqueta_index]

    # Clasificación saludable o no
    etiqueta_saludable_map = {et: (et != 'no saludable') for et in etiqueta_clases}
    es_saludable = etiqueta_saludable_map[etiqueta_nombre]
    texto_saludable = "Saludable" if es_saludable else "No saludable"

    print("\n Resultado de la predicción:")
    print(f"Etiqueta nutricional predicha: {etiqueta_nombre} (confianza: {prob_etiqueta*100:.2f}%)")
    print(f"Clasificación general: {texto_saludable}")

except Exception as e:
    print(f"Error al ingresar datos: {e}")
