import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Creando un diccionario con datos de diferentes características y si la persona está en la FRIENDZONE o no.
data = {
    'Nivel de simpatia': [8, 7, 10, 5, 9, 4, 8.5, 6, 7.5, 5.5, 6.5, 9, 8, 7.5, 5, 9.5, 4.5, 7, 6.5, 8, 5.5],
    'Amistad en meses': [24, 6, 36, 12, 48, 7, 30, 15, 18, 10, 20, 40, 14, 16, 5, 35, 8, 10, 12, 28, 9],
    'Regalos realizados': [5, 3, 10, 2, 7, 1, 6, 2, 4, 0, 3, 8, 3, 5, 1, 10, 0, 3, 2, 4, 2],
    'Mensajes diarios': [30, 50, 45, 15, 35, 10, 40, 20, 25, 10, 20, 42, 28, 30, 15, 50, 10, 22, 20, 32, 12],
    'Encuentros semanales': [2, 1, 4, 1, 3, 0, 3, 1, 2, 1, 2, 4, 2, 3, 1, 5, 0, 1, 2, 3, 1],
    'Recados': [3, 5, 0, 2, 4, 1, 3, 0, 1, 1, 2, 1, 3, 2, 1, 4, 0, 1, 2, 3, 1],
    'FRIENDZONE': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
}

# Convirtiendo el diccionario en un DataFrame para su fácil manipulación.
df = pd.DataFrame(data)

# Separando el DataFrame en características (X) y la variable objetivo (y: FRIENDZONE).
X = df.drop("FRIENDZONE", axis=1)
y = df["FRIENDZONE"]

# Dividiendo el conjunto de datos en entrenamiento y prueba (70% entrenamiento y 30% prueba).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creando un modelo de clasificación utilizando el algoritmo de bosques aleatorios.
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Haciendo predicciones con el conjunto de prueba.
y_pred = clf.predict(X_test)

# Imprimiendo métricas del modelo.
print(classification_report(y_test, y_pred))
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))

# Obteniendo la importancia de cada característica del modelo entrenado.
importances = clf.feature_importances_
features = X.columns

# Definiendo colores para la visualización.
colors = plt.cm.viridis(importances)

# Creando una gráfica de barras horizontales para visualizar la importancia de cada característica.
plt.figure(figsize=(12, 8))
bars = plt.barh(features, importances, color=colors, edgecolor='black')
plt.xlabel("Importancia", fontsize=15)
plt.ylabel("Característica", fontsize=15)
plt.title("Importancia de las Características para predecir FRIENDZONE", fontsize=18, fontweight='bold')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# Añadiendo una anotación para resaltar la característica más importante.
max_idx = importances.argmax()
plt.annotate(f'Máxima importancia: {importances[max_idx]:.2f}', 
             xy=(importances[max_idx], features[max_idx]), 
             xytext=(importances[max_idx]-0.2, max_idx+0.5), 
             arrowprops=dict(facecolor='red', arrowstyle='->'), 
             fontsize=12, 
             color='red', 
             ha='center')

# Invertiendo el eje y para que la característica con mayor importancia esté arriba.
plt.gca().invert_yaxis()

# Mostramos la gráfica
plt.tight_layout()
plt.show()