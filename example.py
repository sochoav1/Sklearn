# Importar las bibliotecas necesarias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Cargar el conjunto de datos de Boston

housing = fetch_california_housing()
X = housing.data
y = housing.target
feature_names = housing.feature_names


# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
regressor = DecisionTreeRegressor(max_depth=3)
regressor.fit(X_train, y_train)

# Predecir valores para el conjunto de prueba
y_pred = regressor.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualizar el árbol de decisión
plt.figure(figsize=(20,10))
plot_tree(regressor, filled=True, feature_names=feature_names, rounded=True)
plt.show()
