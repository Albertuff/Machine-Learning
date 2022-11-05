# Sistema de predicción  para el diagnostico de predicción de diabetes.
# Utilizando el algoritmo de k vecinos mas cercanos

# Paso 0: Cargar las librerias

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split # Para partir el conjunto de datos en entrenamiento y prueba de manera automatica

# Paso 1: Cargar los datos

datos=pd.read_csv("https://raw.githubusercontent.com/Albertuff/Machine-Learning/master/datos/diabetes.csv")

# Definimos los atributos(features) y las variable objeivo(target)

X=datos[["glucose","insulin","sspg"]] # Atributos
Y=datos["class"]                      # Respuesta

# Paso 1.1: Definimos los conjuntos de prueba y entrenamiento

# Nota: train_test_split(atributos,target,train_size=porcion de los datos que se usan para entrenar,random_state=semilla para reproducir los resultados)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.80)

# Paso 2: Entrenar el modelo K_vecinos

K_vecinos=KNeighborsClassifier(n_neighbors=5,n_jobs=-1).fit(X_train,Y_train)

# Paso 3: Hacer las predicciones 

predicciones=K_vecinos.predict(X_train)
metrica=metrics.accuracy_score(Y_train,predicciones)
print(f"Exactitud en el conjunto de entrenamiento {metrica:.4f}")

# En el conjunto de prueba
predicciones=K_vecinos.predict(X_test)
metrica=metrics.accuracy_score(Y_test,predicciones)
print(f"Exactitud en el conjunto de prueba {metrica:.4f}")

