# Sistema de predicción  para el diagnostico de predicción de diabetes.
# Utilizando el algoritmo de k vecinos mas cercanos
# Calibrar el valor de k (numero de vecinos)

# Paso 0: Cargar las librerias

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold, cross_val_score


# Paso 1: Cargar los datos

datos=pd.read_csv("https://raw.githubusercontent.com/Albertuff/Machine-Learning/master/datos/diabetes.csv")

# Definimos los atributos(features) y las variable objeivo(target)

X=datos[["glucose","insulin","sspg"]] # Atributos
Y=datos["class"]                      # Respuesta

# Vamos a dividir al conjunto de datos en 3 partes: Entrenamiento, Validación y Prueba

#Primera division: Entrenamiento y Prueba
X_train_val, X_test, Y_train_val, Y_test=train_test_split(X,Y,test_size=0.20)
#Segunda division: Dividimos en conjunto de entrenamiento en Entrenamiento y Validacion 
X_train, X_val, Y_train,Y_val=train_test_split(X_train_val,Y_train_val,train_size=0.80)


# Definir el modelo K_vecinos

K_vecinos=KNeighborsClassifier(n_neighbors=7,n_jobs=-1).fit(X_train,Y_train)
predicciones=K_vecinos.predict(X_val)

# Medir el desempeño del modelo en el conjunto de validacion 

metrica=metrics.accuracy_score(Y_val,predicciones)
print(f"Predicciones en el conjunto de validacion {metrica:.4f}")

# HAsta aqui podemos resolver el problema de seleccionar el mejor modelo con un k especifico.

# Este modelo, con k vecinos, que desempeño tiene en el modelo de prueba?
# Medir el desempeño del modelo en el conjunto de prueba

predicciones=K_vecinos.predict(X_test)
metrica = metrics.accuracy_score(Y_test,predicciones)
print(f"Predicciones en el conjunto de prueba {metrica:.4f}")