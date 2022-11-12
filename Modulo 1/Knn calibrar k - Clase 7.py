# Sistema de predicción  para el diagnostico de predicción de diabetes.
# Utilizando el algoritmo de k vecinos mas cercanos
# Calibrar el valor de k (numero de vecinos)

# Observamos que el valor de k_estrella (el mejor modelo) depende del conjunto de entrenamiento-validacion y de prueba.
# Es decir, depende de como partimos el conjunto de datos, es decir, este resultado depende de la segunda division en el conjunto de datos.

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
X_train_val, X_test, Y_train_val, Y_test=train_test_split(X,Y,test_size=0.20,random_state=1234)
#Segunda division: Dividimos en conjunto de entrenamiento en Entrenamiento y Validacion 
X_train, X_val, Y_train,Y_val=train_test_split(X_train_val,Y_train_val,test_size=0.20,random_state=(0))

# Definir el modelo K_vecinos

Modelo_1=KNeighborsClassifier(n_neighbors=3,n_jobs=-1).fit(X_train,Y_train)
Modelo_2=KNeighborsClassifier(n_neighbors=5,n_jobs=-1).fit(X_train,Y_train)

predicciones_modelo_1=Modelo_1.predict(X_val)
predicciones_modelo_2=Modelo_2.predict(X_val)

# Medir el desempeño del modelo en el conjunto de validacion 

metrica_m1=metrics.accuracy_score(Y_val,predicciones_modelo_1)
metrica_m2=metrics.accuracy_score(Y_val,predicciones_modelo_2)
print(f"Predicciones en el conjunto de validacion Modelo 1 :  {metrica_m1:.4f}")
print(f"Predicciones en el conjunto de validacion Modelo 2 :  {metrica_m2:.4f}")

# HAsta aqui podemos resolver el problema de seleccionar el mejor modelo con un k especifico. K=3

# Este modelo, con k vecinos, que desempeño tiene en el modelo de prueba?
# Medir el desempeño del modelo en el conjunto de prueba
# Importante! : Una vez seleccionado el mejor modelo, es necesario reentrenarlo.
Modelo_k_estrella=KNeighborsClassifier(n_neighbors=3,n_jobs=-1).fit(X_train_val,Y_train_val)
predicciones_modelo_estrella=Modelo_k_estrella.predict(X_test)
metrica_estrella=metrics.accuracy_score(Y_test,predicciones_modelo_estrella)
print(f"Predicciones en el conjunto de prueba:  {metrica_estrella:.4f}")

# La metrica de desempeño de este modelo es de 93.1% 
# Ojo, Solo es un valor de la metrica, necesitamos mas valores de la metrica, y que ademas este valor no dependa del conjunto de prueba y entrenamiento

Mejor_k=max([metrica_m1,metrica_m2])
print(Mejor_k)
