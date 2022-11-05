# Sistema de predicción  para el diagnostico de predicción de diabetes.
# Utilizando el algoritmo de k vecinos mas cercanos
# Empleando la validacion cruxada con k repeticiones o capas (k-folds)

# Paso 0: Cargar las librerias

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import KFold # Para emplear la validación cruzada k repetida
from sklearn.model_selection import cross_val_score # Para calcular la lmetrica de desempeño en cada iteracion/capa
# Las ultimas dos librerias se pueden incorporar en una sola :
# from sklearn.model_selection import Kfold, cross_val_score


# Paso 1: Cargar los datos

datos=pd.read_csv("https://raw.githubusercontent.com/Albertuff/Machine-Learning/master/datos/diabetes.csv")

# Definimos los atributos(features) y las variable objeivo(target)

X=datos[["glucose","insulin","sspg"]] # Atributos
Y=datos["class"]                      # Respuesta

# Definir el modelo K_vecinos

K_vecinos=KNeighborsClassifier(n_neighbors=5,n_jobs=-1).fit(X,Y)

# Medir el desempeño del modelo con validacion cruzada por capas/k-repetida/k-folds 
# Definir el tipo de validacion

KFolds=KFold(n_splits=10,shuffle=True)

# Calcular la mterica de desempeño en cada una de las capas de prueba 
puntajes=cross_val_score(K_vecinos,X,Y,scoring="accuracy",cv=KFolds,n_jobs=-1)
#print(puntajes)

# Promedio de exactitud del modelo
print(f"Exactitud promedio del modelo {puntajes.mean():.4f}")