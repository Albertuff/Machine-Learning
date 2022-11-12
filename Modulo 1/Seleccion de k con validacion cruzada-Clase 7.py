
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
from sklearn.model_selection import KFold, cross_val_score


# Paso 1: Cargar los datos

datos=pd.read_csv("https://raw.githubusercontent.com/Albertuff/Machine-Learning/master/datos/diabetes.csv")

# Definimos los atributos(features) y las variable objeivo(target)

X=datos[["glucose","insulin","sspg"]] # Atributos
Y=datos["class"]                      # Respuesta


#Primera division: Entrenamiento y Prueba

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.20)

#Reservamos el conjunto de prueba, no lo tocamos hasta el final donde ya podemos evaluar el desempeño del mejor modelo

# Definir el modelo K_vecinos

Modelo_1=KNeighborsClassifier(n_neighbors=3)
Modelo_2=KNeighborsClassifier(n_neighbors=5)

# Para eliminar el efecto del conjunto de validacion y de entrenamiento utilizamos un esquema de validacion cruzada.
val_cruzada_k=KFold(n_splits=10)

# Vamos a medir el desempeño predictivo de cada uno de los modelos en el conjunto de entrenamiento.

# Modelo_1:

scores_Modelo_1=cross_val_score(Modelo_1,X_train,Y_train,scoring="accuracy",cv=val_cruzada_k,n_jobs=-1)
scores_Modelo_2=cross_val_score(Modelo_2,X_train,Y_train,scoring="accuracy",cv=val_cruzada_k,n_jobs=-1)
#print(scores_Modelo_1)
#print(scores_Modelo_2)
# Comparamos el desempeño predictivo de ambos modelos.

print(f" Desempeño predictivo del Modelo 1 en el conjunto de entrenamiento :  {scores_Modelo_1.mean():.4f}")
print(f" Desempeño predictivo del Modelo 2 en el conjunto de entrenamiento :  {scores_Modelo_2.mean():.4f}")

# Hasta aquí eliminamos el problema de la selección de que el modelo dependa del conjunto de entrenamiento y validación

# Lo importante es que pudimos seleccionar al mejor modelo sin tocar al conjunto de prueba

Modelo_k_estrella=KNeighborsClassifier(n_neighbors=3).fit(X_train,Y_train)
predicciones=Modelo_k_estrella.predict(X_test)

# Medimos el desempeño del modelo

cal_modelo_estrella=metrics.accuracy_score(Y_test,predicciones)
print(f" El desempeño del modelo con k=3 es : {cal_modelo_estrella*100:.4f}%")

# Este valor no es un estimador insesgado del modelo k_estrella
# El resultado depende del conjunto que elejimos para prueba