# Calibración de hiperparámetros en el problema de Default
# Resolvemos el problema de la selección de modelos 
# Calificamos el desempeño predictivo del mejor modelo
# Entrenamos el mejor modelo utilizando todo el conjunto de datos
# Resolvemos el problema de sobreajuste del modelo: es decir, tendremos al final un modelo que opera bien en el conjunto de entrenamiento y tiene la mejor capacidad de predicción,
# es decir, es capaz de generalizar para mas datos nuevos.

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# Cargar los datos

datos=pd.read_csv("https://raw.githubusercontent.com/Albertuff/Machine-Learning/master/datos/default.csv")

# Definimos los atributos(features) y las variable objeivo(target)

X=datos[["Empleado","Balance","Salario_anual"]]           # Atributos
Y=datos["Impago"]                                         # Respuesta

# Vamos a definir lo sposibles valores de los hiperparámetros

espacio_parametros={"n_neighbors":np.arange(1,20)}  # Buscamos el mejor valor de k entre 1 y 20.

# El algoritmo que vamos a entrenar es el de k vecinos mas cercanos.

Modelo=KNeighborsClassifier()

# Definimos el tipo de busqueda de k-
# Primero definimos la rejilla

cvr=RepeatedKFold(n_splits=10,n_repeats=10)

rejilla=GridSearchCV(Modelo,param_grid=espacio_parametros,cv=cvr,n_jobs=-1).fit(X,Y)  # El primer argumento es el modelo a entrenar, el segundo es el espacio donde están de los hiperparámetros,
                                                                  # es decir, el conjunto de los posibles hiperparámetros, el tercer argumento es una validación cruzada iterada10-folds
                                                                  # y en este caso utilizamos una cv 5-iterada 100 repetida.

print(f" La mejor configuración de hiperparámetros es K= {rejilla.best_params_}")
print(f" La exactitud mas alta alcanzada con los mejores hiperparámetros es : {rejilla.best_score_}")

# hasta aquí resolvimos el problema de seleccionar el mejor modelo k=6 funciona bien, y tiene un desempeño predictivo de 92.33%

# Esta metrica de desempeño puede se roptimista, necesitamos probar el modelo k=6 en datos nuevos o que el modelo no haya visto

# Cómo obtenemos una metrica mas realista...?  ---> Con validación cruzada anidada

# Definimos el esquema de validación cruzada interna que selecciona el mejor modelo
cv_interna=RepeatedKFold(n_splits=10,n_repeats=50)

# Definimos la validación cruzada externa que permite calificar el desempeño predictivo del mejor modelo seleccionado que imprime en rejilla.best_params_

cv_externa=KFold(n_splits=10)

# Realizamos la busqueda de la mejor configuración 

rejilla=GridSearchCV(Modelo,param_grid=espacio_parametros,n_jobs=-1,cv=cv_interna)

# Ya que encontró el mejor modelo, lo entrena en el conjunto de entrenamiento y lo prueba en el conjunto de prueba 
scores=cross_val_score(rejilla,X,Y,cv=cv_externa,n_jobs=-1)
print(f" El desempeño predictivo promedio del mejor modelo es : {scores.mean():.4f}")

# El desempeño predictivo del modelo fue de 90.86%, esta es una metrica mas realista del desempeño predictivo, ya que trabajó con datos que no habia visto.

# El modelo final

modelo_estrella=KNeighborsClassifier(n_neighbors=6).fit(X,Y)
predicciones=modelo_estrella.predict(X)
plot_confusion_matrix(modelo_estrella,X,Y)
plt.show()
