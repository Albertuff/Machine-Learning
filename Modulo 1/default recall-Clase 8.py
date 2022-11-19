#Lo que se sugiere cuando se tienen datos desbalanceados es cambiar la metrica de desempeño.
#
# Deteccion de clientes que caen en impago a travez de la sensibilidad: Proporcion de casos reales positivos que son predichos positivos por el modelo

# Cargamos las librerias

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics

# Cargar los datos

datos=pd.read_csv("https://raw.githubusercontent.com/Albertuff/Machine-Learning/master/datos/default.csv")

# Definimos los atributos(features) y las variable objeivo(target)

X=datos[["Empleado","Balance","Salario_anual"]]           # Atributos
Y=datos["Impago"]                                         # Respuesta

# Dividimos al conjunto de datos en conjuntos de prueba y entrenamiento

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,train_size=0.90,random_state=1234)

# Modelo con k=7, y lo entrenamos

Modelo=KNeighborsClassifier(n_neighbors=5).fit(X_train,Y_train)

# Hacemos las predicciones

predicciones=Modelo.predict(X_test)

# Analizamos la matriz de confusión

plot_confusion_matrix(Modelo,X_test,Y_test)
plt.show()

# La sensibilidad del modelo

metrics.recall_score(Y_test,predicciones,pos_label=0)

# Resultado de dividir: Los casos verdaderos positivos (3) entre el total de casos verdaderos positivos  en la muestra (34)----> recall=3/34