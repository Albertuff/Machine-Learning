# Sistema de predicción  para el diagnostico de predicción de diabetes.
# Utilizando el algoritmo de k vecinos mas cercanos

# Paso 0: Cargar las librerias

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix

# Paso 1: Cargar los datos

datos=pd.read_csv("https://raw.githubusercontent.com/Albertuff/Machine-Learning/master/datos/diabetes.csv")

