# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:59:17 2023

@author: ALEXIS
"""

import numpy as np

#ENTROPIA DATOS ALEATORIOS

datos= 2
Y = np.array(np.random.randint(0,2,datos))
P = np.array(np.random.rand(datos,1))


def cross_entropy(Y, P):
   Y = np.float_(Y)
   P = np.float_(P)
   return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

entrop=cross_entropy(Y, P)
print('La Entropia Cruzada ALEATORIO es', entrop)

#%%
#ENTROPIA DATOS PROFE

import numpy as np
import pandas as pd


#Funcion Sigmoide
def sigmoid(x):
    return 1/(1+np.exp(-x))

def prediccion(X, W, b):
    return sigmoid((np.matmul(X,W)+b)) #elige el primer numero de la matriz o vector

def cross_entropy1(y1, P1):
   y1 = np.float_(y1)
   P1 = np.float_(P1)
   return -np.sum(y1 * np.log(P1) + (1 - y1) * np.log(1 - P1))


# Cargamos data
data = np.array(pd.read_csv('D:/Udenar/Semestre 10/Inteligencia Artificial/Talleres/Taller 6 Entropia Cruzada/data.csv', header=0))
#data= np.array(data)
X = data[:,0:2]
y1 = data[:,2]

#Inicializamos pesos y bias aleatorios 
W = np.array(np.random.rand(2,1)) 
b = np.random.rand(1)[0]

#Calculamos la probabilidad
P1=prediccion(X, W, b)


Entropiadatos=cross_entropy1(y1, P1)

print('Entropia cruzada de datos es',Entropiadatos)
