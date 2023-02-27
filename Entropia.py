# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:59:17 2023

@author: ALEXIS
"""

import numpy as np


Y = [0,1,0,1,0,1]
P = np.array(np.random.rand(5,1))

def cross_entropy(Y, P):
   Y = np.float_(Y)
   P = np.float_(P)
   return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

vect=[]

for i in Y:
    a=cross_entropy(Y[i], P[i])
    vect.append(a)
    
entrp =sum(vect)