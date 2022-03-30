import numpy as np
import methods as met
import matplotlib.pyplot as plt
import time

''' Cenário 1'''
# Constantes
K = 1.00
sigma = 0.01
T = 1.00
r = 0.01
L = 10
N = 10000

M = int(3*met.findMmin(L, T, N, sigma))

t01 = time.time()
solution1 = met.solveBlackScholesNumerically1(M, N, L, T, K, sigma, r)
tf1 = time.time()
solution2 = met.solveBlackScholesNumerically2(M, N, L, T, K, sigma, r)
tf2 = time.time()
print("tempo 1 : ", tf1 - t01)
print("tempo 2 : ", tf2 - tf1)

V = met.solveBlackScholesAnalytically(K, K, sigma, r, T, 0)
print("fim")