import numpy as np
import methods as met
import time

''' Cenário 1'''
# Constantes
K = 1.00
sigma = 0.01
T = 1.00
r = 0.01
L = 10
N = 10000
S = K

M = int(3*met.findMmin(L, T, N, sigma))

t01 = time.time()
solution1 = met.solveBlackScholesNumerically1(M, N, L, T, K, sigma, r)
tf1 = time.time()
solution2 = met.solveBlackScholesNumerically2(M, N, L, T, K, sigma, r)
tf2 = time.time()
print("tempo da solucao numerica com 2 loops : ", tf1 - t01)
print("tempo da solucao com vetorizacao: ", tf2 - tf1)

solutionAnalytical = met.solveBlackScholesAnalytically(S, K, sigma, r, T, 0)
u = solutionAnalytical[0]
v = solutionAnalytical[1]

print("Solucao analitica:")
print("u = ", u)
print("v = ", v)

""" Aproximacao simples """

x = met.findVarX(S, 0, r, sigma, T, K)
tau = met.findVarTau(0, T)
i = met.findClosestX(x, L, N)[1]
j = met.findClosestTau(tau, T, M)[1]
u_ij = solution2[0][i][j]
v_ij = solution2[1][i][j]

print("Solucao numerica simples: ")
print("")
print("i = ", i)
print("j = ", j)
print("u_ij = ", u_ij)
print("v_ij = ", v_ij)
print("Diferencas em relacao a solucao analitica: ")
print("| delta u | = ", np.abs(u_ij - u))
print("| delta v | = ", np.abs(v_ij - v))
print("")

v_interpolation = met.findNumericSolutionWithInterpolation(solution2, x, tau, L, T, M, N)
print("Solucao numerica com interpolacao: ")
print("")
print("v = ", v_interpolation)
print("Diferenca em relacao a solucao analitica: ")
print("| delta v | = ", np.abs(v_interpolation - v))

print("fim")