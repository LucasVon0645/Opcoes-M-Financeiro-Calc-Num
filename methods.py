import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def solveBlackScholesNumerically1(M, N, L, T, K, sigma, r):
    u = np.zeros((N + 1, M + 1))
    v = np.zeros((N + 1, M + 1))
    deltaX = 2*L/N
    deltaTau = T/M

    #determinar a primeira linha da matriz u e da matriz v
    j = 0

    tal_j = j*deltaTau
    for i in range (1, N):
        x_i = i * deltaX - L
        u[i][j] = K*max(np.exp(x_i), 0)
        v[i][j] = u[i][j]*np.exp(-r*tal_j)

    u[N][j] = K*np.exp(L + (sigma**2)*tal_j/2)
    v[N][j] = u[N][j]*np.exp(-r*tal_j)

    # determinar as demais linhas da matriz u e da matriz v com 2 loops
    for j in range(1, M + 1):
        tal_j = j*deltaTau

        for i in range (1, N):
            u[i][j] = u[i][j-1] + (deltaTau/(deltaX**2))*((sigma**2)/2)*(u[i-1][j-1] - 2*u[i][j-1] + u[i+1][j-1])
            v[i][j] = u[i][j]*np.exp(-r*tal_j)

        u[N][j] = K*np.exp(L + (sigma**2)*tal_j/2)
        v[N][j] = u[N][j]*np.exp(-r*tal_j)

    return (u, v)

def solveBlackScholesNumerically2(M, N, L, T, K, sigma, r):
    u = np.zeros((N + 1, M + 1))
    v = np.zeros((N + 1, M + 1))
    deltaX = 2*L/N
    deltaTau = T/M

    #determinar a primeira linha da matriz u e da matriz v
    j = 0

    tal_j = j*deltaTau
    for i in range (1, N):
        x_i = i * deltaX - L
        u[i][j] = K*max(np.exp(x_i), 0)
        v[i][j] = u[i][j]*np.exp(-r*tal_j)

    u[N][j] = K*np.exp(L + (sigma**2)*tal_j/2)
    v[N][j] = u[N][j]*np.exp(-r*tal_j)

    # determinar as demais linhas da matriz u e da matriz v com 1 loop em j e vetorização em i
    for j in range(1, M + 1):
        tal_j = j*deltaTau

        u_j_previous = u[:,j-1]
        u_j_previous_shiftR = np.roll(u_j_previous, 1)
        u_j_previos_shiftL = np.roll(u_j_previous, -1)

        u_j = u_j_previous + (deltaTau/(deltaX**2))*((sigma**2)/2)*(u_j_previos_shiftL - 2*u_j_previous + u_j_previous_shiftR)
        u_j[0] = 0 
        u_j[N] = K*np.exp(L + (sigma**2)*tal_j/2)
        u[:,j] = u_j
        v[:,j] = u_j*np.exp(-r*tal_j)
    return (u, v)

def findMmin (L, T, N, sigma):
    deltaX = 2*L/N
    return np.ceil((sigma**2)*T/(deltaX**2))

""" def calculateXi (T, M):
    x = []
    delta = 2*L/N
    for j in range(M):
        x_i = j*delta
        x.append(x_i)
    return x

def calculateTauj (L, N):
    tau = []
    delta = 2*L/N
    for i in range(N):
        tau_i = i*delta - L
        tau.append(tau_i)
    return tau """

def findVarX (S, t, r, sigma, T, K):
    return np.log(S/K) + (r - (sigma**2)/2)*(T - t)

def findVarTau(t, T):
    return T - t

def solveBlackScholesAnalytically (S, K, sigma, r, T, t):
    tau = findVarTau(t, T)
    x = findVarX(S, t, r, sigma, T, K)
    d1 = (x + tau*(sigma**2))/(sigma*np.sqrt(tau))
    d2 = x/(sigma*np.sqrt(tau))
    V = S*norm.cdf(d1) - K*np.exp((-r)*(T - t))*norm.cdf(d2)
    return V
    