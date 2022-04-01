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
        u[i][j] = K*max(np.exp(x_i) - 1, 0)
        v[i][j] = u[i][j]*np.exp(-r*tal_j)

    u[N][j] = K*np.exp(L + (sigma**2)*tal_j/2)
    v[N][j] = u[N][j]*np.exp(-r*tal_j)

    # determinar as demais linhas da matriz u e da matriz v com 2 loops
    for j in range(1, M + 1):
        tal_j = j*deltaTau

        for i in range (1, N):
            u[i][j] = u[i][j-1] + (deltaTau/(deltaX**2))*((sigma**2)/2)*(u[i-1][j-1] - 2*u[i][j-1] + u[i+1][j-1])
            v[i][j] = u[i][j]*np.exp(-r*tal_j)

        x = K*np.exp(L + (sigma**2)*tal_j/2)
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
        u[i][j] = K*max(np.exp(x_i) - 1, 0)
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

def findVarX (S, t, r, sigma, T, K):
    return np.log(S/K) + (r - (sigma**2)/2)*(T - t)

def findVarTau(t, T):
    return T - t

def solveBlackScholesAnalytically (S, K, sigma, r, T, t):
    tau = findVarTau(t, T)
    x = findVarX(S, t, r, sigma, T, K)
    d1 = (x + tau*(sigma**2))/(sigma*np.sqrt(tau))
    d2 = x/(sigma*np.sqrt(tau))
    u = K*np.exp(x+(sigma**2)*tau/2)*norm.cdf(d1) - K*norm.cdf(d2)
    v = S*norm.cdf(d1) - K*np.exp((-r)*(T - t))*norm.cdf(d2)
    return (u, v)

def findClosestTau(tau, T, M):
    deltaTau = T/M
    j = tau/deltaTau
    j_ceil = int(np.ceil(j))
    j_floor = int(np.floor(j))
    tau_ceil = deltaTau*j_ceil
    tau_floor = deltaTau*j_floor
    if (np.abs(tau - tau_ceil) <= np.abs(tau - tau_floor)):
        return (tau_ceil, j_ceil)
    return (tau_floor, j_floor)

def findClosestX(x, L, N):
    deltaX = 2*L/N
    i = (x + L)/deltaX
    i_ceil = int(np.ceil(i))
    i_floor = int(np.floor(i))
    x_ceil = deltaX*i_ceil - L
    x_floor = deltaX*i_floor - L
    if (np.abs(x - x_ceil) <= np.abs(x - x_floor)):
        return (x_ceil, i_ceil)
    return (x_floor, i_floor)

def findNumericSolutionWithInterpolation(numericSolution, x, tau, L, T, M, N):
    deltaTau = T/M
    deltaX = 2*L/N

    i = (x + L)/deltaX
    i_ceil = int(np.ceil(i))
    i_floor = int(np.floor(i))
    x_ceil = deltaX*i_ceil - L
    x_floor = deltaX*i_floor - L

    tauValues = findClosestTau(tau, T, M)
    j = tauValues[1]

    V = numericSolution[1]

    result = ((x_ceil - x)*V[i_floor][j] - (x_floor - x)*V[i_ceil][j])/(x_ceil - x_floor)
    
    return result
    