import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.colors as mcolors

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

def generateSamplesOfS(S):
    listOfS = []
    delta = 0.05
    for i in range(1, 5):
         value = (i - 5)*delta + S
         listOfS.append(value)
    listOfS.append(S)
    for i in range(1,5):
        value =  i*delta + S
        listOfS.append(value)
    return listOfS

def profitAndPriceAnalysis(listOfS, t, M, N, L, T, K, sigma, r, quantity, prize, filename):
    solution = solveBlackScholesNumerically2(M, N, L, T, K, sigma, r)
    listOfProfits = [] # lista dos cenários de lucro/prejuízo no instante t
    listOfPrices = [] # lista com os possíveis preços unitário de uma opção no instante t 
    for S in listOfS:
        x = findVarX(S, t, r, sigma, T, K)
        tau = findVarTau(t, T)
        i = findClosestX(x, L, N)[1]
        j = findClosestTau(tau, T, M)[1]
        v_ij = solution[1][i][j]
        listOfPrices.append(v_ij)
        profit = quantity*v_ij - prize
        listOfProfits.append(profit)
    plt.plot(listOfS, listOfProfits)
    plt.xlabel('Cotação do ativo (R$) no instante ' + str(t) + ' ano')
    plt.ylabel('Lucro (R$)')
    plt.title("Análise de lucro/prejuízo do comprador")
    plt.legend(str(quantity) + "opções de compra")
    plt.savefig("graficos/"+filename+".png")
    plt. clf()

    table = np.zeros((len(listOfS), 3))
    columns = ["Preço unitário do ativo (S)", "Valor unitário da opção (V)", "Lucro da operação" ]
    table[:,0] = list(map(lambda S: round(S, 4), listOfS))
    table[:,1] = list(map(lambda P: round(P, 4), listOfPrices))
    table[:,2] = list(map(lambda P: round(P, 4), listOfProfits))
    table = table.tolist()
    for i in range(len(listOfS)):
        for j in range(3):
            table[i][j] = "R$ " + "{:.3f}".format(table[i][j]).replace('.', ',')
    fig, ax =plt.subplots(1,1)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText= table,
        colLabels=columns,
        cellLoc = "left",
        loc="center")
    plt.savefig("tabelas/"+filename+".png")    
    plt. clf()
    return (listOfProfits, listOfPrices)

def profitAndPriceAnalysisSpecificS(S, t, M, N, L, T, K, sigma, r, quantity, prize, filename):
    solution = solveBlackScholesNumerically2(M, N, L, T, K, sigma, r)
    x = findVarX(S, t, r, sigma, T, K)
    tau = findVarTau(t, T)
    i = findClosestX(x, L, N)[1]
    j = findClosestTau(tau, T, M)[1]
    v_ij = solution[1][i][j]    
    profit = quantity*v_ij - prize
    print("Analise de lucro/prejuizo do comprador")
    print("- Lucro no instante t: R$", profit)
    print("- Valor da opcao no instante t: R$", v_ij)
    return (profit, v_ij)
