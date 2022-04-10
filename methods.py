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

    plt.plot(listOfS, listOfProfits, "")
    plt.xlabel('Cotação do ativo (R$) no instante ' + str(t) + ' ano')
    plt.ylabel('Lucro (R$)')
    plt.title("Análise de lucro/prejuízo do comprador")
    plt.grid(True, linestyle='--')

    textstr = '\n'.join((
    r'$\mu=%.2f$' % (1, ),
    r'$\mathrm{median}=%.2f$' % (1, ),
    r'$\sigma=%.2f$' % (sigma, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(-5, 60, textstr, fontsize = 22, 
         bbox = props)

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

def profitAndPriceAnalysisSpecificS(S, t, M, N, L, T, K, sigma, r, quantity, prize, external_file):
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

    print("Analise de lucro/prejuizo do comprador", file=external_file)
    print("- Lucro no instante t: R$", profit, file=external_file)
    print("- Valor da opcao no instante t: R$", v_ij, file=external_file)
    return (profit, v_ij)

def printFileIntroduction(external_file, N, L, sigma, K, T, r, t, quantity, S, knownS, St = 0):
    print("*** Opcoes no mercado financeiro ***", file=external_file)
    print("\n")
    print("Parametros")
    print("- N (o parametro de discretizacao no espaco)? : " + str(N), file=external_file)
    print("- L (o dominio da variavel x de espaco)? : " + str(L), file=external_file)
    print("- sigma (volatilidade anualizada)? : " + str(sigma), file=external_file)
    print("- K (preco de exercicio da opcao)? : " + str(K), file=external_file)
    print("- T (periodo de vencimento da opcao)? : " + str(T), file=external_file)
    print("- r (taxa de juros)? : " + str(r), file=external_file)
    print("- t (tempo, em anos, no qual o valor da opcao eh calculado)? : " + str(t), file=external_file)
    print("- quantidade de opcoes? : " + str(quantity), file=external_file)
    print("- S0 (preco do ativo em t = 0)? : " + str(S), file=external_file)
    print("- Voce deseja usar um valor conhecido de S em t (S_t)? [s/n] : " + str(knownS), file=external_file)
    if (knownS == "s"):
        print("- Qual é o St (preco do ativo no tempo t)? : " + str(St), file=external_file)

def printTimeComparation(t01, tf1, tf2, external_file, testSolutionWithoutVectorization):
    if (testSolutionWithoutVectorization == "s"):
        print("\ntempo da solucao numerica com 2 loops : ","{:.3f}".format(tf1 - t01) + " s")
    print("tempo da solucao com vetorizacao: ","{:.3f}".format(tf2 - tf1) + " s")

    if (testSolutionWithoutVectorization == "s"):
        print("\ntempo da solucao numerica com 2 loops : ","{:.3f}".format(tf1 - t01) + " s", file=external_file)
    print("tempo da solucao com vetorizacao: ","{:.3f}".format(tf2 - tf1) + " s", file=external_file)

def printAnalyticalSolution(u, v, external_file):
    print("\nSolucao analitica (para t = 0):")
    print("u = ", u)
    print("Preco da opcao de compra de 1 unidade: v = R$", v)

    print("\nResultados:", file=external_file)
    print("\nSolucao analitica (para t = 0):", file=external_file)
    print("u = " +str(u), file=external_file)
    print("Preco da opcao de compra de 1 unidade: v = R$"+str(v), file=external_file)

def printSimpleNumericSolution(i, j, u_ij, v_ij, u, v, external_file):
    print("\nSolucao numerica sem interpolacao (para t = 0): ")
    print("")
    print("i = ", i)
    print("j = ", j)
    print("u_ij = ", u_ij)
    print("preco da opcao de compra de 1 unidade: v = R$", v_ij)
    print("Diferencas em relacao a solucao analitica: ")
    print("| delta u | = ", np.abs(u_ij - u))
    print("| delta v | = ", np.abs(v_ij - v))
    print("")

    print("\nSolucao numerica sem interpolacao (para t = 0): ", file=external_file)
    print("", file=external_file)
    print("i = ", i, file=external_file)
    print("j = ", j, file=external_file)
    print("u_ij = ", u_ij, file=external_file)
    print("preco da opcao de compra de 1 unidade: v = R$", v_ij, file=external_file)
    print("Diferencas em relacao a solucao analitica: ", file=external_file)
    print("| delta u | = ", np.abs(u_ij - u), file=external_file)
    print("| delta v | = ", np.abs(v_ij - v), file=external_file)
    print("", file=external_file)

def printNumericSolutionWithInterpolation(v_interpolation, v, external_file):
    print("\nSolucao numerica com interpolacao (para t = 0): ")
    print("")
    print("preco da opcao de compra de 1 unidade: v = R$", v_interpolation)
    print("Diferenca em relacao a solucao analitica: ")
    print("| delta v | = ", np.abs(v_interpolation - v))

    print("\nSolucao numerica com interpolacao (para t = 0): ", file=external_file)
    print("", file=external_file)
    print("preco da opcao de compra de 1 unidade: v = R$", v_interpolation, file=external_file)
    print("Diferenca em relacao a solucao analitica: ", file=external_file)
    print("| delta v | = ", np.abs(v_interpolation - v), file=external_file)

def printFileIntroductionProfitAnalysis(t, quantity, v, prize, external_file):
    print("\nAnalise de lucro (para t =" + str(t) + " ano)", file=external_file)
    print("- instante de tempo considerado (t): " + str(t) + " ano", file=external_file)
    print("- quantidade: " + str(quantity) + " opcoes de compra", file=external_file)
    print("- preco da opcao no momento da compra (V0): R$" + str(v), file=external_file)
    print("- premio total: R$" + str(prize) + " ( " + str(quantity) + " x " + "V0 )", file=external_file)