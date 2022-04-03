import numpy as np
import methods as met
import time

def main():

    print("*** Opcoes no mercado financeiro ***")
    print("\n")
    print("Parametros")
    N = int(input("- N (o parametro de discretizacao no espaco)? : "))
    L = int(input("- L (o dominio da variavel x de espaco)? : "))
    sigma = float(input("- sigma (volatilidade anualizada)? : "))
    K = float(input("- K (preco de exercicio da opcao)? : "))
    T = float(input("- T (periodo de vencimento da opcao)? : "))
    r = float(input("- r (taxa de juros)? : "))
    t = float(input("- t (tempo, em anos, no qual o valor da opcao eh calculado)? : "))
    quantity = int(input("- quantidade de opcoes? : "))
    S = float(input("- S0 (preco do ativo em t = 0)? : "))
    knownS = input("- Voce deseja usar um valor conhecido de S em t (S_t)? [s/n] : ")
    if (knownS == "s"):
        St = input("- Qual é o St (preco do ativo no tempo t)? : ")
    filename = input("- nome base para arquivos de saida: ")


    M = int(3*met.findMmin(L, T, N, sigma))

    print("\nparametro M calculado : ", str(M))

    print("\nSolucao da equacao de Black-Scholes")
    print("aguarde alguns segundos...")
    t01 = time.time()
    met.solveBlackScholesNumerically1(M, N, L, T, K, sigma, r)
    tf1 = time.time()
    solution = met.solveBlackScholesNumerically2(M, N, L, T, K, sigma, r)
    tf2 = time.time()
    print("\ntempo da solucao numerica com 2 loops : ","{:.3f}".format(tf1 - t01).replace('.', ',')+" s")
    print("tempo da solucao com vetorizacao: ","{:.3f}".format(tf2 - tf1).replace('.', ',')+" s")

    solutionAnalytical = met.solveBlackScholesAnalytically(S, K, sigma, r, T, 0)
    u = solutionAnalytical[0]
    v = solutionAnalytical[1]

    print("\nSolucao analitica (para t = 0):")
    print("u = ", u)
    print("v (premio da opcao de compra) = ", v)

    """ Aproximacao simples (sem interpolacao)"""

    x = met.findVarX(S, 0, r, sigma, T, K)
    tau = met.findVarTau(0, T)
    i = met.findClosestX(x, L, N)[1]
    j = met.findClosestTau(tau, T, M)[1]
    u_ij = solution[0][i][j]
    v_ij = solution[1][i][j]

    print("\nSolucao numerica simples (para t = 0): ")
    print("")
    print("i = ", i)
    print("j = ", j)
    print("u_ij = ", u_ij)
    print("v_ij (premio da opcao de compra) = R$", v_ij)
    print("Diferencas em relacao a solucao analitica: ")
    print("| delta u | = ", np.abs(u_ij - u))
    print("| delta v | = ", np.abs(v_ij - v))
    print("")

    v_interpolation = met.findNumericSolutionWithInterpolation(solution, x, tau, L, T, M, N)
    print("\nSolucao numerica com interpolacao (para t = 0): ")
    print("")
    print("v (premio da opcao de compra) = R$", v_interpolation)
    print("Diferenca em relacao a solucao analitica: ")
    print("| delta v | = ", np.abs(v_interpolation - v))

    print("\nAnalise de lucro")
    print("- instante de tempo considerado (t): " + str(t) + " ano")
    print("- quantidade: " + str(quantity) + " opcoes de compra")
    print("- preco da opcao no momento da compra (V0): R$" + str(v))
    prize = v*quantity
    print("- premio: R$" + str(prize) + " ( " + str(quantity) + " x " + "V0 )")
    if (knownS == 's'):
        print("- preco St do ativo: R$", St)
        met.profitAndPriceAnalysisSpecificS(St, t, M, N, L, T, K, sigma, r, quantity, prize, filename)
    else:
        listOfS = met.generateSamplesOfS(K)
        met.profitAndPriceAnalysis(listOfS, t, M, N, L, T, K, sigma, r, quantity, prize, filename)
        print("Tabela e grafico gerados")
    print("")


    print("*** fim ***")

fim = False
while not fim :
    main()
    print("")
    print("")
    answer = input("Deseja testar um novo cenario? [s/n] : ")
    if (answer == "n"):
        fim = True

