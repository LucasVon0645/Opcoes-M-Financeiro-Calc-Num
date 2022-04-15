import numpy as np
import methods as met
import time

def main():

    print("\n*** Opcoes no mercado financeiro ***")
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
    filename = input("- nome base para arquivos de saida (somente o nome, sem '.'): ")

    with open("resultados/" + filename + ".txt", "w") as external_file:
        if (knownS == "s"):
            met.printFileIntroduction(external_file, N, L, sigma, K, T, r, t, quantity, S, knownS, St)
        else:
            met.printFileIntroduction(external_file, N, L, sigma, K, T, r, t, quantity, S, knownS)

        M = int(3*met.findMmin(L, T, N, sigma))

        print("\nparametro M calculado : ", str(M))
        print("\nparametro M calculado : " + str(M), file=external_file)


        print("\nSolucao da equacao de Black-Scholes")
        print("\nSolucao da equacao de Black-Scholes", file=external_file)
        testSolutionWithoutVectorization = input("deseja comparar os tempos das solucoes com e sem vetorizacao? (isso pode demorar mais) [s/n] : ")
        print("deseja comparar os tempos das solucoes com e sem vetorizacao? (isso pode demorar mais) [s/n] : ", testSolutionWithoutVectorization, file=external_file)

        print("aguarde alguns segundos...")
        t01 = time.time()
        if (testSolutionWithoutVectorization == "s"):
            met.solveBlackScholesNumerically1(M, N, L, T, K, sigma, r)
        tf1 = time.time()
        solution = met.solveBlackScholesNumerically2(M, N, L, T, K, sigma, r)
        tf2 = time.time()
        
        met.printTimeComparation(t01, tf1, tf2, external_file, testSolutionWithoutVectorization)

        print("\nResultados:")

        # Solução analítica para t = 0
        solutionAnalytical = met.solveBlackScholesAnalytically(S, K, sigma, r, T, 0)
        u = solutionAnalytical[0]
        v = solutionAnalytical[1]
        met.printAnalyticalSolution(u, v, external_file)

        # solução númerica sem interpolação para t = 0
        x = met.findVarX(S, 0, r, sigma, T, K)
        tau = met.findVarTau(0, T)
        i = met.findClosestX(x, L, N)[1]
        j = met.findClosestTau(tau, T, M)[1]
        u_ij = solution[0][i][j]
        v_ij = solution[1][i][j]
        met.printSimpleNumericSolution(i, j, u_ij, v_ij, u, v, external_file)

        # solução númerica com interpolação para t = 0
        v_interpolation = met.findNumericSolutionWithInterpolation(solution, x, tau, L, T, M, N)
        met.printNumericSolutionWithInterpolation(v_interpolation, v, external_file)

        print("\nAnalise de lucro (para t = " + str(t) + " ano)")
        print("- instante de tempo considerado (t): " + str(t) + " ano")
        print("- quantidade: " + str(quantity) + " opcoes de compra")
        print("- preco da opcao no momento da compra (V0): R$" + str(v))
        prize = v*quantity
        print("- premio total: R$" + str(prize) + " ( " + str(quantity) + " x " + "V0 )")
        met.printFileIntroductionProfitAnalysis(t, quantity, v, prize, external_file)

        if (knownS == 's'):
            print("- preco St do ativo: R$", St)
            print("- preco St do ativo: R$", St, file=external_file)
            met.profitAndPriceAnalysisSpecificS(St, t, M, N, L, T, K, sigma, r, quantity, prize, filename)
        else:
            listOfS = met.generateSamplesOfS(K)
            print("Gerando tabelas e graficos...")
            met.generateProfitAndPriceAnalysisGraphics( solution, listOfS, t, M, N, L, T, K, sigma, r, quantity, prize, filename)
            print("Tabela e grafico gerados para analise de lucros! Consulte as pastas 'tabelas' e 'graficos'")
            print("Tabela e grafico gerados para analise de lucros! Consulte as pastas 'tabelas' e 'graficos'", file=external_file)
            print("")
            priceForDifferentTimes = input("- Voce deseja analisar graficamente o valor dessa opcao para diferentes instantes? [s/n] : ")
            if (priceForDifferentTimes == "s"):
                met.generateGraphicOfOptionPriceOverDiferentTimes(solution, listOfS, t, M, N, L, T, K, sigma, r, quantity, prize, filename)

        print("\n*** fim ***")
        print("\n*** fim ***", file=external_file)

        external_file.close()

fim = False
while not fim :
    main()
    print("")
    print("")
    answer = input("Deseja testar um novo cenario? [s/n] : ")
    if (answer == "n"):
        fim = True

