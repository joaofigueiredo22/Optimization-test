#!/usr/bin/env python


# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

# Prepare the data
x = np.linspace(-1, 1, 100)

# Read from the command line, the function degree
ap = argparse.ArgumentParser()
ap.add_argument("-ngrau", "--numero_grau", help="Function Degree", type=int, required=True)
ap.add_argument("-maxit", "--max_iterations", help="Maximum number of iterations", type=int, required=False, default=500)
args = vars(ap.parse_args())
ngrau = args['numero_grau']

if ngrau < 0:
    ngrau = 0

# First function
def f1(xx):
    return xx * np.cos(xx)
    #  return xx


# Second function
def f2(xx, vv, grau):
    f2_1 = 0
    for numgrau in range(grau + 1):
        f2_1 = f2_1 + vv[numgrau] * xx ** numgrau
    return f2_1


# Error function
def f3(xx, vv, grau):
    f=f1(xx)
    for numgrau in range(grau + 1):
        f = f - vv[numgrau] * xx ** numgrau
    return f

# Create vetor with variables


def createv(grau1):
    vv = np.zeros(grau1 + 1, dtype=float)
    for numgrau1 in range(0, grau1 + 1):
        vv[numgrau1] = random.uniform(-2, 2)
    return vv


# Function to help adding all the values of one array
def calcular_erro(lista):
    soma = 0
    for numero in lista:
        soma += np.absolute(numero)
    # erro_medio=soma/len(lista)
    # return erro_medio
    return soma
    # return np.amax(np.absolute(lista))


# First guess error
v = createv(ngrau)
err1 = calcular_erro(f3(x, v, ngrau))


plt.plot(x, f1(x), 'b', label='Funcao original')
plt.plot(x, f2(x, v, ngrau), 'r', label='Funcao otimizada')
# Add a legend
plt.legend()
plt.title("Funcao de grau " + str(ngrau) + "   Iteracao: " + str(0) + "\n  Erro Total: " + str(err1))

dir = 0.0001
step = 0.1

# Main loop
for i in range(args['max_iterations']):

    ind = 0

    # direcoes=np.zeros((1, 5), np.float32)
    direcoes = np.zeros(ngrau+1)

    for r in v:
        r1 = r+dir
        r2 = r-dir

        v1 = v * 1
        v1[ind] = r1 * 1
        v2 = v * 1
        v2[ind] = r2 * 1
        erro1 = calcular_erro(f3(x, v1, ngrau))
        erro2 = calcular_erro(f3(x, v2, ngrau))

        if erro2 < erro1:

            direcoes[ind] = -1

        elif erro1 < erro2:

            direcoes[ind] = 1

        ind += 1

    if calcular_erro(direcoes) == 0:
        break

    newv = v + np.dot(step,direcoes)

    erro = calcular_erro(f3(x, v, ngrau))
    newerro = calcular_erro(f3(x, newv, ngrau))

    if newerro < erro:
        v = newv * 1
        plt.clf()
        # Plot the data
        plt.plot(x, f1(x), 'b', label='Funcao original')
        plt.plot(x, f2(x, v, ngrau), 'r', label='Funcao otimizada')
        # Add a legend
        plt.legend()
        plt.title("Funcao de grau " + str(ngrau) + "   Iteracao: " + str(i) + "\n  Erro Total: " + str(erro))
        plt.pause(0.00001)
    else:
        step = step/2

    # a1 = random.uniform(-1, 1)
    # b1 = random.uniform(-1, 1)
    # c1 = random.uniform(-0.5, 0.5)
    # d1 = random.uniform(-0.05*100, 0.05*100)
    # e1 = random.uniform(-0.05*100, 0.05*100)
    # count = 0
    # media_error = 0
    # for xi in x:
    #     count = count + 1
    #     err = f1(xi) - f2(xi, a1, b1, c1, d1, e1)
    #     media_error = media_error + np.absolute(err)
    # media_error=media_error/count
    #
    # if np.absolute(err1) > np.absolute(media_error):
    #     err1 = media_error
    #     a = a1
    #     b = b1
    #     c = c1
    #     d = d1
    #     e = e1



 # Plot the data
# plt.plot(x, f1(x), 'b', label='Funcao original')
# plt.plot(x, f2(x, v[0], v[1], v[2], v[3], v[4]), 'r', label='Funcao otimizada')
# Add a legend
# plt.legend()
# plt.title('Iteracao: '+ str(i) + '  Erro medio: '+ str(err1))

# Show the plot
plt.show()
print(v, direcoes, step)
print("\nGrau = " + str(ngrau))

