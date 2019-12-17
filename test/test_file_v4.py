#!/usr/bin/env python


# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from scipy import optimize

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
def f3(vv):
    f=f1(x)
    for numgrau in range(ngrau + 1):
        f = f - vv[numgrau] * x ** numgrau
    plt.plot(x, f2(x, v, ngrau), 'r', label='Funcao otimizada')
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

plt.plot(x, f1(x), 'b', label='Funcao original')

# Optimization
v_optimized, ier = optimize.leastsq(f3, v)
v = v_optimized * 1
erro = calcular_erro(f3(v))


# Add a legend
plt.legend()
plt.title("Funcao de grau " + str(ngrau) + "   Iteracao: " + str(0) + "\n  Erro Total: " + str(erro))

# Show the plot
plt.show()
# print(v, direcoes, step)
# print("\nGrau = " + str(ngrau))
print(v, erro)
