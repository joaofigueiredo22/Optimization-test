#!/usr/bin/env python


# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np
import random

# Prepare the data
x = np.linspace(-1, 1, 100)

# First guess
e = 1
d = 1
c = 3
b = 2
a = 4


# First function
def f1(xx):
    return np.cos(xx) * xx + np.sin(xx)
    # return xx

# Second function
def f2(xx, ax, bx, cx, dx, ex):
    return ex * xx ** 4 + dx * xx ** 3 + cx * xx ** 2 + bx * xx + ax


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

err1 = f1(x) - f2(x, a, b, c, d, e)
err1 = calcular_erro(err1)
v=[a , b, c, d, e]

plt.plot(x, f1(x), 'b', label='Funcao original')
plt.plot(x, f2(x, a, b, c, d, e), 'r', label='Funcao otimizada')
# Add a legend
plt.legend()
plt.title('Iteracao: 1' + '  Erro medio: '+ str(err1))

dir = 0.0001
step = 0.1

# Main loop
for i in range(500):

    ind = 0

    # direcoes=np.zeros((1, 5), np.float32)
    direcoes=[0,0,0,0,0]

    for r in v:
        r1 = r+dir
        r2 = r-dir

        v1 = v * 1
        v1[ind] = r1 * 1
        v2 = v * 1
        v2[ind] = r2 * 1
        erro1 = calcular_erro(f1(x)-f2(x,v1[0],v1[1],v1[2],v1[3],v1[4]))
        erro2 = calcular_erro(f1(x)-f2(x,v2[0],v2[1],v2[2],v2[3],v2[4]))

        if erro2 < erro1:

            direcoes[ind] = -1

        elif erro1 < erro2:

            direcoes[ind] = 1

        ind += 1

    newv = v + np.dot(step,direcoes)

    erro = calcular_erro(f1(x) - f2(x, v[0], v[1], v[2], v[3], v[4]))
    newerro = calcular_erro(f1(x) - f2(x, newv[0], newv[1], newv[2], newv[3], newv[4]))

    if newerro < erro:
        v = newv * 1
        plt.clf()
        # Plot the data
        plt.plot(x, f1(x), 'b', label='Funcao original')
        plt.plot(x, f2(x, v[0], v[1], v[2], v[3], v[4]), 'r', label='Funcao otimizada')
        # Add a legend
        plt.legend()
        plt.title('Iteracao: ' + str(i) + '  Erro medio: ' + str(erro))
        plt.pause(0.00001)
    else:
        step=step/2

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
print(v)

