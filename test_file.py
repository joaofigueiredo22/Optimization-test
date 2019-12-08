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
    return np.cos(xx)
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
v3 = v * 1

plt.plot(x, f1(x), 'b', label='Funcao original')
plt.plot(x, f2(x, a, b, c, d, e), 'r', label='Funcao otimizada')
# Add a legend
plt.legend()
plt.title('Iteracao: 1' + '  Erro medio: '+ str(err1))
# Main loop
for i in range(2000):

    ind=0

    for r in v:
        # print(r)
        if r==0.000000000000:
            r=0.25
        r1=float(r)+np.absolute(float(r))/100
        r2=float(r)-np.absolute(float(r))/100
        v1 = v * 1
        v1[ind] = r1 * 1
        v2 = v * 1
        v2[ind] = r2 * 1
        erro = calcular_erro(f1(x) - f2(x, v[0], v[1], v[2], v[3], v[4]))
        erro1 = calcular_erro(f1(x)-f2(x,v1[0],v1[1],v1[2],v1[3],v1[4]))
        erro2 = calcular_erro(f1(x)-f2(x,v2[0],v2[1],v2[2],v2[3],v2[4]))

        inc=0
        while (erro1 > erro) and (erro2 > erro) and (inc < 100):
            r1 = float(r1) + np.absolute(float(r1)) * 2
            r2 = float(r2) - np.absolute(float(r2)) * 2
            v1[ind] = r1 * 1
            v2[ind] = r2 * 1
            erro1 = calcular_erro(f1(x) - f2(x, v1[0], v1[1], v1[2], v1[3], v1[4]))
            erro2 = calcular_erro(f1(x) - f2(x, v2[0], v2[1], v2[2], v2[3], v2[4]))
            inc += 1

        if erro2 < erro:

            v3[ind] = r2 * 1

        elif erro1 < erro:

            v3[ind] = r1 * 1
        ind += 1
    print (v,v3)
    v = v3 * 1
    print(v)



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
    plt.clf()
    # Plot the data
    plt.plot(x, f1(x), 'b', label='Funcao original')
    plt.plot(x, f2(x, v[0], v[1], v[2], v[3], v[4]), 'r', label='Funcao otimizada')
    # Add a legend
    plt.legend()
    erro = calcular_erro(f1(x) - f2(x, v[0], v[1], v[2], v[3], v[4]))
    plt.title('Iteracao: '+ str(i) + '  Erro medio: '+ str(erro))
    plt.pause(0.01)


 # Plot the data
# plt.plot(x, f1(x), 'b', label='Funcao original')
# plt.plot(x, f2(x, v[0], v[1], v[2], v[3], v[4]), 'r', label='Funcao otimizada')
# Add a legend
# plt.legend()
# plt.title('Iteracao: '+ str(i) + '  Erro medio: '+ str(err1))

# Show the plot
plt.show()






