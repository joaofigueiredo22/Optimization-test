#!/usr/bin/env python


# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np
import random

# Prepare the data
x = np.linspace(-10, 10, 100)

# First guess
e = 0
d = 1
c = 3
b = 2
a = 4


# First function
def f1(xx):
    return xx * np.sin(xx) + np.cos(xx)


# Second function
def f2(xx, ax, bx, cx, dx, ex):
    return ex ** 2 * xx ** 4 + dx ** 2 * xx ** 3 + cx ** 2 * xx ** 2 + bx ** 2 * xx + ax ** 2


# Function to help adding all the values of one array

def somar_elementos(lista):
  soma = 0
  for numero in lista:
    soma += np.absolute(numero)
  return soma


# First guess error

err1 = f1(x) - f2(x, a, b, c, d, e)
err1 = somar_elementos(err1)/len(err1)

plt.plot(x, f1(x), 'b', label='Funcao original')
plt.plot(x, f2(x, a, b, c, d, e), 'r', label='Funcao otimizada')
# Add a legend
plt.legend()
plt.title('Iteracao: 1' + '  Erro medio: '+ str(err1))
# Main loop
for i in range(500):
    a1 = random.uniform(-1, 1)
    b1 = random.uniform(-1, 1)
    c1 = random.uniform(-0.5, 0.5)
    d1 = random.uniform(-0.05, 0.05)
    e1 = random.uniform(-0.05, 0.05)
    count = 0
    media_error = 0
    for xi in x:
        count = count + 1
        err = f1(xi) - f2(xi, a1, b1, c1, d1, e1)
        media_error = media_error + np.absolute(err)
    media_error=media_error/count

    if np.absolute(err1) > np.absolute(media_error):
        err1 = media_error
        a = a1
        b = b1
        c = c1
        d = d1
        e = e1


    # Plot the data
    plt.plot(x, f1(x), 'b', label='Funcao original')
    plt.plot(x, f2(x, a, b, c, d, e), 'r', label='Funcao otimizada')
    # Add a legend
    plt.legend()

    plt.title('Iteracao: '+ str(i) + '  Erro medio: '+ str(err1))
    plt.pause(0.01)
    plt.clf()

 # Plot the data
plt.plot(x, f1(x), 'b', label='Funcao original')
plt.plot(x, f2(x, a, b, c, d, e), 'r', label='Funcao otimizada')
# Add a legend
plt.legend()
plt.title('Iteracao: '+ str(i) + '  Erro medio: '+ str(err1))

# Show the plot
plt.show()






