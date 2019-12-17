#!/usr/bin/env python

# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from scipy import optimize

# Read from the command line, the function degree
ap = argparse.ArgumentParser()
ap.add_argument("-ngrau", "--numero_grau", help="Function Degree", type=int, required=True)
ap.add_argument("-maxit", "--max_iterations", help="Maximum number of iterations", type=int, required=False,
                default=500)
args = vars(ap.parse_args())

ngrau = args['numero_grau']
if ngrau < 0:
    ngrau = 0

# Prepare the data
x = np.linspace(-1, 1, 100)
iteration_number = 0


def original_function(xx):
    """
    Computes the values of ys for the input xx, using the original function
    @param xx: a list of x values
    @return: a list of y values, same size as xx
    """
    return xx * np.cos(xx)


def polynomial_function(xx, params):
    """
    Computes the values of y, for the input values xx, using a polynomial with parameters vv
    @param xx: a list of x values
    @param params: a list of the parameters of this polynomial. The lenght of the list defines the polynomial degree.
    @return: a list of y values, same size as xx
    """
    yy = xx * 0  # initialize yy
    for idx, param in enumerate(params):
        yy += param * (xx ** idx)
    return yy


def objective_function(params):
    """
    Computes the error between the original_function and the polynomial function
    @param params:
    @return: the error for each xx
    """
    global iteration_number, x, ngrau

    yy_original = original_function(x)
    yy_polynomial = polynomial_function(x, params)
    residuals = yy_original - yy_polynomial

    # Compute total error
    total_error = np.sum(np.abs(residuals))

    plt.clf()
    plt.plot(x, original_function(x), 'b', label='Funcao original')  # draw original function
    plt.plot(x, yy_polynomial, 'r', label='Funcao otimizada')  # draw new polynomial
    plt.legend()  # add a legend
    plt.title(
        "Funcao de grau " + str(ngrau) + " Iteracao: " + str(iteration_number) + "\n  Erro Total: " + str(total_error))
    plt.pause(0.02)

    iteration_number += 1
    return residuals


def main():
    params_initial = np.random.uniform(low=-2, high=2, size=ngrau+1)  # initialize params

    # Run optimization
    params_optimized, _, infodict, msg, ler = optimize.leastsq(objective_function, params_initial, full_output=True, xtol=10e-2)

    print("Finished optimization. Details: " + str(infodict))
    print("msg: " + str(msg))
    print("ler: " + str(ler))

    plt.show()  # show the plot and wait for someone to close it


if __name__ == "__main__":
    main()
