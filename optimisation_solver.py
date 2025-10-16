import numpy as np
import sympy as sy
from typing import Callable

'''

The OptimisationSolver class provides methods for solving optimization problems using various algorithms.
It supports gradient descent, Armijo method, and momentum-based optimization.

The library "sympy" is used for symbolic differentiation to compute gradients of the objective function. This 
was done in order to simplify the implementation. Instead of requiring the user to provide a gradient function,
the class can compute it automatically from the objective function.

However, this approach does not always yeild the best result or performance for all functions, especially 
for those that are more complex.

For more information on "sympy", see: https://www.sympy.org/en/index.html

'''



class OptimisationSolver:
    def __init__(self, objective_function: Callable, num_vars: int):
        self.objective_function = objective_function
        self.num_vars = num_vars          
        self.calc_symbolic_gradient()      

    def calc_symbolic_gradient(self):
        """
        Calculate the symbolic gradient of the objective function.

        This method uses sympy to calculate the symbolic gradient of the objective function.
        It first creates symbolic variables for the input variables, then calculates the symbolic gradient
        by differentiating the objective function with respect to each variable.

        :return: None
        """

        x_syms = sy.symbols(f'x0:{self.num_vars}') 
        f_sym = self.objective_function(*x_syms) 
        gradient_syms = [sy.diff(f_sym, var) for var in x_syms]
        self.gradient_function = sy.lambdify(x_syms, gradient_syms, 'numpy') # Create a callable function for the gradient (numpy library compatible)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the objective function at a given point x.

        Parameters
        ----------
        x : np.ndarray
            The point at which to calculate the gradient.

        Returns
        -------
        np.ndarray
            The gradient of the objective function at x.

        """
        
        x = np.array(x, dtype=float)
        return np.array(self.gradient_function(*x), dtype=float)

    def gradient_descent(self, initial_guess: np.ndarray, learning_rate: float = 0.001, max_iter: int = 10000, tol: float = 1e-6) -> np.ndarray:
        """
        Perform gradient descent to find the minimum of the objective function.

        Parameters
        ----------
        initial_guess : np.ndarray
            The initial guess for the minimum of the objective function.
        learning_rate : float, optional
            The learning rate for the gradient descent algorithm. Defaults to 0.001.
        max_iter : int, optional
            The maximum number of iterations for the gradient descent algorithm. Defaults to 10000.
        tol : float, optional
            The tolerance for the convergence of the gradient descent algorithm. Defaults to 1e-6.

        Returns
        -------
        np.ndarray
            The minimum of the objective function found by the gradient descent algorithm.
        """

        x_k = np.array(initial_guess, dtype=float)

        for _ in range(max_iter):
            grad = self.gradient(x_k) # Computes the gradient at the current point (does this on each iteration)
            x_next = x_k - learning_rate * grad
            if np.linalg.norm(x_next - x_k) < tol: # Checks for convergence at each iteration
                break
            x_k = x_next # Update the current point with the next point


        print("Gradient Descent converged in", _ , "iterations.")
        print("Minimum value at:", x_k)

    
    def armijo_method(self, initial_guess, max_iter=1000, tol=1e-6):
        pass
    
    def momentum_method(self, initial_guess, learning_rate=0.01, momentum=0.9, max_iter=1000, tol=1e-6):
        pass 
    
