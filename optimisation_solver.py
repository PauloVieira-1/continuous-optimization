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
            gradient = self.gradient(x_k) # Computes the gradient at the current point (does this on each iteration)
            x_next = x_k - learning_rate * gradient
            if np.linalg.norm(x_next - x_k) < tol: # Checks for convergence at each iteration
                break
            x_k = x_next # Update the current point with the next point

        print("\n---------------------------------------------- \n")
        print("Gradient Descent converged in", _ , "iterations.")
        print("Minimum value at:", x_k)
        print("\n---------------------------------------------- \n")

    
    def armijo_method(self, initial_guess, max_iter=1000, learning_rate: float = 1.0, tol=1e-6, beta=0.5, sigma=1e-4):
        """
        Perform Armijo method to find the minimum of the objective function.

        Parameters
        ----------
        initial_guess : np.ndarray
            The initial guess for the minimum of the objective function.
        max_iter : int, optional
            The maximum number of iterations for the Armijo method algorithm. Defaults to 1000.
        learning_rate : float, optional (represents eta_0 in the algorithm)
            The learning rate for the Armijo method algorithm. Defaults to 1.0.
        tol : float, optional
            The tolerance for the convergence of the Armijo method algorithm. Defaults to 1e-6.
        beta : float, optional
            The reduction factor for the step size in the Armijo method algorithm. Defaults to 0.5.
        sigma : float, optional
            The reduction limit for the step size in the Armijo method algorithm. Defaults to 1e-4.

        Returns
        -------
        np.ndarray
            The minimum of the objective function found by the Armijo method algorithm.
        """
        x_k = np.array(initial_guess, dtype=float)
        
        for _ in range(max_iter):
            gradient_k = self.gradient(x_k) 
            grad_norm_sqrd = np.dot(gradient_k, gradient_k)
            n_k = 0
            eta = learning_rate * (beta ** n_k)  # Initial step size + learning rate variiable name kept the same across functions for consistency (eta0 = )

            while True: # finds the smallest non-negative integer n_k such that the Armijo condition is satisfied 
                x_test = x_k - eta * gradient_k
                left = self.objective_function(*x_test)
                right = self.objective_function(*x_k) - eta * sigma * grad_norm_sqrd

                if left <= right or eta < 1e-12:  # stop reducing step if too small
                    break

                n_k += 1
                eta = learning_rate * (beta ** n_k)


        x_next = x_k - eta * gradient_k

        if np.linalg.norm(x_next - x_k) < tol: # calculates the norm and then checks for convergence
            print("\n---------------------------------------------- \n")
            print(f"Armijo Method converged in {_} iterations.")
            print("Minimum value at:", x_next)
            print("\n---------------------------------------------- \n")
            return x_next

        x_k = x_next

        print("\n---------------------------------------------- \n")
        print("Armijo Method reached maximum iterations. Try increasing max_iter or adjusting learning_rate.")
        print("Minimum value at:", x_k)
        print("\n---------------------------------------------- \n")

        return x_k

        
    def momentum_method(self, initial_guess, learning_rate=1e-7, gamma=0.4, max_iter=1000, tol=1e-6):
        """
        Perform momentum method to find the minimum of the objective function.

        Parameters
        ----------
        initial_guess : np.ndarray
            The initial guess for the minimum of the objective function.
        learning_rate : float, optional
            The learning rate for the momentum method algorithm. Defaults to 1e-7.
        gamma : float, optional
            The momentum coefficient for the momentum method algorithm. Defaults to 0.4.
        max_iter : int, optional
            The maximum number of iterations for the momentum method algorithm. Make sure to set this to a very low value 
            if the function is not converging. 
        tol : float, optional
            The tolerance for the convergence of the momentum method algorithm. Defaults to 1e-6.

        Returns
        -------
        np.ndarray
            The minimum of the objective function found by the momentum method algorithm.
        """


        x_k = np.array(initial_guess, dtype=float)
        x_k_prev = np.array(initial_guess, dtype=float)  # The auxiliary variable x_{k-1} (initialized to the initial guess according to the provided algorithm)

        for k in range(max_iter):
            gradient_k = self.gradient(x_k)
            gradient_k = np.clip(gradient_k, -100, 100) # Clipping was used to avoid overflow issues with some functions. Before, the algorithm would go to infinity.

            x_k_next = x_k - learning_rate * gradient_k + gamma * (x_k - x_k_prev)

            if np.linalg.norm(x_k_next - x_k) > np.linalg.norm(x_k - x_k_prev):
                learning_rate *= 0.8 # Decrease learning rate if getting worse
            else:
                learning_rate *= 1.001 # Slightly increase learning rate if going well

            if np.linalg.norm(x_k_next - x_k) < tol:
                print("\n---------------------------------------------- \n")
                print(f"Momentum Method converged in {k} iterations.")
                print("Minimum value at:", x_k_next)
                print("\n---------------------------------------------- \n")
                return x_k_next

            x_k_prev = x_k
            x_k = x_k_next

        print("\n---------------------------------------------- \n")
        print("Momentum Method reached maximum iterations.")
        print("Approximate minimum at:", x_k)
        print("\n---------------------------------------------- \n")


        return x_k


