import numpy as np
import sympy as sy
from typing import Callable
import matplotlib.pyplot as plt

'''
The OptimisationSolver class provides methods for solving optimization problems using various algorithms.
It supports gradient descent, Armijo method, and momentum-based optimization.

The library "sympy" is used for symbolic differentiation to compute gradients of the objective function. This 
simplifies implementation by not requiring the user to provide a gradient function. Now, the user only needs to supply the objective 
function and the number of variables (making testing for Task 3,4,5 easier).

For more information on "sympy", see: https://www.sympy.org/en/index.html


Other Sources:
    - 


'''

class OptimisationSolver:
    def __init__(self, objective_function: Callable, num_vars: int):
        self.objective_function = objective_function
        self.num_vars = num_vars          
        self.calc_symbolic_gradient()      

    def calc_symbolic_gradient(self):
        """
        Calculate the symbolic gradient of the objective function using sympy.

        This method is used to simplify the implementation of the optimization algorithms by not requiring the user to provide a gradient function.
        The gradient is calculated using symbolic differentiation and then converted to a numpy-compatible function using sympy's lambdify method.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        x_syms = sy.symbols(f'x0:{self.num_vars}') 
        f_sym = self.objective_function(*x_syms) 
        gradient_syms = [sy.diff(f_sym, var) for var in x_syms]
        self.gradient_function = sy.lambdify(x_syms, gradient_syms, 'numpy') # Convert to numpy-compatible function ( otherwise sympy objects are returned which are not compatible with numpy operations )

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the objective function at a given point x.

        Parameters
        ----------
        x : np.ndarray
            The point at which to compute the gradient.

        Returns
        -------
        gradient : np.ndarray
            The gradient of the objective function at x. This is used in all optimization methods at every iteration.
        """
        
        x = np.array(x, dtype=float)
        return np.array(self.gradient_function(*x), dtype=float)

    def gradient_descent(self, initial_guess: np.ndarray, learning_rate: float = 0.001, 
                         max_iterations: int = 10000, tol: float = 1e-6, return_history=False, return_path=False): # User cab choose to return history of objective function values and/or path taken during optimization
        

        """
        Gradient Descent optimization algorithm.

        Parameters
        ----------
        initial_guess : np.ndarray
            Initial guess for the optimization algorithm
        learning_rate : float
            Learning rate for the gradient descent algorithm
        max_iterations : int
            Maximum number of iterations
        tol : float
            Tolerance for convergence
        return_history : bool
            Whether to return the history of the objective function values
        return_path : bool
            Whether to return the path taken during optimization

        Returns
        -------
        x_k : np.ndarray
            Minimum value obtained
        history : list
            History of the objective function values
        path : np.ndarray
            Path taken during optimization
        """


    
        x_k: np.ndarray = np.array(initial_guess, dtype=float)
        history: list = [self.objective_function(*x_k)]
        path = [x_k.copy()] # coppied to avoid reference issues (error occurs where all elements in path point to same memory location)

        for i in range(max_iterations):
           
            gradient: np.ndarray = self.gradient(x_k)
            x_next: np.ndarray = x_k - learning_rate * gradient
            history.append(self.objective_function(*x_next))
           
            path.append(x_next.copy()) 


            if np.linalg.norm(x_next - x_k) < tol:
                break
            x_k = x_next

        print("\n----------------------------------------------")
        print(f"Gradient Descent converged in {i} iterations.")
        print("Minimum value at:", x_k)
        print("----------------------------------------------\n")

        if return_history and return_path:
            return x_k, history, np.array(path)
        elif return_history:
            return x_k, history
        elif return_path:
            return x_k, np.array(path)
        else:
            return x_k
    def armijo_method(self, initial_guess, max_iterations=1000, learning_rate: float = 0.001, 
                      tol=1e-6, beta=0.5, sigma=1e-4, return_history=False, return_path=False): # User can choose to return history of objective function values and/or path taken during optimization

        """
        Armijo Method optimization algorithm.

        Parameters
        ----------
        initial_guess : np.ndarray
            Initial guess for the optimization algorithm
        max_iterations : int
            Maximum number of iterations
        learning_rate : float
            Learning rate for the gradient descent algorithm
        tol : float
            Tolerance for convergence
        beta : float
            Parameter for the step size reduction in the Armijo method
        sigma : float
            Parameter for the sufficient decrease condition in the Armijo method
        return_history : bool
            Whether to return the history of the objective function values
        return_path : bool
            Whether to return the path taken during optimization

        Returns
        -------
        x_k : np.ndarray
            Minimum value obtained
        history : list
            History of the objective function values
        path : np.ndarray
            Path taken during optimization
        """

        x_k = np.array(initial_guess, dtype=float)
        history = [self.objective_function(*x_k)] # used to store objective function values for plotting convergence later (see graph_convergence method and plot_optimization_path_2d method)
        path = [x_k.copy()]

        for i in range(max_iterations):
            gradient_k = self.gradient(x_k)
            grad_norm_sqrd = np.dot(gradient_k, gradient_k) # Squared norm of the gradient (see formula)
            n_k = 0
            eta = learning_rate * (beta ** n_k) # Initial step size for this iteration

            while True:

                x_test = x_k - eta * gradient_k

                left = self.objective_function(*x_test)
                right = self.objective_function(*x_k) - eta * sigma * grad_norm_sqrd # "left" and "right" as in the Armijo formula

                path.append(x_test.copy())

                if left <= right or eta < 1e-12:
                    break

                n_k += 1
                eta = learning_rate * (beta ** n_k)

            x_next = x_k - eta * gradient_k
            history.append(self.objective_function(*x_next))

            if np.linalg.norm(x_next - x_k) < tol:
                print("\n----------------------------------------------")
                print(f"Armijo Method converged in {i} iterations.")
                print("Minimum value at:", x_next)
                print("----------------------------------------------\n")
                
                if return_history and return_path:
                    return x_k, history, np.array(path)
                elif return_history:    
                    return x_k, history
                elif return_path:
                    return x_k, np.array(path)
                else:
                    return x_k
                
            x_k = x_next

        print("\n----------------------------------------------")
        print("Armijo Method reached maximum iterations.")
        print("Minimum value at:", x_k)
        print("----------------------------------------------\n")

        if return_history and return_path:
            return x_k, history, np.array(path)
        elif return_history:    
            return x_k, history
        elif return_path:
            return x_k, np.array(path)
        else:
            return x_k


    def momentum_method(self, initial_guess, learning_rate=1e-7, gamma=0.4, 
                    max_iterations=1000, tol=1e-6, return_history=False, return_path=False): 
        """
        Momentum-based optimization algorithm.

        Parameters
        ----------
        initial_guess : np.ndarray
            Initial guess for the optimization algorithm
        learning_rate : float
            Learning rate for the momentum-based optimization algorithm
        gamma : float
            Momentum parameter
        max_iterations : int
            Maximum number of iterations
        tol : float
            Tolerance for convergence
        return_history : bool
            Whether to return the history of the objective function values
        return_path : bool
            Whether to return the path taken during optimization

        Returns
        -------
        x_k : np.ndarray
            Minimum value obtained
        history : list
            History of the objective function values (if return_history=True)
        path : np.ndarray
            Path taken during optimization (if return_path=True)
        """

        x_k = np.array(initial_guess, dtype=float)
        x_k_prev = np.array(initial_guess, dtype=float)

        path = [x_k.copy()]  
        history = [self.objective_function(*x_k)]  

        for k in range(max_iterations):
            gradient_k = self.gradient(x_k)
            gradient_k = np.clip(gradient_k, -100, 100)  # Prevent gradient explosion at edges (helps with complex functions like Rosenbrock which was used iin testing)

            x_k_next = x_k - learning_rate * gradient_k + gamma * (x_k - x_k_prev)


            path.append(x_k_next.copy())
            history.append(self.objective_function(*x_k_next))

            if np.linalg.norm(x_k_next - x_k) > np.linalg.norm(x_k - x_k_prev):
                learning_rate *= 0.8 # Decrease learning rate slightly if not converging
            else:
                learning_rate *= 1.001 # Slightly increase learning rate if converging


            if np.linalg.norm(x_k_next - x_k) < tol:
                print("\n----------------------------------------------")
                print(f"Momentum Method converged in {k} iterations.")
                print("Minimum value at:", x_k_next)
                print("----------------------------------------------\n")


                if return_history and return_path:
                    return x_k_next, history, np.array(path)
                elif return_history:
                    return x_k_next, history
                elif return_path:
                    return x_k_next, np.array(path)
                else:
                    return x_k_next

            x_k_prev = x_k
            x_k = x_k_next

        print("\n----------------------------------------------")
        print("Momentum Method reached maximum iterations.")
        print("Approximate minimum at:", x_k)
        print("----------------------------------------------\n")

        if return_history and return_path:
            return x_k, history, np.array(path)
        elif return_history:
            return x_k, history
        elif return_path:
            return x_k, np.array(path)
        else:
            return x_k

        
    def graph_convergence(self, method: str, initial_guess: np.ndarray, **kwargs):
        
        """
        Graph the convergence of the optimization algorithms.

        Parameters
        ----------
        method : str
            The name of the optimization algorithm to graph.
        initial_guess : np.ndarray
            The initial guess for the optimization algorithm.
        **kwargs
            Additional keyword arguments to pass to the optimization algorithm.

        Raises
        ------
        ValueError
            If the method is unknown.
        """


        if method == 'gradient_descent':
            _, history = self.gradient_descent(initial_guess, return_history=True, **kwargs)
        elif method == 'armijo_method':
            _, history = self.armijo_method(initial_guess, return_history=True, **kwargs)
        elif method == 'momentum_method':
            _, history = self.momentum_method(initial_guess, return_history=True, **kwargs)
        else:
            raise ValueError("Unknown method")

        plt.plot(history)
        plt.xlabel('Iterations')
        plt.ylabel('Objective Function Value')
        plt.title(f'{method} Convergence')
        plt.grid(True)
        plt.show()

    def plot_optimization_path_2d(
        self,
        method: str,
        initial_guess,
        learning_rate=0.001,
        gamma=0.4,
        beta=0.5,
        sigma=1e-4,
        max_iterations=1000,
        tol=1e-6
    ):
        """
        Plot the optimization path for 2D functions using the chosen method.

        Parameters
        ----------
        method : str
            'gradient_descent', 'momentum_method', or 'armijo_method'
        initial_guess : list or np.ndarray
            Starting point for optimization
        Other params :
            Parameters passed to each method
        """

        if self.num_vars != 2:
            raise ValueError("This visualization method only works for 2D functions.")

        if method == "gradient_descent":
            _, path = self.gradient_descent(
                initial_guess,
                learning_rate=learning_rate,
                max_iterations=max_iterations,
                tol=tol,
                return_path=True
            )

        elif method == "momentum_method":
            _, path = self.momentum_method(
                initial_guess,
                learning_rate=learning_rate,
                gamma=gamma,
                max_iterations=max_iterations,
                tol=tol,
                return_path=True
            )

        elif method == "armijo_method":
            _, path = self.armijo_method(
                initial_guess,
                learning_rate=learning_rate,
                beta=beta,
                sigma=sigma,
                max_iterations=max_iterations,
                tol=tol,
                return_path=True
            )

        else:
            raise ValueError("Unknown method. Use 'gradient_descent', 'momentum_method', or 'armijo_method'.")

        path = np.array(path)

        # AI GENERATED ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

        x_vals = np.linspace(path[:, 0].min() - 1, path[:, 0].max() + 1, 200)
        y_vals = np.linspace(path[:, 1].min() - 1, path[:, 1].max() + 1, 200)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.vectorize(lambda x, y: self.objective_function(x, y))(X, Y)

        # AI GENERATED ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


        plt.figure(figsize=(8, 6))
        plt.contour(X, Y, Z, levels=40, cmap="viridis") # Source : https://matplotlib.org/stable/gallery/images_contours_and_fields/contourf_demo.html

    
        # AI GENERATED ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        plt.plot(path[:, 0], path[:, 1], "r.-", label=f"{method.replace('_', ' ').title()} Path")
        # AI GENERATED ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        
        
        plt.scatter(path[-1, 0], path[-1, 1], color="red", marker="x", s=100, label="Minimum Point")
        plt.xlabel("x_0")
        plt.ylabel("x_1")
        plt.title(f"{method.replace('_', ' ').title()} Optimization Path (2D)")
        plt.legend()
        plt.grid(True)
        plt.show()
