import optimisation_solver as ops
import functions as fns

def main():

    # -----------------------------
    # Test Implementation
    # -----------------------------

    '''
        Example Usage:

        solver = ops.OptimisationSolver(fns.rosenbrock, num_vars=2)
        solver.gradient_descent(initial_guess=[1.4, 1.4], learning_rate=0.001)

    '''

    solver = ops.OptimisationSolver(fns.rosenbrock, num_vars=2)
    solver.gradient_descent(initial_guess=[1.4, 1.4], learning_rate=0.001)
    solver.armijo_method(initial_guess=[1.4, 1.4], learning_rate=1.0)
    solver.momentum_method(initial_guess=[0.0, 0.0])

    # -----------------------------
    # Task 3 Implementation
    # -----------------------------

    # -----------------------------
    # Task 4 Implementation
    # -----------------------------

    # -----------------------------
    # Task 5 Implementation
    # -----------------------------
    

if __name__ == "__main__":
    main()

