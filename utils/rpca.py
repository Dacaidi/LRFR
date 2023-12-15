import numpy as np

def rpca(X, regularization_parameter, augmented_lagrangian_parameter, error_tolerance, max_iterations):
    """Compute RPCA"""

    number_of_rows = X.shape[0]
    number_of_columns = X.shape[1]

    L = np.zeros(number_of_rows, number_of_columns)
    S = np.zeros(number_of_rows, number_of_columns)
    Y = np.zeros(number_of_rows, number_of_columns)

    for i in range(1, max_iterations):
        L = solve_L(1 / augmented_lagrangian_parameter, X - S + (1 / augmented_lagrangian_parameter) * Y)
        S = solve_D((regularization_parameter / augmented_lagrangian_parameter),
                    X - L + (1 / augmented_lagrangian_parameter) * Y)
        Z = X - L - S
        Y = Y + augmented_lagrangian_parameter * Z;

if __name__== "__main__" :
    X = np.random.rand(3, 3)
    rpca(X)