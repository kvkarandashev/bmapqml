"""
Simple sklearn style class for kernel ridge regression (KRR)
using the BIGMAPQML library.
"""

class KRR():
    
    def __init__(self, kernel_type):
        
        """
        User must provide a kernelfunction
        """
        self.kernel_type = kernel_type



    def fit(self, X_train, y_train):
        
        """
        Fit kernel model to the training data and perform a hyperparameter search.
        """

        import numpy as np
        from bmapqml.kernels import laplacian_sym_kernel_matrix, gaussian_sym_kernel_matrix
        from bmapqml.hyperparameter_optimization import stochastic_gradient_descend_hyperparam_optimization    
        from bmapqml.linear_algebra import scipy_cho_solve

        if self.kernel_type == "gaussian":
            self.kernel_func = gaussian_sym_kernel_matrix
        if self.kernel_type == "laplacian":
            self.kernel_func = laplacian_sym_kernel_matrix
        else:
            raise ValueError("Kernel type not supported")

        optimized_hyperparams=stochastic_gradient_descend_hyperparam_optimization(X_train, y_train, init_param_guess=np.array([1.0, 100.0]), max_stagnating_iterations=8,
                                    randomized_iterator_kwargs={"default_step_magnitude" : 0.05}, sym_kernel_func=self.kernel_func, additional_BFGS_iters=8)

        self.sigmas=optimized_hyperparams["sigmas"]
        self.lambda_val=optimized_hyperparams["lambda_val"]

        print("Finalized parameters:", self.sigmas)
        print("Finalized lambda:", self.lambda_val)
        K_train=self.kernel_func(X_train, self.sigmas)
        K_train[np.diag_indices_from(K_train)]+=self.lambda_val
        alphas=scipy_cho_solve(K_train, y_train)
        del(K_train)
   
        self.X_train  = X_train
        self.alphas = alphas


    def predict(self, X_test):

        """
        Predict the labels of the test data X_test 
        using the trained model.
        """

        import numpy as np
        from bmapqml.kernels import laplacian_kernel_matrix, gaussian_kernel_matrix

        if self.kernel_type == "gaussian":
            self.kernel_func = gaussian_kernel_matrix
        if self.kernel_type == "laplacian":
            self.kernel_func = laplacian_kernel_matrix
        else:
            raise ValueError("Kernel type not supported")


        K_test  =   self.kernel_func(X_test, self.X_train, self.sigmas)
        y_pred  =   np.dot(K_test, self.alphas)

        return y_pred