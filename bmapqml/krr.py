"""
Simple sklearn style class for kernel ridge regression (KRR)
using the BIGMAPQML library.
"""
from .utils import OptionUnavailableError
import numpy as np

def arr_scale(arr, minval, maxval, lbound=0, ubound=1):
    return lbound+(arr-minval)*(ubound-lbound)/(maxval-minval)

def arr_rescale(scaled_arr, minval, maxval, lbound=0, ubound=1):
    return minval+(maxval-minval)*(scaled_arr-lbound)/(ubound-lbound)


class KRR():
    
    def __init__(self, kernel_type="Gaussian", scale_features=False, scale_labels=False, kernel_function=None, sym_kernel_function=None,
                    hyperparam_opt_kwargs={"max_stagnating_iterations" : 8, "randomized_iterator_kwargs" : {"default_step_magnitude" : 0.05}},
                    sigmas=None, lambda_val=None):
        
        """
        User must provide a kernelfunction
        kernel_type: string
        """
        self.kernel_type    = kernel_type
        self.scale_features = scale_features
        self.scale_labels   = scale_labels

        self.hyperparam_opt_kwargs=hyperparam_opt_kwargs

        self.sigmas=sigmas
        self.lambda_val=lambda_val

        self.kernel_function=kernel_function
        self.sym_kernel_function=sym_kernel_function

        if self.kernel_function is None:
            from .kernels import common_kernels, common_sym_kernels
            if self.kernel_type in common_kernels:
                self.kernel_function=common_kernels[self.kernel_type]
                self.sym_kernel_function=common_sym_kernels[self.kernel_type]
            else:
                raise OptionUnavailableError

        if self.sym_kernel_function is None:
            from .kernels import symmetrized_kernel_matrix
            self.sym_kernel_function=symmetrized_kernel_matrix(self.kernel_function)

    def X_MinMaxScaler(self, X):   

        """
        MinMaxScaler for the features.
        X: numpy array
        """

        return arr_scale(X, self.X_minval, self.X_maxval)

    def y_MinMaxScaler(self, y):
            
        """
        MinMaxScaler for the labels.
        y: numpy array
        """
        return arr_scale(y, self.y_minval, self.y_maxval)

    def y_MinMaxScaler_inverse(self, y):
            
        """
        Inverse MinMaxScaler for the labels.
        y: numpy array
        """
        return arr_rescale(y, self.y_minval, self.y_maxval)

    def opt_hyperparameter_guess(self, X_train, y_train):
        if self.sigmas is None:
            sigma_guess=.0
            for i1, vec1 in enumerate(X_train):
                for vec2 in X_train[:i1]:
                    sigma_guess=max(sigma_guess, np.sum(np.abs(vec1-vec2)))
        else:
            sigma_guess=self.sigmas
        if self.lambda_val is None:
            lambda_guess=1.e-6
        else:
            lambda_guess=self.lambda_val
        return np.array([lambda_guess, sigma_guess])

    def optimize_hyperparameters(self, X_train, y_train, init_param_guess=None):
        if init_param_guess is None:
            init_param_guess=self.opt_hyperparameter_guess(X_train, y_train)
        from .hyperparameter_optimization import stochastic_gradient_descend_hyperparam_optimization    
        optimized_hyperparams=stochastic_gradient_descend_hyperparam_optimization(X_train, y_train, init_param_guess=init_param_guess,
                    **self.hyperparam_opt_kwargs, sym_kernel_func=self.sym_kernel_function)
        self.sigmas=optimized_hyperparams["sigmas"]
        self.lambda_val=optimized_hyperparams["lambda_val"]
        print("Optimized parameters:", self.sigmas)
        print("Optimized lambda:", self.lambda_val)

    def fit(self, X_train, y_train, optimize_hyperparameters=False):
        from .linear_algebra import scipy_cho_solve
        """
        Fit kernel model to the training data and perform a hyperparameter search.
        X_train: numpy array of representations of the training data
        y_train: numpy array of labels of the training data
        """
        if self.scale_features:
            self.X_maxval  = np.max(X_train)
            self.X_minval  = np.min(X_train)
            X_train = self.X_MinMaxScaler(X_train)

        if self.scale_features:
            self.y_maxval  = np.max(y_train)
            self.y_minval  = np.min(y_train)
            y_train = self.y_MinMaxScaler(y_train)

        if optimize_hyperparameters or (self.sigmas is None) or (self.lambda_val is None):
            self.optimize_hyperparameters(X_train, y_train)

        K_train=self.sym_kernel_function(X_train, self.sigmas)
        K_train[np.diag_indices_from(K_train)]+=self.lambda_val
        alphas=scipy_cho_solve(K_train, y_train)
        del(K_train)
   
        self.X_train  = X_train
        self.alphas = alphas

    def predict(self, X_test):

        """
        Predict the labels of the test data X_test 
        using the trained model.

        X_test: numpy array
        """

        if self.scale_features:
            X_test = self.X_MinMaxScaler(X_test)

        K_test  =   self.kernel_function(X_test, self.X_train, self.sigmas)
        y_pred  =   np.dot(K_test, self.alphas)

        if self.scale_features:
            y_pred = self.y_MinMaxScaler_inverse(y_pred)

        return y_pred

    def save(self, filename):

        from .utils import dump2pkl

        """
        Save the trained model to a file. 
        filename: string
        """

        dump2pkl(self, filename)
