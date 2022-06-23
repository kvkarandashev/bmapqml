"""
Simple sklearn style class for kernel ridge regression (KRR)
using the BIGMAPQML library.
"""

class KRR():
    
    def __init__(self, kernel_type, scale_features=True, scale_labels=True):
        
        """
        User must provide a kernelfunction
        kernel_type: string
        """
        self.kernel_type    = kernel_type
        self.scale_features = scale_features
        self.scale_labels   = scale_labels


    def X_MinMaxScaler(self, X):   

        """
        MinMaxScaler for the features.
        X: numpy array
        """
        import numpy as np

        a = 0
        b = 1
        X_scaled = a + ((X - self.X_minval) * (b - a) / (self.X_maxval - self.X_minval))
        return X_scaled

    def y_MinMaxScaler(self, y):
            
        """
        MinMaxScaler for the labels.
        y: numpy array
        """
        import numpy as np
        y = y.reshape(-1, 1)
        a = 0
        b = 1
        y_scaled = a + ((y - self.y_minval) * (b - a) / (self.y_maxval - self.y_minval))
        return y_scaled.flatten()

    def y_MinMaxScaler_inverse(self, y):
            
        """
        Inverse MinMaxScaler for the labels.
        y: numpy array
        """
        import numpy as np
        y = y.reshape(-1, 1)
        a = 0
        b = 1
        y_inverse = (y - a) * ((self.y_maxval - self.y_minval) / (b - a)) + self.y_minval
        return y_inverse.flatten()

    def fit(self, X_train, y_train):

        """
        Fit kernel model to the training data and perform a hyperparameter search.
        X_train: numpy array of representations of the training data
        y_train: numpy array of labels of the training data
        """
        import numpy as np
        from bmapqml.kernels import laplacian_sym_kernel_matrix, gaussian_sym_kernel_matrix
        from bmapqml.hyperparameter_optimization import stochastic_gradient_descend_hyperparam_optimization    
        from bmapqml.linear_algebra import scipy_cho_solve

        if self.scale_features:
            self.X_maxval  = np.max(X_train)
            self.X_minval  = np.min(X_train)
            X_train = self.X_MinMaxScaler(X_train)

        if self.scale_features:
            self.y_maxval  = np.max(y_train)
            self.y_minval  = np.min(y_train)
            y_train = self.y_MinMaxScaler(y_train)



        if self.kernel_type == "gaussian":
            self.kernel_func = gaussian_sym_kernel_matrix
        if self.kernel_type == "laplacian":
            self.kernel_func = laplacian_sym_kernel_matrix
        else:
            raise ValueError("Kernel type not supported")


        if len(X_train) < 10000:
            """
            these settings are feasible for small trainingset sizes
            """
            optimized_hyperparams=stochastic_gradient_descend_hyperparam_optimization(X_train, y_train, init_param_guess=np.array([1.0, 100.0]), max_stagnating_iterations=8,
                                        randomized_iterator_kwargs={"default_step_magnitude" : 0.05}, sym_kernel_func=self.kernel_func, additional_BFGS_iters=8)
        else:
            """
            these settings are feasible for large trainingset sizes
            """
            optimized_hyperparams=stochastic_gradient_descend_hyperparam_optimization(X_train, y_train,
                                init_param_guess=np.array([1.0, 100.0]), max_stagnating_iterations=8,
                                randomized_iterator_kwargs={"default_step_magnitude" : 0.05}, sym_kernel_func=self.kernel_func)


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

        X_test: numpy array
        """

        import numpy as np
        from bmapqml.kernels import laplacian_kernel_matrix, gaussian_kernel_matrix

        if self.kernel_type == "gaussian":
            self.kernel_func = gaussian_kernel_matrix
        if self.kernel_type == "laplacian":
            self.kernel_func = laplacian_kernel_matrix
        else:
            raise ValueError("Kernel type not supported")


        if self.scale_features:
            X_test = self.X_MinMaxScaler(X_test)



        K_test  =   self.kernel_func(X_test, self.X_train, self.sigmas)
        y_pred  =   np.dot(K_test, self.alphas)

        if self.scale_features:
            y_pred = self.y_MinMaxScaler_inverse(y_pred)

        return y_pred

    def save(self, filename):

        from bmapqml.utils import dump2pkl

        """
        Save the trained model to a file. 
        filename: string
        """

        dump2pkl(self, filename)