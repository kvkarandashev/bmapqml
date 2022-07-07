"""
Simple sklearn style class for kernel ridge regression (KRR)
using the BIGMAPQML library.
"""
from hashlib import new
from .utils import OptionUnavailableError
import numpy as np
from .training_set_optimization import KernelUnstable
from .hyperparameter_optimization import max_Laplace_dist


def arr_scale(arr, minval, maxval, lbound=0.0, ubound=1.0):
    return lbound + (arr - minval) * (ubound - lbound) / (maxval - minval)


def arr_rescale(scaled_arr, minval, maxval, lbound=0.0, ubound=1.0):
    return minval + (maxval - minval) * (scaled_arr - lbound) / (ubound - lbound)


class KRR:
    def __init__(
        self,
        kernel_type="Gaussian",
        scale_features=False,
        scale_labels=False,
        use_label_drift=False,
        optimize_label_drift=False,
        label_drift=None,
        kernel_function=None,
        sym_kernel_function=None,
        hyperparam_opt_kwargs={
            "max_stagnating_iterations": 4,
            "randomized_iterator_kwargs": {"default_step_magnitude": np.log(2.0)},
        },
        sigmas=None,
        lambda_val=None,
        updatable=False,
        init_sigma_guess_func=max_Laplace_dist,
        parallelized_model_evaluation=True,
    ):

        """
        User must provide a kernelfunction
        kernel_type: string
        """
        self.kernel_type = kernel_type
        self.scale_features = scale_features
        self.scale_labels = scale_labels
        self.use_label_drift = use_label_drift
        if self.use_label_drift:
            self.label_drift = label_drift
            self.optimize_label_drift = optimize_label_drift
        else:
            self.optimize_label_drift = False
        self.hyperparam_opt_kwargs = hyperparam_opt_kwargs

        self.sigmas = sigmas
        self.lambda_val = lambda_val

        self.kernel_function = kernel_function
        self.sym_kernel_function = sym_kernel_function
        self.init_sigma_guess_func = init_sigma_guess_func

        self.parallelized_model_evaluation = parallelized_model_evaluation

        self.alphas = None
        self.X_train = None

        if self.kernel_function is None:
            from .kernels import common_kernels, common_sym_kernels

            if self.kernel_type in common_kernels:
                self.kernel_function = common_kernels[self.kernel_type]
                self.sym_kernel_function = common_sym_kernels[self.kernel_type]
            else:
                raise OptionUnavailableError

        if self.sym_kernel_function is None:
            from .kernels import symmetrized_kernel_matrix

            self.sym_kernel_function = symmetrized_kernel_matrix(self.kernel_function)

        self.updatable = updatable
        if self.updatable:
            self.K_train = None
            self.GS_basis = None
            self.y_train = None
            self.max_considered_molecules = None

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
        """
        Initial guess of the optimal hyperparameters.
        """
        if self.sigmas is None:
            sigma_guess = self.init_sigma_guess_func(X_train)
        else:
            sigma_guess = self.sigmas
        if self.lambda_val is None:
            lambda_guess = 1.0e-6
        else:
            lambda_guess = self.lambda_val
        return np.array([lambda_guess, sigma_guess])

    def optimize_hyperparameters(self, X_train, y_train, init_param_guess=None):
        """
        Use stochastic gradient descent for hyperparameter optimization.
        """
        if init_param_guess is None:
            init_param_guess = self.opt_hyperparameter_guess(X_train, y_train)
        from .hyperparameter_optimization import (
            stochastic_gradient_descent_hyperparam_optimization,
        )

        optimized_hyperparams = stochastic_gradient_descent_hyperparam_optimization(
            X_train,
            y_train,
            init_param_guess=init_param_guess,
            **self.hyperparam_opt_kwargs,
            sym_kernel_func=self.sym_kernel_function,
            optimize_drift=self.optimize_label_drift
        )
        self.sigmas = optimized_hyperparams["sigmas"]
        self.lambda_val = optimized_hyperparams["lambda_val"]
        print("Optimized parameters:", self.sigmas)
        print("Optimized lambda:", self.lambda_val)
        if self.use_label_drift and self.optimize_label_drift:
            self.label_drift = optimized_hyperparams["label_drift"]
            print("Optimized label drift:", self.label_drift)

    def calc_train_kernel(self, X_train):
        """
        Calculates the symmetric kernel matrix according to the current hyperparameters.
        X_train: numpy array of representations of the training data.
        """
        output = self.sym_kernel_function(X_train, self.sigmas)
        output[np.diag_indices_from(output)] += self.lambda_val
        return output

    def calc_kernel(self, X1, X2):
        """
        Calculates the kernel matrix according to the current hyperparameters.
        X1, X2: numpy arrays of representations
        """
        return self.kernel_function(X1, X2, self.sigmas)

    def fit(self, X_train, y_train, optimize_hyperparameters=False, K_train=None):
        from .linear_algebra import scipy_cho_solve

        """
        Fit kernel model to the training data and perform a hyperparameter search.
        X_train: numpy array of representations of the training data
        y_train: numpy array of labels of the training data
        """
        if self.scale_features:
            self.X_maxval = np.max(X_train)
            self.X_minval = np.min(X_train)
            X_train = self.X_MinMaxScaler(X_train)

        if self.scale_labels:
            self.y_maxval = np.max(y_train)
            self.y_minval = np.min(y_train)
            y_train = self.y_MinMaxScaler(y_train)

        if self.use_label_drift:
            y_train = np.copy(y_train)
            if not self.optimize_label_drift:
                if self.label_drift is None:
                    self.label_drift = np.mean(y_train)
                y_train[:] -= self.label_drift

        if (
            optimize_hyperparameters
            or (self.sigmas is None)
            or (self.lambda_val is None)
        ):
            self.optimize_hyperparameters(X_train, y_train)
            if self.use_label_drift:
                if self.optimize_label_drift:
                    y_train[:] -= self.label_drift

        if K_train is None:
            K_train = self.calc_train_kernel(X_train)
        alphas = scipy_cho_solve(K_train, y_train)

        if self.updatable:
            self.initialize_updatable_components(K_train, X_train, y_train)

            self.X_train

        else:
            self.X_train = X_train

        del K_train
        self.alphas = alphas

    def predict(self, X_test, K_test=None):

        """
        Predict the labels of the test data X_test
        using the trained model.

        X_test: numpy array
        """

        if self.scale_features:
            X_test = self.X_MinMaxScaler(X_test)

        if K_test is None:
            if self.parallelized_model_evaluation:
                K_test = self.calc_kernel(self.X_train, X_test).T
            else:
                K_test = self.calc_kernel(X_test, self.X_train)
        y_pred = np.dot(K_test, self.alphas)

        if self.use_label_drift:
            y_pred += self.label_drift

        if self.scale_labels:
            y_pred = self.y_MinMaxScaler_inverse(y_pred)

        return y_pred

    def save(self, filename):

        from .utils import dump2pkl

        """
        Save the trained model to a file. 
        filename: string
        """

        dump2pkl(self, filename)

    # For functionality related to updating alpha coefficients at O(N**2) cost.
    def update_max_considered_molecules(self, new_max_considered_molecules):
        from .utils import np_resize

        self.K_train = np_resize(
            self.K_train, (new_max_considered_molecules, new_max_considered_molecules)
        )
        self.GS_train = np_resize(
            self.GS_train, (new_max_considered_molecules, new_max_considered_molecules)
        )

    def update(self, new_rep, new_y):
        pass


def learning_curve(
    krr_model,
    X_train,
    y_train,
    X_test,
    y_test,
    training_set_sizes,
    max_subset_num=8,
    model_fit_reusable=False,
):
    """
    Generate a MAE learning curve.
    krr_model : KRR class object with pre-initialized hyperparameters
    max_subset_num : maximal number of random subset for a training set size
    model_fit_reusable : ensure that after the last fitting instance the model is reusable
    """
    import random

    K_train = krr_model.calc_train_kernel(X_train)
    K_test = krr_model.calc_kernel(X_test, X_train)
    all_ids = list(range(len(X_train)))

    MAEs = []

    for training_set_size in training_set_sizes:
        MAEs_line = []
        num_subsets = min(max_subset_num, len(X_train) // training_set_size)

        all_subset_ids = np.array(
            random.sample(all_ids, training_set_size * num_subsets)
        )
        lb = 0
        ub = training_set_size
        for subset_counter in range(num_subsets):
            subset_ids = all_subset_ids[lb:ub]
            cur_K_test = K_test[:, subset_ids]
            cur_K_train = K_train[subset_ids][:, subset_ids]
            cur_y_train = y_train[subset_ids]

            if (
                model_fit_reusable
                and (subset_counter == num_subsets - 1)
                and (training_set_size is training_set_sizes[-1])
            ):
                cur_X_train = X_train[subset_ids]
            else:
                # We don't need X_train since we have the kernel.
                cur_X_train = None
            krr_model.fit(cur_X_train, cur_y_train, K_train=cur_K_train)
            MAE = np.mean(np.abs(krr_model.predict(None, K_test=cur_K_test) - y_test))
            MAEs_line.append(MAE)

            lb += training_set_size
            ub += training_set_size
        MAEs.append(MAEs_line)
    return MAEs
