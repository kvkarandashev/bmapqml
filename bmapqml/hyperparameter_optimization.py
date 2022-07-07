# MIT License
#
# Copyright (c) 2021-2022 Konstantin Karandashev
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Implements several optimization techniques based on stochastic gradient descent for conventient hyperparameter optimization.
# The code was written assuming the kernel element calculation to be expensive enough to warrant separate lambda optimization (without
# kernel recalculation) after each step.

import numpy as np
from numba import njit, prange
from scipy.linalg import cho_factor, cho_solve
import math, random, copy
from .utils import dump2pkl, nullify_ignored
from .kernels import (
    gaussian_kernel_matrix,
    gaussian_sym_kernel_matrix,
    symmetrized_kernel_matrix,
)
from scipy.optimize import minimize
from .utils import embarrassingly_parallel
from .linear_algebra import Cho_multi_factors


#   For going between the full hyperparameter set (lambda, global sigma, and other sigmas)
#   and a reduced hyperparameter set. The default class uses no reduction, just rescaling to logarithms.
class Reduced_hyperparam_func:
    def __init__(self):
        self.num_full_params = None
        self.num_reduced_params = None

    def initiate_param_nums(self, param_dim_arr):
        if (self.num_full_params is None) and (self.num_reduced_params is None):
            if isinstance(param_dim_arr, int):
                self.num_full_params = param_dim_arr
            else:
                self.num_full_params = len(param_dim_arr)
            self.num_reduced_params = self.num_full_params

    def reduced_params_to_full(self, reduced_parameters):
        return np.exp(reduced_parameters)

    def full_derivatives_to_reduced(self, full_derivatives, full_parameters):
        return full_derivatives * full_parameters

    def full_params_to_reduced(self, full_parameters):
        self.initiate_param_nums(full_parameters)
        return np.log(full_parameters)

    def initial_reduced_parameter_guess(self, sigmas):
        num_full_params = len(sigmas) + 1
        self.initiate_param_nums(len(sigmas) + 1)
        output = np.zeros((self.num_full_params,))
        output[0] = 0.0
        output[-len(sigmas) :] = np.log(sigmas * np.sqrt(len(sigmas)))
        return output

    def jacobian(self, parameters):
        self.initiate_param_nums(parameters)
        output = np.zeros((self.num_full_params, self.num_reduced_params))
        for param_id in range(self.num_full_params):
            cur_unity_vector = np.zeros((self.num_full_params,))
            cur_unity_vector[param_id] = 1.0
            output[param_id, :] = self.full_derivatives_to_reduced(
                cur_unity_vector, parameters
            )[:]
        return output

    def transform_der_array_to_reduced(self, input_array, parameters):
        jacobian = self.jacobian(parameters)
        return np.matmul(input_array, jacobian)

    def str_output_dict(self, global_name, output_dict=None):
        str_output = global_name + "\n"
        if output_dict is not None:
            str_output = global_name + "\n"
            for str_id in output_dict:
                str_output += str_id + ": " + str(output_dict[str_id]) + "\n"
        return str_output[:-1]

    def __str__(self):
        return self.str_output_dict("Default Reduced_hyperparam_func")

    def __repr__(self):
        return str(self)


class Gradient_optimization_obj:
    def __init__(
        self,
        training_kernel_input,
        training_quants,
        check_kernel_input,
        check_quants,
        training_quants_ignore=None,
        check_quants_ignore=None,
        use_MAE=True,
        reduced_hyperparam_func=None,
        sym_kernel_func=gaussian_sym_kernel_matrix,
        kernel_func=gaussian_kernel_matrix,
        quants_ignore_orderable=False,
        **kernel_additional_args
    ):

        self.reduced_hyperparam_func = reduced_hyperparam_func

        self.training_kernel_input = training_kernel_input
        self.training_quants = training_quants
        self.check_kernel_input = check_kernel_input
        self.check_quants = check_quants

        self.kernel_additional_args = kernel_additional_args

        self.training_quants = training_quants
        self.check_quants = check_quants

        self.training_quants_ignore = training_quants_ignore
        self.check_quants_ignore = check_quants_ignore
        self.quants_ignore_orderable = quants_ignore_orderable

        self.use_MAE = use_MAE

        self.init_kernel_funcs(sym_kernel_func=sym_kernel_func, kernel_func=kernel_func)

    def init_kernel_funcs(
        self,
        sym_kernel_func=gaussian_sym_kernel_matrix,
        kernel_func=gaussian_kernel_matrix,
    ):
        self.kernel_func = kernel_func
        if sym_kernel_func is None:
            self.sym_kernel_func = symmetrized_kernel_matrix(self.kernel_func)
        else:
            self.sym_kernel_func = sym_kernel_func

    def reinitiate_basic_params(self, parameters):
        self.all_parameters = parameters
        self.num_params = len(self.all_parameters)
        self.lambda_val = parameters[0]
        self.sigmas = parameters[1:]

    def recalculate_kernel_matrices(self):
        self.train_kernel = self.sym_kernel_func(
            self.training_kernel_input, self.sigmas, **self.kernel_additional_args
        )
        self.check_kernel = self.kernel_func(
            self.check_kernel_input,
            self.training_kernel_input,
            self.sigmas,
            **self.kernel_additional_args
        )

    def recalculate_kernel_mats_ders(self):
        train_kernel_wders = self.sym_kernel_func(
            self.training_kernel_input,
            self.sigmas,
            with_ders=True,
            **self.kernel_additional_args
        )
        check_kernel_wders = self.def_kern_func(
            self.check_kernel_input,
            self.training_kernel_input,
            self.sigmas,
            with_ders=True,
            **self.kernel_additional_args
        )

        self.train_kernel = train_kernel_wders[:, :, 0]
        self.check_kernel = check_kernel_wders[:, :, 0]

        num_train_compounds = self.train_kernel.shape[0]
        num_check_compounds = self.check_kernel.shape[0]

        self.train_kernel_ders = one_diag_unity_tensor(
            num_train_compounds, self.num_params
        )
        self.check_kernel_ders = np.zeros(
            (num_check_compounds, num_train_compounds, self.num_params)
        )

        self.train_kernel_ders[:, :, 1:] = train_kernel_wders[:, :, 1:]
        self.check_kernel_ders[:, :, 1:] = check_kernel_wders[:, :, 1:]
        if self.reduced_hyperparam_func is not None:
            self.train_kernel_ders = (
                self.reduced_hyperparam_func.transform_der_array_to_reduced(
                    self.train_kernel_ders, self.all_parameters
                )
            )
            self.check_kernel_ders = (
                self.reduced_hyperparam_func.transform_der_array_to_reduced(
                    self.check_kernel_ders, self.all_parameters
                )
            )

    def reinitiate_cho_decomps(self):

        modified_train_kernel = np.copy(self.train_kernel)
        modified_train_kernel[
            np.diag_indices_from(modified_train_kernel)
        ] += self.lambda_val
        try:
            self.train_cho_decomp = Cho_multi_factors(
                modified_train_kernel,
                indices_to_ignore=self.training_quants_ignore,
                ignored_orderable=self.quants_ignore_orderable,
            )
        except np.linalg.LinAlgError:  # means mod_train_kernel is not invertible
            self.train_cho_decomp = None
        self.train_kern_invertible = self.train_cho_decomp is not None

    def reinitiate_alphas_errors(self):
        if self.train_kern_invertible:
            self.talphas = self.train_cho_decomp.solve_with(self.training_quants).T
            self.predictions = np.matmul(self.check_kernel, self.talphas)
            self.prediction_errors = self.predictions - self.check_quants
            nullify_ignored(self.prediction_errors, self.check_quants_ignore)

    def reinitiate_error_measures(self):
        if self.train_kern_invertible:
            self.cur_MAE = np.mean(np.abs(self.prediction_errors))
            self.cur_MSE = np.mean(self.prediction_errors**2)
        else:
            self.cur_MAE = None
            self.cur_MSE = None

    def der_predictions(self, train_der, check_der):
        output = np.matmul(train_der, self.talphas)
        output = self.train_cho_decomp.solve_with(output).T
        output = -np.matmul(self.check_kernel, output)
        output += np.matmul(check_der, self.talphas)
        return output

    def drift_response_coeffs(self):
        output = self.train_cho_decomp.solve_with(
            np.ones(self.talphas.shape, dtype=float)
        ).T
        return np.matmul(self.check_kernel, output) - np.ones(self.talphas.shape)

    def reinitiate_error_measure_ders(self, lambda_der_only=False):
        if self.train_kern_invertible:
            if lambda_der_only:
                num_ders = 1
            else:
                num_ders = self.check_kernel_ders.shape[2]
            self.cur_MSE_der = np.zeros((num_ders,))
            self.cur_MAE_der = np.zeros((num_ders,))
            for der_id in range(num_ders):
                cur_der_predictions = self.der_predictions(
                    self.train_kernel_ders[:, :, der_id],
                    self.check_kernel_ders[:, :, der_id],
                )
                nullify_ignored(cur_der_predictions, self.check_quants_ignore)
                self.cur_MSE_der[der_id] = 2 * np.mean(
                    self.prediction_errors * cur_der_predictions
                )
                self.cur_MAE_der[der_id] = np.mean(
                    cur_der_predictions * np.sign(self.prediction_errors)
                )
        else:
            self.cur_MSE_der = non_invertible_default_log_der()
            self.cur_MAE_der = non_invertible_default_log_der()

    def error_measure(self, parameters):
        self.reinitiate_basic_params(parameters)
        self.recalculate_kernel_matrices()
        self.reinitiate_cho_decomps()
        self.reinitiate_alphas_errors()
        self.reinitiate_error_measures()
        print("# Current parameter vales:", parameters)
        print("# Current MAE:", self.cur_MAE, "MSE:", self.cur_MSE)
        if self.use_MAE:
            return self.cur_MAE
        else:
            return self.cur_MSE

    def error_measure_ders(
        self,
        parameters,
        lambda_der_only=False,
    ):

        self.reinitiate_basic_params(parameters)
        self.recalculate_kernel_mats_ders()
        self.reinitiate_cho_decomps()
        self.reinitiate_alphas_errors()
        self.reinitiate_error_measure_ders(lambda_der_only=lambda_der_only)

        print("# Current parameter values:", parameters)
        print("# Current MSE derivatives:", self.cur_MSE_der)
        print("# Current MAE derivatives:", self.cur_MAE_der)
        if self.use_MAE:
            return self.cur_MAE_der
        else:
            return self.cur_MSE_der

    def error_measure_wders(
        self, parameters, lambda_der_only=False, recalc_cho_decomps=True
    ):
        if recalc_cho_decomps:
            self.reinitiate_basic_params(parameters)
            self.recalculate_kernel_mats_ders()
            self.reinitiate_cho_decomps()
        self.reinitiate_alphas_errors()
        self.reinitiate_error_measure_ders(lambda_der_only=lambda_der_only)
        self.reinitiate_error_measures()
        if self.use_MAE:
            return self.cur_MAE, self.cur_MAE_der
        else:
            return self.cur_MSE, self.cur_MSE_der


def non_invertible_default_log_der():
    return np.array([-1.0])  # we only have derivative for lambda value


class GOO_ensemble_subset(Gradient_optimization_obj):
    def __init__(
        self,
        training_indices,
        check_indices,
        all_quantities,
        use_MAE=True,
        quants_ignore=None,
        quants_ignore_orderable=False,
    ):
        self.training_indices = training_indices
        self.check_indices = check_indices

        self.init_quants(all_quantities)

        if quants_ignore is None:
            self.training_quants_ignore = None
            self.check_quants_ignore = None
        else:
            self.training_quants_ignore = quants_ignore[self.training_indices, :]
            self.check_quants_ignore = quants_ignore[self.check_indices, :]

        self.use_MAE = use_MAE
        self.quants_ignore_orderable = quants_ignore_orderable

    def init_quants(self, all_quantities):
        if len(all_quantities.shape) == 1:
            self.training_quants = all_quantities[self.training_indices]
            self.check_quants = all_quantities[self.check_indices]
        else:
            self.training_quants = all_quantities[self.training_indices, :]
            self.check_quants = all_quantities[self.check_indices, :]

    def recalculate_kernel_matrices(self, global_matrix=None, global_matrix_ders=None):
        if global_matrix is not None:
            self.train_kernel = global_matrix[self.training_indices, :][
                :, self.training_indices
            ]
            self.check_kernel = global_matrix[self.check_indices, :][
                :, self.training_indices
            ]

    def recalculate_kernel_mats_ders(self, global_matrix=None, global_matrix_ders=None):
        self.recalculate_kernel_matrices(global_matrix=global_matrix)
        if global_matrix_ders is not None:
            self.train_kernel_ders = global_matrix_ders[self.training_indices, :][
                :, self.training_indices
            ][:]
            self.check_kernel_ders = global_matrix_ders[self.check_indices, :][
                :, self.training_indices
            ][:]


# This class was introduced to enable multiple cross-validation.
class GOO_ensemble(Gradient_optimization_obj):
    def __init__(
        self,
        all_compounds_kernel_input,
        all_quantities,
        train_id_lists,
        check_id_lists,
        quants_ignore=None,
        use_MAE=True,
        reduced_hyperparam_func=None,
        num_procs=1,
        fixed_num_threads=None,
        kernel_func=None,
        sym_kernel_func=gaussian_sym_kernel_matrix,
        optimize_drift=False,
        quants_ignore_orderable=False,
        **kernel_additional_args
    ):

        self.reduced_hyperparam_func = reduced_hyperparam_func

        self.init_kernel_funcs(sym_kernel_func=sym_kernel_func, kernel_func=kernel_func)

        self.kernel_additional_args = kernel_additional_args

        self.all_compounds_kernel_input = all_compounds_kernel_input

        self.tot_num_points = all_quantities.shape[-1]

        self.optimize_drift = optimize_drift

        self.all_quantities = np.copy(all_quantities)
        if self.optimize_drift:
            self.init_quantities = np.copy(all_quantities)

        self.goo_ensemble_subsets = []
        for train_id_list, check_id_list in zip(train_id_lists, check_id_lists):
            self.goo_ensemble_subsets.append(
                GOO_ensemble_subset(
                    train_id_list,
                    check_id_list,
                    self.all_quantities,
                    quants_ignore=quants_ignore,
                    use_MAE=use_MAE,
                    quants_ignore_orderable=quants_ignore_orderable,
                )
            )
        self.num_subsets = len(self.goo_ensemble_subsets)

        self.presaved_parameters = None

        self.num_procs = num_procs

        self.fixed_num_threads = fixed_num_threads

    def error_measure_wders(
        self,
        parameters,
        recalc_global_matrices=True,
        recalc_cho_decomps=True,
        lambda_der_only=False,
        negligible_red_param_distance=None,
    ):
        if recalc_global_matrices:
            need_recalc = True
            if negligible_red_param_distance is not None:
                if self.presaved_parameters is not None:
                    need_recalc = (
                        np.sqrt(np.sum((parameters - self.presaved_parameters) ** 2))
                        > negligible_red_param_distance
                    )
                if need_recalc:
                    self.presaved_parameters = np.copy(parameters)
            if need_recalc:
                self.recalculate_global_matrices(parameters)

        error_mean = 0.0
        error_mean_ders = None

        error_erders_list = self.subset_error_measures_wders(
            parameters,
            lambda_der_only=lambda_der_only,
            recalc_cho_decomps=recalc_cho_decomps,
        )

        for cur_error, cur_error_ders in error_erders_list:
            if cur_error is None:
                error_mean = 0.0
                error_mean_ders = non_invertible_default_log_der()
                break
            else:
                error_mean += cur_error
                if error_mean_ders is None:
                    error_mean_ders = np.copy(cur_error_ders)
                else:
                    error_mean_ders += cur_error_ders
        error_mean /= self.num_subsets
        error_mean_ders /= self.num_subsets
        return error_mean, error_mean_ders

    def subset_error_measures_wders(
        self, parameters, lambda_der_only=False, recalc_cho_decomps=True
    ):
        if (self.num_procs is None) or (self.num_procs == 1):
            return [
                goo_ensemble_subset.error_measure_wders(
                    parameters,
                    lambda_der_only=lambda_der_only,
                    recalc_cho_decomps=recalc_cho_decomps,
                )
                for goo_ensemble_subset in self.goo_ensemble_subsets
            ]
        else:
            return embarrassingly_parallel(
                single_subset_error_measure_wders,
                self.goo_ensemble_subsets,
                (parameters, lambda_der_only, recalc_cho_decomps),
                fixed_num_threads=self.fixed_num_threads,
                num_procs=self.num_procs,
            )

    def recalculate_global_matrices(self, parameters):
        global_kernel_wders = self.sym_kernel_func(
            self.all_compounds_kernel_input,
            parameters[1:],
            with_ders=True,
            **self.kernel_additional_args
        )
        self.global_matrix = global_kernel_wders[:, :, 0]

        print(
            "# GOO_ensemble: Kernel recalculated, average diagonal element:",
            np.mean(self.global_matrix[np.diag_indices_from(self.global_matrix)]),
        )
        self.global_matrix_ders = one_diag_unity_tensor(
            self.tot_num_points, len(parameters)
        )
        self.global_matrix_ders[:, :, 1:] = global_kernel_wders[:, :, 1:]

        if self.reduced_hyperparam_func is not None:
            self.global_matrix_ders = (
                self.reduced_hyperparam_func.transform_der_array_to_reduced(
                    self.global_matrix_ders, parameters
                )
            )

        for subset_id in range(self.num_subsets):
            self.goo_ensemble_subsets[subset_id].recalculate_kernel_mats_ders(
                global_matrix=self.global_matrix,
                global_matrix_ders=self.global_matrix_ders,
            )

    def update_label_drift(self, label_drift):
        self.all_quantities[:] = self.init_quantities[:]
        self.all_quantities -= label_drift
        for subset_id in range(self.num_subsets):
            self.goo_ensemble_subsets[subset_id].init_quants(self.all_quantities)


# Auxiliary function for joblib parallelization.
def single_subset_error_measure_wders(
    subset, parameters, lambda_der_only, recalc_cho_decomps
):
    return subset.error_measure_wders(
        parameters,
        lambda_der_only=lambda_der_only,
        recalc_cho_decomps=recalc_cho_decomps,
    )


def generate_random_GOO_ensemble(
    all_compounds_kernel_input,
    all_quantities,
    quants_ignore=None,
    num_kfolds=16,
    training_set_ratio=0.5,
    use_MAE=True,
    reduced_hyperparam_func=None,
    num_procs=1,
    fixed_num_threads=None,
    sym_kernel_func=None,
    **other_kwargs
):
    num_points = all_quantities.shape[-1]
    train_point_num = int(num_points * training_set_ratio)

    all_indices = list(range(num_points))

    train_id_lists = []
    check_id_lists = []

    for kfold_id in range(num_kfolds):
        train_id_list = random.sample(all_indices, train_point_num)
        train_id_list.sort()
        check_id_list = []
        for train_interval_id in range(train_point_num + 1):
            if train_interval_id == 0:
                lower_bound = 0
            else:
                lower_bound = train_id_list[train_interval_id - 1] + 1
            if train_interval_id == train_point_num:
                upper_bound = num_points
            else:
                upper_bound = train_id_list[train_interval_id]
            for index in range(lower_bound, upper_bound):
                check_id_list.append(index)
        train_id_lists.append(train_id_list)
        check_id_lists.append(check_id_list)

    return GOO_ensemble(
        all_compounds_kernel_input,
        all_quantities,
        train_id_lists,
        check_id_lists,
        quants_ignore=quants_ignore,
        use_MAE=use_MAE,
        reduced_hyperparam_func=reduced_hyperparam_func,
        num_procs=num_procs,
        fixed_num_threads=fixed_num_threads,
        sym_kernel_func=sym_kernel_func,
        **other_kwargs
    )


class Optimizer_state:
    def __init__(
        self, error_measure, error_measure_red_ders, parameters, red_parameters
    ):
        self.error_measure = error_measure
        self.error_measure_red_ders = error_measure_red_ders
        self.parameters = parameters
        self.red_parameters = red_parameters

    def extended_greater(self, other_state):
        if self.error_measure is None:
            return True
        else:
            if other_state.error_measure is None:
                return False
            else:
                return self.error_measure > other_state.error_measure

    def __gt__(self, other_state):
        return self.extended_greater(other_state)

    def __lt__(self, other_state):
        return not self.extended_greater(other_state)

    def log_lambda_der(self):
        return self.error_measure_red_ders[0]

    def lambda_log_val(self):
        return self.red_parameters[0]

    def __repr__(self):
        return str(self)

    def __str__(self):
        return (
            "Optimizer_state object: Parameters: "
            + str(self.parameters)
            + ", reduced parameters: "
            + str(self.red_parameters)
            + ", error measure: "
            + str(self.error_measure)
        )


class Optimizer_state_generator:
    def __init__(self, goo_ensemble):
        self.goo_ensemble = goo_ensemble

    def __call__(
        self,
        red_parameters,
        recalc_global_matrices=True,
        recalc_cho_decomps=True,
        lambda_der_only=False,
    ):
        parameters = self.goo_ensemble.reduced_hyperparam_func.reduced_params_to_full(
            red_parameters
        )
        error_measure, error_measure_red_ders = self.goo_ensemble.error_measure_wders(
            parameters,
            recalc_global_matrices=recalc_global_matrices,
            recalc_cho_decomps=recalc_cho_decomps,
            lambda_der_only=lambda_der_only,
        )
        return Optimizer_state(
            error_measure, error_measure_red_ders, parameters, red_parameters
        )


class GOO_randomized_iterator:
    def __init__(
        self,
        opt_GOO_ensemble,
        initial_reduced_parameter_vals,
        lambda_opt_tolerance=0.1,
        step_magnitudes=None,
        noise_level_prop_coeffs=None,
        lambda_max_num_scan_steps=256,
        default_step_magnitude=1.0,
        keep_init_lambda=False,
        bisec_lambda_opt=True,
        max_lambda_diag_el_ratio=None,
        min_lambda_diag_el_ratio=1e-12,
    ):

        self.optimizer_state_generator = Optimizer_state_generator(opt_GOO_ensemble)

        self.optimize_drift = self.optimizer_state_generator.goo_ensemble.optimize_drift

        self.cur_optimizer_state = self.optimizer_state_generator(
            initial_reduced_parameter_vals
        )

        if self.optimize_drift:
            self.descend_drift_optimization()

        print(
            "Start params:",
            self.cur_optimizer_state.red_parameters,
            "starting error measure:",
            self.cur_optimizer_state.error_measure,
        )

        self.lambda_opt_tolerance = lambda_opt_tolerance
        self.keep_init_lambda = keep_init_lambda
        self.negligible_lambda_log = -40.0

        self.num_reduced_params = (
            self.optimizer_state_generator.goo_ensemble.reduced_hyperparam_func.num_reduced_params
        )

        if step_magnitudes is None:
            self.step_magnitudes = np.repeat(
                default_step_magnitude, self.num_reduced_params
            )
        else:
            self.step_magnitudes = step_magnitudes

        if noise_level_prop_coeffs is None:
            self.noise_level_prop_coeffs = np.copy(self.step_magnitudes)
        else:
            self.noise_level_prop_coeffs = noise_level_prop_coeffs

        if keep_init_lambda:
            self.noise_level_prop_coeffs[0] = 0.0

        self.noise_levels = np.zeros((self.num_reduced_params,))

        self.lambda_max_num_scan_steps = lambda_max_num_scan_steps
        self.lambda_opt_tolerance = lambda_opt_tolerance

        self.lambda_max_num_scan_steps = lambda_max_num_scan_steps

        self.bisec_lambda_opt = bisec_lambda_opt

        self.change_successful = None

        self.max_lambda_diag_el_ratio = max_lambda_diag_el_ratio
        self.min_lambda_diag_el_ratio = min_lambda_diag_el_ratio

        # If initial lambda value is not large enough for the kernel matrix to be invertible,
        # use bisection to optimize it.
        if (not self.keep_init_lambda) and (self.max_lambda_diag_el_ratio is not None):
            while self.cur_optimizer_state.error_measure is None:
                if not self.keep_init_lambda:
                    self.bisection_lambda_optimization()

    def iterate(self):

        trial_red_params = self.generate_trial_red_params(self.keep_init_lambda)

        if self.lambda_outside_bounds(trial_red_params[0]):
            trial_red_params = self.generate_trial_red_params(True)

        trial_optimizer_state = self.optimizer_state_generator(trial_red_params)

        print(
            "Trial params:",
            trial_red_params,
            ", trial error measure:",
            trial_optimizer_state.error_measure,
        )

        self.change_successful = self.cur_optimizer_state > trial_optimizer_state

        if self.change_successful:
            self.cur_optimizer_state = copy.deepcopy(trial_optimizer_state)
            if (not self.keep_init_lambda) and self.bisec_lambda_opt:
                self.bisection_lambda_optimization()
            if self.optimize_drift:
                self.descend_drift_optimization(need_init_decomp=False)
            self.noise_levels[:] = 0.0
            if self.lambda_outside_bounds():
                self.change_lambda_until_normal()
        else:
            self.noise_levels += self.noise_level_prop_coeffs / np.sqrt(
                self.num_reduced_params
            )

    def generate_trial_red_params(self, keep_init_lambda):
        normalized_red_ders = np.copy(self.cur_optimizer_state.error_measure_red_ders)
        if keep_init_lambda:
            normalized_red_ders[0] = 0.0
        normalized_red_ders = normalized_red_ders / np.sqrt(
            sum((normalized_red_ders / self.step_magnitudes) ** 2)
        )

        print("Normalized reduced derivatives:", normalized_red_ders)

        trial_red_params = np.copy(self.cur_optimizer_state.red_parameters)

        if keep_init_lambda:
            param_range_start = 1
        else:
            param_range_start = 0

        for param_id in range(param_range_start, self.num_reduced_params):
            trial_red_params[param_id] += (
                np.random.normal() * self.noise_levels[param_id]
                - normalized_red_ders[param_id]
            )

        return trial_red_params

    def change_lambda_until_normal(self):
        scan_additive = np.abs(self.step_magnitudes[0])
        if self.lambda_outside_bounds(sign_determinant=scan_additive):
            scan_additive *= -1
        new_red_params = np.copy(self.cur_optimizer_state.red_parameters)

        while self.lambda_outside_bounds(new_red_params[0]):
            trial_red_params = np.copy(new_red_params)
            trial_red_params[0] += scan_additive
            trial_iteration = self.optimizer_state_generator(
                trial_red_params, recalc_global_matrices=False, lambda_der_only=True
            )

            trial_error_measure = trial_iteration.error_measure
            print(
                "Scanning lambda to normalize:",
                trial_red_params[0],
                trial_error_measure,
            )

            if trial_error_measure is None:
                break
            else:
                new_red_params = np.copy(trial_red_params)

        self.recalc_cur_opt_state(new_red_params, recalc_global_matrices=False)

    def recalc_cur_opt_state(
        self,
        new_red_params,
        recalc_global_matrices=True,
        lambda_der_only=False,
        recalc_cho_decomps=True,
    ):
        self.cur_optimizer_state = self.optimizer_state_generator(
            new_red_params,
            recalc_global_matrices=recalc_global_matrices,
            recalc_cho_decomps=recalc_cho_decomps,
            lambda_der_only=lambda_der_only,
        )

    def lambda_outside_bounds(self, log_lambda_value=None, sign_determinant=None):
        if (self.max_lambda_diag_el_ratio is None) and (
            self.min_lambda_diag_el_ratio is None
        ):
            return False
        else:
            if log_lambda_value is None:
                log_lambda_value = self.cur_optimizer_state.red_parameters[0]
            lambda_value = np.exp(log_lambda_value)
            train_kernel_mat = self.optimizer_state_generator.goo_ensemble.global_matrix
            av_kernel_diag_el = np.mean(
                train_kernel_mat[np.diag_indices_from(train_kernel_mat)]
            )
            if self.max_lambda_diag_el_ratio is None:
                too_large = False
            else:
                too_large = (
                    lambda_value > av_kernel_diag_el * self.max_lambda_diag_el_ratio
                )
            if self.min_lambda_diag_el_ratio is None:
                too_small = False
            else:
                too_small = (
                    lambda_value < av_kernel_diag_el * self.min_lambda_diag_el_ratio
                )

            if sign_determinant is None:
                return too_small or too_large
            else:
                if sign_determinant > 0.0:
                    return too_large
                else:
                    return too_small

    def bisection_lambda_optimization(self):
        print(
            "Bisection optimization of lambda, starting position:",
            self.cur_optimizer_state.lambda_log_val(),
            self.cur_optimizer_state.error_measure,
        )

        # Perform a scan to approximately locate the minimum.
        prev_iteration = self.cur_optimizer_state
        prev_lambda_der = prev_iteration.log_lambda_der()
        prev_iteration = copy.deepcopy(self.cur_optimizer_state)

        scan_additive = math.copysign(self.step_magnitudes[0], -prev_lambda_der)

        bisection_interval = None

        for init_scan_step in range(self.lambda_max_num_scan_steps):
            trial_red_params = np.copy(prev_iteration.red_parameters)
            trial_red_params[0] += scan_additive

            if self.lambda_outside_bounds(
                log_lambda_value=trial_red_params[0], sign_determinant=scan_additive
            ):
                self.recalc_cur_opt_state(
                    prev_iteration.red_parameters, recalc_global_matrices=False
                )
                return

            trial_iteration = self.optimizer_state_generator(
                trial_red_params, recalc_global_matrices=False, lambda_der_only=True
            )
            trial_lambda_der = trial_iteration.log_lambda_der()

            print(
                "Initial lambda scan:",
                trial_red_params[0],
                trial_lambda_der,
                trial_iteration.error_measure,
            )

            if (trial_lambda_der * prev_lambda_der) < 0.0:
                bisection_interval = [trial_iteration, prev_iteration]
                bisection_interval.sort(key=lambda x: x.lambda_log_val())
                break
            else:
                if (
                    trial_iteration > prev_iteration
                ):  # something is wrong, we were supposed to be going down in MSE.
                    print("WARNING: Weird behavior during lambda value scan.")
                    self.recalc_cur_opt_state(
                        prev_iteration.red_parameters, recalc_global_matrices=False
                    )
                    return
                else:
                    prev_iteration = trial_iteration

        if bisection_interval is None:
            self.recalc_cur_opt_state(
                trial_iteration.red_parameters, recalc_global_matrices=False
            )
            return

        # Finalize locating the minumum via bisection.
        # Use bisection search to find the minimum. Note that we do not need to recalculate the kernel matrices.
        while (
            bisection_interval[1].lambda_log_val()
            > bisection_interval[0].lambda_log_val() + self.lambda_opt_tolerance
        ):
            cur_log_lambda = (
                bisection_interval[0].lambda_log_val()
                + bisection_interval[1].lambda_log_val()
            ) / 2

            middle_params = np.copy(self.cur_optimizer_state.red_parameters)
            middle_params[0] = cur_log_lambda

            middle_iteration = self.optimizer_state_generator(
                middle_params, recalc_global_matrices=False, lambda_der_only=True
            )

            middle_der = middle_iteration.log_lambda_der()

            print(
                "Bisection lambda optimization, lambda logarithm:",
                cur_log_lambda,
                "derivative:",
                middle_der,
                "error measure:",
                middle_iteration.error_measure,
            )

            for bisec_int_id in range(2):
                if middle_der * bisection_interval[bisec_int_id].log_lambda_der() > 0.0:
                    bisection_interval[bisec_int_id] = middle_iteration
            bisection_interval.sort(key=lambda x: x.lambda_log_val())

        self.recalc_cur_opt_state(
            min(bisection_interval).red_parameters, recalc_global_matrices=False
        )

    def descend_drift_optimization(self, need_init_decomp=True):
        print(
            "Descend drift optimization starting error measure:",
            self.cur_optimizer_state.error_measure,
        )
        label_diffs = np.empty((0,), dtype=float)
        response_coeffs = np.empty((0,), dtype=float)
        # We need to adjust the drift to zero for the analytic formula to be correct.
        self.optimizer_state_generator.goo_ensemble.update_label_drift(0.0)
        for goo_subset_id in range(
            self.optimizer_state_generator.goo_ensemble.num_subsets
        ):
            goo_subset = (
                self.optimizer_state_generator.goo_ensemble.goo_ensemble_subsets[
                    goo_subset_id
                ]
            )
            if need_init_decomp:
                goo_subset.reinitiate_cho_decomps()
            goo_subset.reinitiate_alphas_errors()
            goo_subset.reinitiate_error_measures()

            label_diffs = np.append(label_diffs, goo_subset.prediction_errors)
            response_coeffs = np.append(
                response_coeffs, goo_subset.drift_response_coeffs()
            )

        switching_positions = label_diffs / response_coeffs
        abs_response_coeffs = np.abs(response_coeffs)

        comp_tuples = [
            (switching_position, abs_response_coeff)
            for switching_position, abs_response_coeff in zip(
                switching_positions, abs_response_coeffs
            )
        ]

        comp_tuples.sort(key=lambda x: x[0])
        cur_der = -np.sum(-abs_response_coeffs)
        print("AAA", cur_der, comp_tuples)
        for comp_tuple in comp_tuples:
            cur_der += 2 * comp_tuple[1]
            if cur_der > 0:
                self.label_drift = comp_tuple[0]
                break

        self.optimizer_state_generator.goo_ensemble.update_label_drift(self.label_drift)
        self.recalc_cur_opt_state(
            self.cur_optimizer_state.red_parameters,
            recalc_global_matrices=False,
            recalc_cho_decomps=False,
        )

        print(
            "Descend drift optimization final error measure:",
            self.cur_optimizer_state.error_measure,
        )


list_supported_funcs = [
    "default",
    "single_rescaling",
    "single_rescaling_global_mat_prop_coeffs",
    "ang_mom_classified",
]


def stochastic_gradient_descent_hyperparam_optimization(
    kernel_input,
    quant_arr,
    quant_ignore_list=None,
    quants_ignore_orderable=False,
    max_iterations=256,
    init_param_guess=None,
    init_red_param_guess=None,
    reduced_hyperparam_func=Reduced_hyperparam_func(),
    max_stagnating_iterations=1,
    use_MAE=True,
    num_kfolds=16,
    other_opt_goo_ensemble_kwargs={},
    randomized_iterator_kwargs={},
    iter_dump_name_add=None,
    additional_BFGS_iters=None,
    iter_dump_name_add_BFGS=None,
    negligible_red_param_distance=1e-9,
    num_procs=1,
    fixed_num_threads=None,
    sym_kernel_func=gaussian_sym_kernel_matrix,
    optimize_drift=False,
):

    if init_red_param_guess is None:
        init_red_param_guess = reduced_hyperparam_func.full_params_to_reduced(
            init_param_guess
        )

    opt_GOO_ensemble = generate_random_GOO_ensemble(
        kernel_input,
        quant_arr,
        quants_ignore=quant_ignore_list,
        use_MAE=use_MAE,
        num_kfolds=num_kfolds,
        reduced_hyperparam_func=reduced_hyperparam_func,
        **other_opt_goo_ensemble_kwargs,
        quants_ignore_orderable=quants_ignore_orderable,
        num_procs=num_procs,
        fixed_num_threads=fixed_num_threads,
        sym_kernel_func=sym_kernel_func,
        optimize_drift=optimize_drift
    )

    randomized_iterator = GOO_randomized_iterator(
        opt_GOO_ensemble, init_red_param_guess, **randomized_iterator_kwargs
    )
    num_stagnating_iterations = 0
    num_iterations = 0

    iterate_more = True

    while iterate_more:
        randomized_iterator.iterate()
        cur_opt_state = randomized_iterator.cur_optimizer_state
        num_iterations += 1

        if iter_dump_name_add is not None:
            cur_dump_name = iter_dump_name_add + "_" + str(num_iterations) + ".pkl"
            dump2pkl(cur_opt_state, cur_dump_name)

        print(
            "Parameters:",
            cur_opt_state.parameters,
            "error measure:",
            cur_opt_state.error_measure,
        )
        if max_iterations is not None:
            if num_iterations >= max_iterations:
                iterate_more = False
        if max_stagnating_iterations is not None:
            if randomized_iterator.change_successful:
                num_stagnating_iterations = 0
            else:
                num_stagnating_iterations += 1
                if num_stagnating_iterations >= max_stagnating_iterations:
                    iterate_more = False

    if additional_BFGS_iters is not None:
        error_measure_func = GOO_standalone_error_measure(
            opt_GOO_ensemble,
            reduced_hyperparam_func,
            negligible_red_param_distance,
            iter_dump_name_add=iter_dump_name_add_BFGS,
        )
        if iter_dump_name_add_BFGS is None:
            iter_dump_name_add_BFGS_grad = None
        else:
            iter_dump_name_add_BFGS_grad = iter_dump_name_add_BFGS + "_grad"
        error_measure_ders_func = GOO_standalone_error_measure_ders(
            opt_GOO_ensemble,
            reduced_hyperparam_func,
            negligible_red_param_distance,
            iter_dump_name_add=iter_dump_name_add_BFGS_grad,
        )
        finalized_result = minimize(
            error_measure_func,
            cur_opt_state.red_parameters,
            method="BFGS",
            jac=error_measure_ders_func,
            options={"disp": True, "maxiter": additional_BFGS_iters},
        )
        print("BFGS corrected error measure:", finalized_result.fun)
        if finalized_result.fun < cur_opt_state.error_measure:
            finalized_params = reduced_hyperparam_func.reduced_params_to_full(
                finalized_result.x
            )
            return {
                "sigmas": finalized_params[1:],
                "lambda_val": finalized_params[0],
                "error_measure": finalized_result.fun,
            }
    output = {
        "sigmas": cur_opt_state.parameters[1:],
        "lambda_val": cur_opt_state.parameters[0],
        "error_measure": cur_opt_state.error_measure,
    }
    if optimize_drift:
        output = {**output, "label_drift": randomized_iterator.label_drift}
    return output


######
#   Functions introduced to facilitate coupling with standard minimization protocols from scipy.
#####


class GOO_standalone_error_measure:
    def __init__(
        self,
        GOO_ensemble,
        reduced_hyperparam_func,
        negligible_red_param_distance,
        iter_dump_name_add=None,
    ):
        self.GOO = GOO_ensemble
        self.reduced_hyperparam_func = reduced_hyperparam_func
        self.iter_dump_name_add = iter_dump_name_add
        self.num_calls = 0
        self.negligible_red_param_distance = negligible_red_param_distance

    def __call__(self, red_parameters):
        self.parameters = self.reduced_hyperparam_func.reduced_params_to_full(
            red_parameters
        )
        self.reinit_quants()
        self.dump_intermediate_result()
        return self.result()

    def reinit_quants(self):
        self.error_measure, self.error_measure_ders = self.GOO.error_measure_wders(
            self.parameters,
            negligible_red_param_distance=self.negligible_red_param_distance,
        )

    def dump_intermediate_result(self):
        if self.iter_dump_name_add is not None:
            self.num_calls += 1
            cur_dump_name = self.iter_dump_name_add + "_" + str(self.num_calls) + ".pkl"
            dump2pkl([self.parameters, self.result()], cur_dump_name)

    def result(self):
        return self.error_measure


class GOO_standalone_error_measure_ders(GOO_standalone_error_measure):
    def result(self):
        return self.error_measure_ders


#####
# Auxiliary functions.
#####


def one_diag_unity_tensor(dim12, dim3):
    output = np.zeros((dim12, dim12, dim3))
    for mol_id in range(dim12):
        output[mol_id, mol_id, 0] = 1.0
    return output


#####
# For initial guesses of hyperparameters.
#####
@njit(fastmath=True)
def gauss_dist(X1, X2):
    return np.sum((X1 - X2) ** 2)


@njit(fastmath=True)
def laplace_dist(X1, X2):
    return np.sum(np.abs(X1 - X2))


@njit(fastmath=True, parallel=True)
def max_dist(X_arr, dist_func):
    l = X_arr.shape[0]
    m = np.zeros((2,))
    for i in prange(l):
        for j in range(i):
            m[1] = dist_func(X_arr[i], X_arr[j])
            m[0] = np.max(m)
    return m[0]


def max_Laplace_dist(X_arr):
    return max_dist(X_arr, laplace_dist)


def max_Gauss_dist(X_arr):
    return np.sqrt(max_dist(X_arr, gauss_dist))
