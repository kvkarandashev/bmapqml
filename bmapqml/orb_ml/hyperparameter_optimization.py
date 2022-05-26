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

from .representations import component_id_ang_mom_map, scalar_rep_length, OML_rep_params
from ..hyperparameter_optimization import Reduced_hyperparam_func
import numpy as np

# For using a simple rescaling coefficient for sigmas. The set of reduced hyperparameters used for optimization
# in arXiv:2112.12877/DOI:10.1063/5.0083301.
class Single_rescaling_rhf(Reduced_hyperparam_func):
    def __init__(self, use_Gauss=False, rep_params=None, use_global_mat_prop_coeffs=False, stddevs=None):
        if rep_params is None:
            rep_params=OML_rep_params
        if use_global_mat_prop_coeffs:
            coup_mat_prop_coeffs, prop_coeff_id=matrix_grouped_prop_coeffs(stddevs=stddevs,
                                ang_mom_map=ang_mom_map, rep_params=rep_params)
            coup_sigmas=[]
            for sigma_id, cur_prop_coeff_id in enumerate(prop_coeff_id):
                coup_sigmas.append(coup_mat_prop_coeffs[cur_prop_coeff_id])
            self.coup_sigmas=np.array(coup_sigmas)
        else:
            self.coup_sigmas=stddevs
        self.num_full_params=len(self.coup_sigmas)+1
        self.num_reduced_params=2
        self.use_Gauss=use_Gauss
        self.coup_sigmas_start_id=1
        if self.use_Gauss:
            self.num_full_params+=1
            self.num_reduced_params+=1
            self.coup_sigmas_start_id+=1
    def reduced_params_to_full(self, reduced_parameters):
        output=np.zeros((self.num_full_params,))
        output[:self.coup_sigmas_start_id]=np.exp(reduced_parameters[:self.coup_sigmas_start_id])
        output[self.coup_sigmas_start_id:]=self.coup_sigmas*np.exp(reduced_parameters[-1])
        return output
    def full_derivatives_to_reduced(self, full_derivatives, full_parameters):
        output=np.zeros((self.num_reduced_params,))
        output[:self.coup_sigmas_start_id]=full_derivatives[:self.coup_sigmas_start_id]*full_parameters[:self.coup_sigmas_start_id]
        for sigma_id in range(self.coup_sigmas_start_id, self.num_full_params):
            output[-1]+=full_derivatives[sigma_id]*full_parameters[sigma_id]
        return output
    def full_params_to_reduced(self, full_parameters):
        output=np.zeros((self.num_reduced_params,))
        output[:self.coup_sigmas_start_id]=np.log(full_parameters[:self.coup_sigmas_start_id])
        est_counter=0
        for sigma_id in range(self.coup_sigmas_start_id, self.num_full_params):
            output[-1]+=np.log(full_parameters[sigma_id]/self.coup_sigmas[sigma_id-self.coup_sigmas_start_id])
            est_counter+=1
        output[-1]/=est_counter
        return output
    def initial_reduced_parameter_guess(self, init_lambda, *other_args):
        init_resc_param_guess=np.log(self.num_full_params-2)
        if self.use_Gauss:
            return np.array([np.log(init_lambda), 0.0, init_resc_param_guess])
        else:
            return np.array([np.log(init_lambda), init_resc_param_guess])
    def __str__(self):
        vals_names={"num_full_params" : self.num_full_params,
                    "num_reduced_params" : self.num_reduced_params,
                    "coup_sigmas" : self.coup_sigmas}
        return self.str_output_dict("Single_rescaling_rhf", vals_names)
        

def matrix_grouped_prop_coeffs(rep_params=None, stddevs=None, ang_mom_map=None, forward_ones=0, return_sym_multipliers=False):
    if ang_mom_map is None:
        if rep_params is None:
            raise Exception
        else:
            ang_mom_map=component_id_ang_mom_map(rep_params)
    prop_coeff_id_dict={}

    sym_multipliers=[]
    prop_coeff_id=[]

    if forward_ones != 0:
        last_prop_coeff=0
        for forward_id in range(forward_ones):
            prop_coeff_id.append(last_prop_coeff)
            sym_multipliers.append(1.0)
    else:
        last_prop_coeff=-1

    for sigma_id, ang_mom_classifier in enumerate(ang_mom_map):
        ang_mom1=ang_mom_classifier[0]
        ang_mom2=ang_mom_classifier[1]
        coup_mat_id=ang_mom_classifier[2]
        same_atom=ang_mom_classifier[3]

        if (same_atom and (ang_mom1 != ang_mom2)):
            cur_sym_mult=2.0
        else:
            cur_sym_mult=1.0
        sym_multipliers.append(cur_sym_mult)

        coup_mat_tuple=(coup_mat_id, same_atom)
        try:
            cur_prop_coeff_id=prop_coeff_id_dict[coup_mat_tuple]
        except KeyError:
            last_prop_coeff+=1
            cur_prop_coeff_id=last_prop_coeff
            prop_coeff_id_dict[coup_mat_tuple]=cur_prop_coeff_id
        prop_coeff_id.append(cur_prop_coeff_id)

    coup_mat_prop_coeffs=np.zeros((last_prop_coeff+1,))
    norm_coeffs=np.zeros((last_prop_coeff+1,))
    if stddevs is None:
        coup_mat_prop_coeffs[:]=1.0
    else:
        if forward_ones == 0:
            coup_mat_true_start=0
        else:
            coup_mat_true_start=1
            coup_mat_prop_coeffs[0]=1.0
        for sigma_id, stddev in enumerate(stddevs):
            true_sigma_id=sigma_id+forward_ones
            cur_sym_coeff=sym_multipliers[true_sigma_id]
            cur_prop_coeff_id=prop_coeff_id[true_sigma_id]
            norm_coeffs[cur_prop_coeff_id]+=cur_sym_coeff
            coup_mat_prop_coeffs[cur_prop_coeff_id]+=stddev**2*cur_sym_coeff
        coup_mat_prop_coeffs[coup_mat_true_start:]=np.sqrt(coup_mat_prop_coeffs[coup_mat_true_start:]/norm_coeffs[coup_mat_true_start:])

    if return_sym_multipliers:
        return coup_mat_prop_coeffs, prop_coeff_id, sym_multipliers
    else:
        return coup_mat_prop_coeffs, prop_coeff_id


# Choose hyperparameters for FJK based on a procedure that sorts vector components by the coupling matrix and angular momentum
# they correspond to. 
class Ang_mom_classified_rhf(Reduced_hyperparam_func):
    def __init__(self, rep_params=None, stddevs=None, ang_mom_map=None, use_Gauss=False):
        if ang_mom_map is None:
            if rep_params is None:
                raise Exception("No rep_params defined for Ang_mom_classified_rhf class.")
            else:
                ang_mom_map=component_id_ang_mom_map(rep_params)

        self.num_simple_log_params=1
        if use_Gauss:
            self.num_simple_log_params+=1
        self.num_full_params=self.num_simple_log_params+len(ang_mom_map)

        self.coup_mat_prop_coeffs, self.prop_coeff_id, self.sym_multipliers=matrix_grouped_prop_coeffs(ang_mom_map=ang_mom_map, stddevs=stddevs,
                                                                        forward_ones=self.num_simple_log_params, return_sym_multipliers=True)

        red_param_id_dict={}

        last_red_param=self.num_simple_log_params-1

        self.reduced_param_id_lists=[]

        for simple_log_param_id in range(self.num_simple_log_params):
            self.reduced_param_id_lists.append([simple_log_param_id])

        for sigma_id, ang_mom_classifier in enumerate(ang_mom_map):
            ang_mom1=ang_mom_classifier[0]
            ang_mom2=ang_mom_classifier[1]
            same_atom=ang_mom_classifier[3]

            self.reduced_param_id_lists.append([])
            for cur_ang_mom in [ang_mom1, ang_mom2]:
                ang_mom_tuple=(cur_ang_mom, same_atom)
                if ang_mom_tuple in red_param_id_dict:
                    cur_red_param_id=red_param_id_dict[ang_mom_tuple]
                else:
                    last_red_param+=1
                    cur_red_param_id=last_red_param
                    red_param_id_dict[ang_mom_tuple]=cur_red_param_id
                self.reduced_param_id_lists[-1].append(cur_red_param_id)

        self.num_reduced_params=last_red_param+1

    def reduced_params_to_full(self, reduced_parameters):
        output=np.repeat(1.0, self.num_full_params)
        for param_id in range(self.num_full_params):
            for red_param_id in self.reduced_param_id_lists[param_id]:
                # TO-DO does this choice of sym_multipliers make sense?
                output[param_id]*=np.sqrt(self.coup_mat_prop_coeffs[self.prop_coeff_id[param_id]]/
                                    np.sqrt(self.sym_multipliers[param_id]))*np.exp(reduced_parameters[red_param_id])
        return output

    def full_derivatives_to_reduced(self, full_derivatives, full_parameters):
        output=np.zeros((self.num_reduced_params,))
        for param_id, (param_der, param_val) in enumerate(zip(full_derivatives, full_parameters)):
            for red_param_id in self.reduced_param_id_lists[param_id]:
                output[red_param_id]+=param_der*param_val
        return output

    def full_params_to_reduced(self, full_parameters):
        output=np.zeros((self.num_reduced_params,))
        for param_id, param_val in enumerate(full_parameters):
            red_param_id_list=self.reduced_param_id_lists[param_id]
            if (len(red_param_id_list)==2):
                if red_param_id_list[0] != red_param_id_list[1]:
                    continue
            output[red_param_id_list[0]]=np.log(param_val*np.sqrt(self.sym_multipliers[param_id])/self.coup_mat_prop_coeffs[self.prop_coeff_id[param_id]])/len(red_param_id_list)
        return output

    def initial_reduced_parameter_guess(self, init_lambda, *other_args):
        output=np.zeros((self.num_reduced_params,))
        output[0]=np.log(init_lambda)
        return output

    def __str__(self):
        vals_names={"num_reduced_params" : self.num_reduced_params,
                    "reduced_param_id_lists" : self.reduced_param_id_lists,
                    "sym_multipliers" : self.sym_multipliers,
                    "coup_mat_prop_coeffs" : self.coup_mat_prop_coeffs,
                    "prop_coeff_id" : self.prop_coeff_id}
        return self.str_output_dict("Ang_mom_classified_rhf", vals_names)

