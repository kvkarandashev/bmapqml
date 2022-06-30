"""
A version of KRR class that uses JAX for faster kernel evaluation.
Since JAX currently does not support parallelization Fortran is preferrable for training.
"""
from .utils import OptionUnavailableError
from .krr import KRR
import numpy as np
import jax.numpy as jnp
import jax.config as jconfig
from jax import jit, vmap
from .test_utils import timestamp

def laplacian_kernel_product(sigma_params, X1, X2):
    return jnp.exp(-jnp.sum(jnp.abs(X1-X2))*sigma_params[0])

def gaussian_kernel_product(sigma_params, X1, X2):
    return jnp.exp(-jnp.sum((X1-X2)**2)*sigma_params[0])

def krr_eval_function(kernel_product_func, sigma_params, alphas, X_train, X_t):
    return jnp.dot(alphas, vmap(kernel_product_func, in_axes=(None, 0, None))(sigma_params, X_train, X_t))

kernel_products={"Laplacian" : laplacian_kernel_product, "Gaussian" : gaussian_kernel_product}

class KRR_JAX(KRR):

    def __init__(self, KRR_model, kernel_type):
        
        """
        KRR_model : the already trained KRR model that needs to be accelerated
        kernel_type : type of the kernel function used in KRR model
        """
        # I decided against automatically importing kernel_type from KRR_model
        # for risk of accidental mistakes.
        self.kernel_type    = kernel_type
        self.scale_features = KRR_model.scale_features
        self.scale_labels   = KRR_model.scale_labels



        self.sigmas=KRR_model.sigmas

        if self.kernel_type == "Laplacian":
            self.sigma_params=jnp.array(self.sigmas**(-1))
        if self.kernel_type == "Gaussian":
            self.sigma_params=jnp.array(.5*self.sigmas**(-2))

        self.alphas=jnp.array(KRR_model.alphas)
        self.X_train=jnp.array(KRR_model.X_train)
        # TODO remember why I put x64 here?
        jconfig.update("jax_enable_x64", True)
        self.kernel_product=kernel_products[self.kernel_type]

    def predict(self, X_test):

        """
        Predict the labels of the test data X_test 
        using the trained model.

        X_test: numpy array
        """

        if self.scale_features:
            X_test = self.X_MinMaxScaler(X_test)

        y_pred=[jit(krr_eval_function, static_argnums=(0,))(self.kernel_product, self.sigma_params, self.alphas, self.X_train, X_t) for X_t in X_test]

        if self.scale_labels:
            y_pred = self.y_MinMaxScaler_inverse(y_pred)

        return y_pred


    def prediction_difference(self, KRR_model, X_test):
        """
        Check the difference between output of the original KRR_model object and the new one.
        (If the difference is large first of all check whether self.kernel_type is correct.)
        KRR_model : compared model
        X_test : tested input
        """
        timestamp("Making predictions with the original model.")
        y_benchmark=KRR_model.predict(X_test)
        timestamp("Finished.")
        timestamp("Making predictions without optimizing number of python calls.")
        y_benchmark_other=[KRR_model.predict(X_test[i:i+1])[0] for i in range(X_test.shape[0])]
        timestamp("Finished.")
        print("Standard deviation of the predictions:", np.std(y_benchmark))
        print("Difference between two modes of evaluating old model:", np.mean(np.abs(np.array(y_benchmark_other)-y_benchmark)))
        timestamp("Dummy precompilation run with JAX")
        y_dummy=self.predict(X_test[:1])
        timestamp("Finished.")
        timestamp("Making predictions with JAX")
        y_new=self.predict(X_test)
        timestamp("Finished.")
        print("Average prediction difference:", np.mean(np.abs(np.array(y_new)-y_benchmark)))
