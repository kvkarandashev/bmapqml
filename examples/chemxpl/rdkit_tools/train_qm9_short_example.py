import numpy as np
from sklearn.model_selection import train_test_split
import random
import pandas as pd
from rdkit_descriptors import *
from bmapqml.krr import KRR
from bmapqml.data import conversion_coefficient
random.seed(1337)
np.random.seed(1337)

if __name__ == "__main__":

    N=[400, 800, 1600]
    N_hyperparam_opt=400
    N_test=400

    max_train_size=max(N)

    process= True
    TARGET_PROPERTY = 'gap'

    if process:
        df = process_qm9('/store/common/jan/qm9/')
    else:
        df = pd.read_csv('qm9.csv')

    
    SMILES, y  = df['canon_smiles'].values, np.float_(df[TARGET_PROPERTY].values)*conversion_coefficient["au_eV"]

    inds = random.sample(range(len(SMILES)), max_train_size+N_test)

    SMILES, y = SMILES[inds], y[inds]

    #[32768] #uses only single core why?
    X = get_all_FP(SMILES, fp_type="both")

    X_train=X[:max_train_size]
    X_test=X[max_train_size:]

    y_train=y[:max_train_size]
    y_test=y[max_train_size:]

    """
    Nullmodel
    """
    reg=KRR(kernel_type="Laplacian", scale_labels=True)
    reg.optimize_hyperparameters(X_train[:N_hyperparam_opt], y_train[:N_hyperparam_opt])

    null_error = mae(y_test, np.mean(y_train))
    print('Nullmodel error:', null_error)
    lrn_crv      = []

    errors = []
    for n in N:
        reg.fit(X_train[:n], y_train[:n])
        y_pred = reg.predict(X_test)
        MAE    = mae(y_test, y_pred)

        print(n, MAE)

        errors.append(MAE)

    reg.save('/store/common/jan/qm9/KRR_{}_{}'.format(n, TARGET_PROPERTY))
    print("lc",errors)
