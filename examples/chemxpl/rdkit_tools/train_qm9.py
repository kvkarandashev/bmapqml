import numpy as np
import random
import pandas as pd
from rdkit_descriptors import *
from bmapqml.krr import KRR
from bmapqml.utils import dump2pkl
from bmapqml.data import conversion_coefficient
from datetime import date
random.seed(1337)
np.random.seed(1337)

if __name__ == "__main__":

    N=[4000, 8000, 16000, 32000, 64000]
    N_hyperparam_opt=12000
    N_test=16000

    max_train_size=max(N)

    TARGET_PROPERTY = 'gap'
    PATH = '/store/common/jan/qm9_removed/qm9/'
    qm9_data = process_qm9(PATH)


    
    SMILES, y  = qm9_data['canon_smiles'].values, np.float_(qm9_data[TARGET_PROPERTY].values)*conversion_coefficient["au_eV"]
    inds = random.sample(range(len(SMILES)), max_train_size+N_test)

    SMILES, y = SMILES[inds], y[inds]

    X = get_all_FP(SMILES, fp_type="both")

    X_train=X[:max_train_size]
    X_test=X[max_train_size:]

    y_train=y[:max_train_size]
    y_test=y[max_train_size:]

    reg=KRR(kernel_type="Laplacian", scale_labels=True)
    reg.optimize_hyperparameters(X_train[:N_hyperparam_opt], y_train[:N_hyperparam_opt])

    null_error = mae(y_test, np.mean(y_train))
    print('Nullmodel error:', null_error)
    lrn_crv      = []
    for n in N:
        reg.fit(X_train[:n], y_train[:n])
        y_pred = reg.predict(X_test)
        MAE    = mae(y_test, y_pred)

        print(n, MAE)

        lrn_crv.append(MAE)

    reg.save('/store/common/jan/qm9_removed/ml/KRR_{}_{}'.format(n, TARGET_PROPERTY))
    lrn_crv = np.array(lrn_crv)
    report = {}
    report["property"],report["nullmodel"], report["lrn_crv"], report["unit"],report["dataset_loc"],report["date"] = TARGET_PROPERTY,null_error,np.vstack((N,lrn_crv)), "eV", PATH, date.today()
    dump2pkl(report, "/store/common/jan/qm9_removed/ml/report_{}.pkl".format(TARGET_PROPERTY))
