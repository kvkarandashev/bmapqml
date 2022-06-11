import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import numpy as np
import rdkit
import numpy as np
import pandas as pd
from rdkit import Chem  
import pickle as pickle1
from rdkit_descriptors import *
from bmapqml.kernel_class import KRR
import pdb
random.seed(1337)
np.random.seed(1337)

if __name__ == "__main__":


    process= True
    TARGET_PROPERTY = 'atomization'

    if process:
        df = process_qm9('/store/common/jan/qm9/')
    else:
        df = pd.read_csv('qm9.csv')

    
    SMILES, y  = df['canon_smiles'].values, np.float_(df[TARGET_PROPERTY].values)*27
    inds = np.arange(len(SMILES))
    np.random.shuffle(inds)
    SMILES, y = SMILES[inds], y[inds]
                    
    N = [20000]
    #[32768] #uses only single core why?
    X = get_all_FP(SMILES, fp_type="both")

    X_train, X_test, y_train, y_test = train_test_split(X,   y, random_state=1337, test_size=0.20, shuffle=True)
    
    #pdb.set_trace()
    """
    Nullmodel
    """

    null_error = mae(y_test, np.mean(y_train))
    print('Nullmodel error:', null_error)
    lrn_crv      = []

    errors = []
    for n in N:
        reg = KRR(kernel_type="laplacian", scale_features=True)
        reg.fit(X_train[:n], y_train[:n])
        y_pred = reg.predict(X_test)
        MAE    = mae(y_test, y_pred)

        print(n, MAE)

        errors.append(MAE)
       # pdb.set_trace()

    reg.save('/store/common/jan/qm9/KRR_{}_{}'.format(n, TARGET_PROPERTY))
    print(errors)
