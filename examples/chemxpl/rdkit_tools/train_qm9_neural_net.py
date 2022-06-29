import numpy as np
import random
import pandas as pd
from bmapqml.chemxpl import rdkit_descriptors
from bmapqml.krr import KRR
from bmapqml.utils import dump2pkl
from bmapqml.data import conversion_coefficient
from datetime import date
import torch
import torch.nn as nn 
import torch.nn as nn
import torch.nn.functional as F
import warnings
from sklearn.preprocessing import MinMaxScaler

random.seed(1337)
np.random.seed(1337)

#torch test if gpu available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU available")




def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
warnings.simplefilter('always')
import warnings
warnings.simplefilter("ignore")

class Net(nn.Module):

    def __init__(self, in_dim=None, n_neuro=None, lr=None, num_epochs=None, name=None,xtest=None,ytest=None, verbose=None):
        super(Net, self).__init__()

        self.in_dim      = in_dim or 8242
        self.n_neuo      = n_neuro or 300
        self.lr          = lr or 0.0002 #0.0003
        self.num_epochs  = num_epochs or 10
        self.name        = name or "Net"
        self.loss        = nn.MSELoss()
        self.verbose     = verbose or False
        self.ytest       = ytest
        self.xtest       = xtest

 
        act_fct  = torch.nn.ReLU() #torch.nn.SELU() #
        self.sequ_layer = torch.nn.Sequential(
            torch.nn.Linear(self.in_dim, self.n_neuo),
            torch.nn.BatchNorm1d(self.n_neuo),
            act_fct,
            torch.nn.Linear(self.n_neuo, self.n_neuo),
            act_fct,
            torch.nn.Linear(self.n_neuo, self.n_neuo),
            act_fct,
            torch.nn.Linear(self.n_neuo, self.n_neuo),
            act_fct,
            torch.nn.Linear(self.n_neuo, self.n_neuo),
            act_fct,
            torch.nn.Linear(self.n_neuo, self.n_neuo),
            act_fct,
            torch.nn.Linear(self.n_neuo, self.n_neuo),
            act_fct,
            torch.nn.Linear(self.n_neuo, self.n_neuo),
            act_fct,
            torch.nn.Linear(self.n_neuo, self.n_neuo),
            act_fct,
            torch.nn.Linear(self.n_neuo, self.n_neuo),
            act_fct,
            torch.nn.Linear(self.n_neuo, self.n_neuo),
            act_fct,
            torch.nn.Linear(self.n_neuo, 1)
        )

    
    def forward(self, x):
        net_pred = self.sequ_layer(x)
        return net_pred

    def fit(self, xtrain, ytrain): 
        """
        minibatch training
        """
        optimizer = torch.optim.Adam(self.parameters(), lr =self.lr )
        batch_size = 50 #512 # 128 # or whatever

        for ep in range(self.num_epochs):

            permutation = torch.randperm(xtrain.size()[0])

            for i in range(0,xtrain.size()[0], batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i+batch_size]
                batch_x, batch_y = xtrain[indices], ytrain[indices]
                y_pred = self.forward(batch_x)
                loss= self.loss(y_pred, batch_y) 
                loss.backward()
                optimizer.step()

            if self.verbose and ep%100==0:
                print('epoch {}, loss {}'.format(ep, loss.item()))


            valid_loss = self.loss(self.ytest, self.forward(self.xtest))
            if loss < valid_loss*0.001: #valid_loss*0.0001:.
                print(loss)
                break



    def predict(self, X_test):
        return self(X_test)

    def save(self):
        torch.save(self, self.name)




if __name__ == "__main__":

    N= [4000]
    #[4000, 8000, 16000, 32000, 64000,102000]
    N_hyperparam_opt=6000
    N_test= 16000

    max_train_size=max(N)

    TARGET_PROPERTY = 'mu'
    PATH = '/store/common/jan/qm9_removed/qm9/'
    qm9_data = rdkit_descriptors.process_qm9(PATH, all=True)


    SMILES, y  = qm9_data['canon_smiles'].values, np.float_(qm9_data[TARGET_PROPERTY].values) #*conversion_coefficient["au_eV"]
    print(np.mean(y))
    
    inds = random.sample(range(len(SMILES)), max_train_size+N_test)

    SMILES, y = SMILES[inds], y[inds]

    y = y.reshape(-1,1)
    X = rdkit_descriptors.get_all_FP(SMILES, fp_type="both")
    print(X.shape)

    #exit()
    X_train=X[:max_train_size]
    X_test=X[max_train_size:]

    y_train=y[:max_train_size]
    y_test=y[max_train_size:]




    sc          = MinMaxScaler()
    sct         = MinMaxScaler()
    X_train     = sc.fit_transform(X_train)
    X_test      = sc.transform(X_test)
    y_train     = sct.fit_transform(y_train)
    y_test      = sct.transform(y_test)
    X_train     = torch.from_numpy(X_train.astype(np.float32))
    X_test      = torch.from_numpy(X_test.astype(np.float32))
    y_train     = torch.from_numpy(y_train.astype(np.float32))
    y_test      = torch.from_numpy(y_test.astype(np.float32)) 

    model = Net(n_neuro=200,in_dim=X.shape[1], xtest=X_test.cuda(),ytest=y_test.cuda(), num_epochs=500,verbose=True) #, name="./networks/NET_{}".format(n))
    model.to(torch.device("cuda"))
    model.fit(X_train.cuda(), y_train.cuda())
    
    predictions = sct.inverse_transform(np.array(model.predict(X_test.cuda()).detach().cpu()))
    MAE = rdkit_descriptors.mae(sct.inverse_transform(y_test), predictions)
    print("#",len(X_train), MAE)




    # measure the time needed to execute a block of code
    start = time.time()
    for i in range(100):
        model.forward(X_test)
    end = time.time()
    print("Time needed: {}".format(end-start))





    exit()
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
    report["property"],report["nullmodel"], report["lrn_crv"], report["unit"],report["dataset_loc"],report["date"] = TARGET_PROPERTY,null_error,np.vstack((N,lrn_crv)), "debye", PATH, date.today()
    dump2pkl(report, "/store/common/jan/qm9_removed/ml/report_{}.pkl".format(TARGET_PROPERTY))