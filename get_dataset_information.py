import os
import pdb
from time import time
import pandas as pd
import numpy as np

def main():
    file_list = [
        # 'GM4',    
        # 'balance',
        # 'banana',
        # 'biodeg',
        # 'blood',
        # 'breast-cancer',
        # 'breast-tissue',
        # 'bupa',
        # 'cleveland',
        # 'conn-bench-vowel',
        # 'contraceptive',
        # 'dermatology',
        # 'fertility',
        # 'glass',
        # 'haberman',
        # 'hayes-roth',
        # 'hepatitis',
        # 'led7digit_new',
        # 'libras',
        # 'madelon',
        # 'mammographic',
        # 'marketing',
        # 'monk-2_new',
        # 'multiple-features',
        # 'musk1',
        # 'musk2',
        # 'newthyroid',
        # 'page-blocks',
        # 'pima',
        # 'ring1',
        # 'sonar',
        # 'spambase',
        # 'tae',
        # 'tic-tac-toe',
        # 'titanic_new',
        # 'twonorm1',
        # 'vehicle',
        # 'vertebral_3C',
        # 'waveform_w_noise',
        # 'waveform_wo_noise',
        # 'wdbc',
        # 'wine',
    
        'balance',
        'banana',
        'breast-tissue',
        'cleveland',
        'Colon',
        'conn-bench-vowel',
        'contraceptive',
        'electricity-normalized',
        'Embryonal',
        'fertility',
        'GM4',  
        'heart',
        'isolet',
        'Leukemia',
        'madelon',
        'mammographic',
        'multiple-features',
        'musk1',
        'musk2',
        'newthyroid',
        'penbased_new_fix',
        'phoneme',
        'plant_margin',
        'ring1',
        'satimage',
        'sonar',
        'tic-tac-toe',
        'titanic_new',        
        'vertebral_3C',
        'wine_red',      
        
    ]
    data_folder = r'C:\TRUONG\PhD-code\OptimalConnectionPaper2023\Code\train1_val_test'
    n_file = len(file_list)
    info = np.zeros((n_file, 3))
    for file_idx, file_name in enumerate(file_list):
        D_train = np.loadtxt(os.path.join(data_folder, 'train1', file_name + '_train1.dat'), delimiter=',')
        D_val = np.loadtxt(os.path.join(data_folder, 'val', file_name + '_val.dat'), delimiter=',')
        D_test = np.loadtxt(os.path.join(data_folder, 'test', file_name + '_test.dat'), delimiter=',')

        X_train = D_train[:, :-1]
        y_train = D_train[:, -1].astype(np.int32)
        X_val = D_val[:, :-1]
        y_val = D_val[:, -1].astype(np.int32)
        X_test = D_test[:, :-1]
        y_test = D_test[:, -1].astype(np.int32)

        classes = np.unique(np.concatenate((y_train, y_val, y_test)))
        if np.any(classes.astype(np.int32) == 0):
            raise Exception("Labels have to start from 1")

        n_classes = np.size(classes)
        n_inst = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        n_dim = X_train.shape[1]

        info[file_idx, 0] = n_inst
        info[file_idx, 1] = n_dim
        info[file_idx, 2] = n_classes
        
    df = pd.DataFrame(
        data = info,
        index=file_list,
        columns=["Number of instances", "Number of dimensions", "Number of classes"]
    )
    df.to_excel("DeepEnsembleEvoConnection_dataset_information.xlsx")

if __name__ == '__main__':
    main()   
    
    