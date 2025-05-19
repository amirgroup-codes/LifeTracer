import itertools
import os
import numpy as np
import pandas as pd
import random as python_random
from loguru import logger
from sklearn.linear_model import LogisticRegression

from .utils.heatmap_utils import create_folder_if_not_exists


def parameters_selection(config):
    np.random.seed(config["seed"])
    python_random.seed(config["seed"])

    parameters_selection_path = config["parameters_selection_path"]
    log_path = os.path.join(parameters_selection_path, 'parameters_selection.log')
    logger.add(log_path, rotation="10 MB")
    
    Cs = config["C"]
    lam1 = config["lambda1"]
    lam2 = config["lambda2"]
    rt1_threshold = config["rt1_threshold"]
    rt2_threshold = config["rt2_threshold"]

    params_to_search = list(itertools.product(lam1,lam2,rt1_threshold,rt2_threshold,Cs))

    samples = pd.read_csv(config["labels_path"])
    sample_indices = np.arange(0,len(samples))

    create_folder_if_not_exists(parameters_selection_path)

    log = []
    for param in params_to_search:
        lam1 = param[0]
        lam2 = param[1]
        rt1_th = param[2]
        rt2_th = param[3]
        C = param[4]
        features = np.load(os.path.join(config['features_path'], f'features_lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_th}_rt2th_{rt2_th}.npy'))

        correct_val = 0
        for val_id in sample_indices:
            train_index = np.delete(sample_indices,np.where(sample_indices == val_id))

            X_val = features[val_id].reshape(1,-1)
            y_val = samples[config['label_column_name']].to_numpy()[val_id].reshape(1,-1)

            X_train = features[train_index]
            y_train = samples[config['label_column_name']].to_numpy()[train_index]
            
            lr = LogisticRegression(penalty='l2', solver='liblinear', C=C,random_state=config['seed'])
        
            # Fit the classifier to the data
            lr.fit(X_train, y_train)

            # Make predictions on the validation set and apply threshold
            prediction = lr.predict(X_val)

            if prediction == y_val:
                correct_val += 1
        
        logger.info(f'lr l2 - Validation Accuracy: {correct_val/(sample_indices.shape[0])}, lam1 : {lam1}, lam2 : {lam2}, rt1_threshold = {rt1_th}, rt2_threshold = {rt2_th}, C: {C}')
        
        log.append((C,lam1,lam2,rt1_th,rt2_th,correct_val/(sample_indices.shape[0])))

    pd.DataFrame(log,columns=['C','lam1','lam2','rt1_th','rt2_th','val_acc']).to_csv(os.path.join(parameters_selection_path,'parameters_selection.csv'))



