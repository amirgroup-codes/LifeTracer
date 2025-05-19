import itertools
import os
import numpy as np
import concurrent.futures
import random as python_random
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from loguru import logger
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from tqdm import tqdm
from .utils.heatmap_utils import create_folder_if_not_exists

def process_seed_lr(seed, params_combination, config):
    model = 'l2' if config['model'] == 'lr_l2' else 'l1'
    # solver = 'saga' if model == 'l2' else 'liblinear'  # saga is faster for l2, must use liblinear for l1
    solver = 'liblinear'
    
    np.random.seed(seed)
    python_random.seed(seed)
    kf = KFold(n_splits=9,shuffle=True, random_state=seed)

    samples = pd.read_csv(config['labels_path'])
    features_combined = np.zeros((len(samples),2))

    # Cache for features to avoid repeated disk reads
    features_cache = {}

    log = []
    for rest_index, test_index in kf.split(features_combined):
       
        for param in tqdm(params_combination, desc=f'Seed {seed} - Processing parameters'):
            lam1 = param[0]
            lam2 = param[1]
            rt1_th = param[2]
            rt2_th = param[3]
            C = param[4]

            # Use cached features if available
            feature_key = f'lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_th}_rt2th_{rt2_th}'
            if feature_key not in features_cache:
                features_cache[feature_key] = np.load(os.path.join(config['features_path'], f'features_{feature_key}.npy'))
            features = features_cache[feature_key]

            correct_val = 0
            for val_id in rest_index:
                train_index = np.delete(rest_index,np.where(rest_index == val_id))

                X_val = features[val_id].reshape(1,-1)
                y_val = samples[config['label_column_name']].to_numpy()[val_id].reshape(1,-1)
                X_train = features[train_index]
                y_train = samples[config['label_column_name']].to_numpy()[train_index]
                
                lr = LogisticRegression(
                    penalty=model,
                    solver=solver,
                    C=C,
                    random_state=seed,
                )

                # Fit the classifier to the data
                lr.fit(X_train, y_train)

                # Make predictions on the test set and apply threshold
                prediction = lr.predict(X_val)

                if prediction == y_val:
                    correct_val += 1
            
            # Train on all training set
            lr = LogisticRegression(
                penalty=model,
                solver=solver,
                C=C,
                random_state=seed,
            )
            lr.fit(features[rest_index], samples[config['label_column_name']].to_numpy()[rest_index])

            X_test = features[test_index]
            y_test = samples[config['label_column_name']].to_numpy()[test_index]
            predictions_test = lr.predict(X_test)
            test_acc = accuracy_score(y_test, predictions_test)
            log.append((lam1,lam2,rt1_th,rt2_th,correct_val/(rest_index.shape[0]),test_acc,test_index,C))

            # Clear some memory if cache is too large
            if len(features_cache) > 10:  # Keep only last 10 feature sets
                oldest_key = next(iter(features_cache))
                del features_cache[oldest_key]

    pd.DataFrame(log,columns=['lam1','lam2','rt1_threshold','rt2_threshold','val_acc','test_acc','test_id','C']).to_csv(os.path.join(config['eval_path'],f'eval_seed_{seed}.csv'),index=False)

def process_seed_svm(seed, params_combination, config):
    np.random.seed(seed)
    python_random.seed(seed)
    kf = KFold(n_splits=9,shuffle=True, random_state=seed)

    samples = pd.read_csv(config['labels_path'])
    features_combined = np.zeros((len(samples),2))

    # Cache for features to avoid repeated disk reads
    features_cache = {}

    log = []
    for rest_index, test_index in kf.split(features_combined):
       
        for param in tqdm(params_combination, desc=f'Seed {seed} - Processing parameters'):
            lam1 = param[0]
            lam2 = param[1]
            rt1_th = param[2]
            rt2_th = param[3]
            C = param[4]
            kernel = param[5]

            # Use cached features if available
            feature_key = f'lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_th}_rt2th_{rt2_th}'
            if feature_key not in features_cache:
                features_cache[feature_key] = np.load(os.path.join(config['features_path'], f'features_{feature_key}.npy'))
            features = features_cache[feature_key]

            correct_val = 0
            for val_id in rest_index:
                train_index = np.delete(rest_index,np.where(rest_index == val_id))

                X_val = features[val_id].reshape(1,-1)
                y_val = samples[config['label_column_name']].to_numpy()[val_id].reshape(1,-1)
                X_train = features[train_index]
                y_train = samples[config['label_column_name']].to_numpy()[train_index]
                
                # Optimize SVM configuration for speed
                svc = svm.SVC(
                    kernel=kernel,
                    C=C,
                    random_state=seed,
                )

                # Fit the classifier to the data
                svc.fit(X_train, y_train)

                # Make predictions on the test set and apply threshold
                prediction = svc.predict(X_val)

                if prediction == y_val:
                    correct_val += 1
            
            # Train on all training set
            svc = svm.SVC(
                kernel=kernel,
                C=C,
                random_state=seed,
            )
            svc.fit(features[rest_index], samples[config['label_column_name']].to_numpy()[rest_index])

            X_test = features[test_index]
            y_test = samples[config['label_column_name']].to_numpy()[test_index]
            predictions_test = svc.predict(X_test)
            test_acc = accuracy_score(y_test, predictions_test)
            log.append((lam1,lam2,rt1_th,rt2_th,correct_val/(rest_index.shape[0]),test_acc,test_index,C))

            # Clear some memory if cache is too large
            if len(features_cache) > 10:  # Keep only last 10 feature sets
                oldest_key = next(iter(features_cache))
                del features_cache[oldest_key]

    pd.DataFrame(log,columns=['lam1','lam2','rt1_threshold','rt2_threshold','val_acc','test_acc','test_id','C']).to_csv(os.path.join(config['eval_path'],f'eval_seed_{seed}.csv'),index=False)

def process_seed_rf(seed, params_combination, config):
    np.random.seed(seed)
    python_random.seed(seed)
    kf = KFold(n_splits=9,shuffle=True, random_state=seed)

    samples = pd.read_csv(config['labels_path'])
    features_combined = np.zeros((len(samples),2))

    # Cache for features to avoid repeated disk reads
    features_cache = {}

    log = []
    for rest_index, test_index in kf.split(features_combined):
        for param in tqdm(params_combination, desc=f'Seed {seed} - Processing parameters'):
            lam1 = param[0]
            lam2 = param[1]
            rt1_th = param[2]
            rt2_th = param[3]
            n_estimators = param[4]

            # Use cached features if available
            feature_key = f'lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_th}_rt2th_{rt2_th}'
            if feature_key not in features_cache:
                features_cache[feature_key] = np.load(os.path.join(config['features_path'], f'features_{feature_key}.npy'))
            features = features_cache[feature_key]

            correct_val = 0
            for val_id in rest_index:
                train_index = np.delete(rest_index,np.where(rest_index == val_id))

                X_val = features[val_id].reshape(1,-1)
                y_val = samples[config['label_column_name']].to_numpy()[val_id].reshape(1,-1)
                X_train = features[train_index]
                y_train = samples[config['label_column_name']].to_numpy()[train_index]
                
                # Optimize RandomForest configuration for speed
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=seed,
                    n_jobs=-1,  # Use all CPU cores
                    max_features='sqrt',  # Faster feature selection
                    bootstrap=True,  # Enable bootstrapping for better parallelization
                )

                # Fit the classifier to the data
                rf.fit(X_train, y_train)

                # Make predictions on the test set and apply threshold
                prediction = rf.predict(X_val)

                if prediction == y_val:
                    correct_val += 1
            
            # Train on all training set
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=seed,
                n_jobs=-1,
                max_features='sqrt',
                bootstrap=True,
            )
            rf.fit(features[rest_index], samples[config['label_column_name']].to_numpy()[rest_index])

            X_test = features[test_index]
            y_test = samples[config['label_column_name']].to_numpy()[test_index]
            predictions_test = rf.predict(X_test)
            test_acc = accuracy_score(y_test, predictions_test)
            log.append((lam1,lam2,rt1_th,rt2_th,correct_val/(rest_index.shape[0]),test_acc,test_index,n_estimators))

            # Clear some memory if cache is too large
            if len(features_cache) > 10:  # Keep only last 10 feature sets
                oldest_key = next(iter(features_cache))
                del features_cache[oldest_key]

            logger.info(f'lam1: {lam1}, lam2: {lam2}, rt1_threshold: {rt1_th}, rt2_threshold: {rt2_th}, n_estimators: {n_estimators}, test_acc: {test_acc}')
    pd.DataFrame(log,columns=['lam1','lam2','rt1_threshold','rt2_threshold','val_acc','test_acc','test_id', 'n_estimators']).to_csv(os.path.join(config['eval_path'],f'eval_seed_{seed}.csv'),index=False)

def process_seed_xgboost(seed, params_combination, config):
    # Ensure reproducibility
    np.random.seed(seed)
    python_random.seed(seed)

    # Define KFold cross-validation
    kf = KFold(n_splits=9, shuffle=True, random_state=seed)

    # Load your data
    samples = pd.read_csv(config['labels_path'])
    features_combined = np.zeros((len(samples), 2))

    # Initialize log
    log = []

    # Perform KFold cross-validation
    for rest_index, test_index in kf.split(features_combined):
        for param in params_combination:
            lam1 = param[0]
            lam2 = param[1]
            rt1_th = param[2]
            rt2_th = param[3]
            n_estimators = param[4]

            features = np.load(os.path.join(config['features_path'], f'features_lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_th}_rt2th_{rt2_th}.npy'))

            correct_val = 0
            for val_id in rest_index:
                train_index = np.delete(rest_index, np.where(rest_index == val_id))

                X_val = features[val_id].reshape(1, -1)
                y_val = samples[config['label_column_name']].to_numpy()[val_id].reshape(1, -1)
                X_train = features[train_index]
                y_train = samples[config['label_column_name']].to_numpy()[train_index]

                xgb = XGBClassifier(n_estimators=n_estimators, random_state=seed)

                # Fit the classifier to the data
                xgb.fit(X_train, y_train)

                # Make predictions on the validation set
                prediction = xgb.predict(X_val)

                if prediction == y_val:
                    correct_val += 1

            # Train on all training set
            xgb = XGBClassifier(n_estimators=n_estimators, random_state=seed)
            xgb.fit(features[rest_index], samples[config['label_column_name']].to_numpy()[rest_index])

            X_test = features[test_index]
            y_test = samples[config['label_column_name']].to_numpy()[test_index]
            predictions_test = xgb.predict(X_test)
            test_acc = accuracy_score(y_test, predictions_test)
            log.append((lam1, lam2, rt1_th, rt2_th, correct_val / rest_index.shape[0], test_acc, test_index, n_estimators))

            logger.info(f'lam1: {lam1}, lam2: {lam2}, rt1_threshold: {rt1_th}, rt2_threshold: {rt2_th}, n_estimators: {n_estimators}, test_acc: {test_acc}')

    # Save the log to a CSV file
    pd.DataFrame(log, columns=['lam1', 'lam2', 'rt1_threshold', 'rt2_threshold', 'val_acc', 'test_acc', 'test_id', 'n_estimators']).to_csv(os.path.join(config['eval_path'], f'eval_seed_{seed}.csv'), index=False)


def process_seed_NB(seed, params_combination, config):
    np.random.seed(seed)
    python_random.seed(seed)
    kf = KFold(n_splits=9,shuffle=True, random_state=seed)

    samples = pd.read_csv(config['labels_path'])
    features_combined = np.zeros((len(samples),2))

    # Cache for features to avoid repeated disk reads
    features_cache = {}

    log = []
    for rest_index, test_index in kf.split(features_combined):
        for param in tqdm(params_combination, desc=f'Seed {seed} - Processing parameters'):
            lam1 = param[0]
            lam2 = param[1]
            rt1_th = param[2]
            rt2_th = param[3]
            alpha = param[4]

            # Use cached features if available
            feature_key = f'lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_th}_rt2th_{rt2_th}'
            if feature_key not in features_cache:
                features_cache[feature_key] = np.load(os.path.join(config['features_path'], f'features_{feature_key}.npy'))
            features = features_cache[feature_key]

            correct_val = 0
            for val_id in rest_index:
                train_index = np.delete(rest_index,np.where(rest_index == val_id))

                X_val = features[val_id].reshape(1,-1)
                y_val = samples[config['label_column_name']].to_numpy()[val_id].reshape(1,-1)
                X_train = features[train_index]
                y_train = samples[config['label_column_name']].to_numpy()[train_index]
                
                nb = BernoulliNB(
                    alpha=alpha,
                    force_alpha=True,  # Ensures alpha is used even with sparse data
                    binarize=None,  # Data is already binary
                )

                # Fit the classifier to the data
                nb.fit(X_train, y_train)

                # Make predictions on the test set and apply threshold
                prediction = nb.predict(X_val)

                if prediction == y_val:
                    correct_val += 1
            
            # Train on all training set
            nb = BernoulliNB(
                alpha=alpha,
                force_alpha=True,
                binarize=None,
            )
            nb.fit(features[rest_index], samples[config['label_column_name']].to_numpy()[rest_index])

            X_test = features[test_index]
            y_test = samples[config['label_column_name']].to_numpy()[test_index]
            predictions_test = nb.predict(X_test)
            test_acc = accuracy_score(y_test, predictions_test)
            log.append((lam1,lam2,rt1_th,rt2_th,correct_val/(rest_index.shape[0]),test_acc,test_index,alpha))

            # Clear some memory if cache is too large
            if len(features_cache) > 10:  # Keep only last 10 feature sets
                oldest_key = next(iter(features_cache))
                del features_cache[oldest_key]

            logger.info(f'lam1: {lam1}, lam2: {lam2}, rt1_threshold: {rt1_th}, rt2_threshold: {rt2_th}, alpha: {alpha}, test_acc: {test_acc}')
    pd.DataFrame(log,columns=['lam1','lam2','rt1_threshold','rt2_threshold','val_acc','test_acc','test_id', 'alpha']).to_csv(os.path.join(config['eval_path'],f'eval_seed_{seed}.csv'),index=False)

def calc_accuracy(config):
    labels = pd.read_csv(config['labels_path'])
    test = np.ones((len(labels),2))

    num_seeds = 10
    np.random.seed(42)
    seeds = np.random.choice(range(1, 1001), size=num_seeds, replace=False)

    avg_val_per_seed = []
    avg_test_per_seed = []
    std_test_per_seed = []

    for seed in seeds:
        val_acc = []
        test_acc = []

        result = pd.read_csv(os.path.join(config['eval_path'],f'eval_seed_{seed}.csv'),skipinitialspace=True)

        kf = KFold(n_splits=9,shuffle=True, random_state=seed)

        for rest_index, test_index in kf.split(test):

            test_id = f'{test_index}'

            rs = result[result['test_id'] == test_id]
            rs = rs.sort_values(by=['val_acc','rt2_threshold'], ascending=[False, True]).reset_index(drop=True)

            row_with_max_val_acc = rs.iloc[0]

            val_acc.append(row_with_max_val_acc['val_acc'])
            test_acc.append(row_with_max_val_acc['test_acc'])

        avg_val_per_seed.append(np.array(val_acc).mean()*100)
        avg_test_per_seed.append(np.array(test_acc).mean()*100)

        std_test_per_seed.append(np.array(test_acc).std()*100)
    logger.info(f'avg validation acc: {np.array(avg_val_per_seed).mean()}±{np.array(avg_val_per_seed).std()}')
    logger.info(f'avg test acc: {np.array(avg_test_per_seed).mean()}±{np.array(avg_test_per_seed).std()}')

    print(avg_test_per_seed)
    print(std_test_per_seed)

def eval(config):
    model = config['model']

    log_path = os.path.join(config['eval_path'], 'eval.log')
    logger.add(log_path, rotation="10 MB")

    lam1 = config[model]['lambda1']
    lam2 = config[model]['lambda2']
    rt1_threshold = config[model]['rt1_threshold']
    rt2_threshold = config[model]['rt2_threshold']
    num_seeds = 10

    # Generate a set of unique random seeds
    np.random.seed(42)
    seeds = np.random.choice(range(1, 1001), size=num_seeds, replace=False)

    create_folder_if_not_exists(config['eval_path'])

    logger.info(f'Model: {model}')
    logger.info('Starting evaluation')    
    if model == 'lr_l2' or model == 'lr_l1':
        Cs = config[model]['C']
        params_combination = list(itertools.product(lam1,lam2,rt1_threshold,rt2_threshold,Cs))
        if config['parallel_processing']:
            print('Parallel processing')
            with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                executor.map(process_seed_lr, seeds, [params_combination]*num_seeds, [config]*num_seeds)
        else:
            for seed in seeds:
                process_seed_lr(seed, params_combination, config)

    elif model == 'svm':
        Cs = config[model]['C']
        kernels = config[model]['kernel']
        params_combination = list(itertools.product(lam1,lam2,rt1_threshold,rt2_threshold,Cs,kernels))

        if config['parallel_processing']:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                executor.map(process_seed_svm, seeds, [params_combination]*num_seeds, [config]*num_seeds)
        else:
            for seed in seeds:
                process_seed_svm(seed, params_combination, config)
    elif model == 'rf':
        n_estimators = config[model]['n_estimators']
        params_combination = list(itertools.product(lam1,lam2,rt1_threshold,rt2_threshold,n_estimators))

        if config['parallel_processing']:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                executor.map(process_seed_rf, seeds, [params_combination]*num_seeds, [config]*num_seeds)
        else:
            for seed in seeds:
                process_seed_rf(seed, params_combination, config)
    elif model == 'xgboost':
        n_estimators = config[model]['n_estimators']
        params_combination = list(itertools.product(lam1,lam2,rt1_threshold,rt2_threshold,n_estimators))

        if config['parallel_processing']:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                executor.map(process_seed_xgboost, seeds, [params_combination]*num_seeds, [config]*num_seeds)
        else:
            for seed in seeds:
                process_seed_xgboost(seed, params_combination, config)
    
    elif model == 'NaiveBayes':
        alpha = config[model]['alpha']
        params_combination = list(itertools.product(lam1,lam2,rt1_threshold,rt2_threshold,alpha))

        if config['parallel_processing']:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                executor.map(process_seed_NB, seeds, [params_combination]*num_seeds, [config]*num_seeds)
        else:
            for seed in seeds:
                process_seed_NB(seed, params_combination, config)

    
    logger.info('Evaluation finished')

    # Calculate the average accuracy

    calc_accuracy(config)
    