import itertools
import os
import numpy as np
import concurrent.futures
import random as python_random
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from loguru import logger
from sklearn.naive_bayes import BernoulliNB
from tqdm import tqdm
from .utils.heatmap_utils import create_folder_if_not_exists

def process_seed_lr(seed, params_combination, config):
    model = 'l2' if config['model'] == 'lr_l2' else 'l1'
    # solver = 'saga' if model == 'l2' else 'liblinear'  # saga is faster for l2, must use liblinear for l1
    solver = 'liblinear'
    
    np.random.seed(seed)
    python_random.seed(int(seed))
    skf = StratifiedKFold(n_splits=6,shuffle=True, random_state=seed)

    samples = pd.read_csv(config['labels_path'])
    features_combined = np.zeros((len(samples),2))

    # Cache for features to avoid repeated disk reads
    features_cache = {}

    samples_labels = samples[config['label_column_name']].to_numpy()

    log = []
    for rest_index, test_index in skf.split(features_combined, samples_labels):
       
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

            # Inner loop for leave-one-out validation
            skf_inner = StratifiedKFold(n_splits=5,shuffle=True, random_state=seed)
            inner_features = features[rest_index]
            inner_labels = samples_labels[rest_index]
            inner_auc = []
            for train_index, val_index in skf_inner.split(inner_features, inner_labels):

                X_val = inner_features[val_index]
                y_val = inner_labels[val_index]
                X_train = inner_features[train_index]
                y_train = inner_labels[train_index]
                
                lr = LogisticRegression(
                    penalty=model,
                    solver=solver,
                    C=C,
                    random_state=seed,
                )

                # Fit the classifier to the data
                lr.fit(X_train, y_train)

                # Compute auc
                auc = roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])

                inner_auc.append(auc)

            auc_val_mean = np.mean(inner_auc)

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
            predictions_test = lr.predict_proba(X_test)
            test_AUC = roc_auc_score(y_test, predictions_test[:, 1])
            log.append((lam1,lam2,rt1_th,rt2_th,auc_val_mean,test_AUC,test_index,C))

            # Clear some memory if cache is too large
            if len(features_cache) > 10:  # Keep only last 10 feature sets
                oldest_key = next(iter(features_cache))
                del features_cache[oldest_key]

    pd.DataFrame(log,columns=['lam1','lam2','rt1_threshold','rt2_threshold','val_AUC','test_AUC','test_id','C']).to_csv(os.path.join(config['eval_path'],f'eval_seed_{seed}.csv'),index=False)

def process_seed_svm(seed, params_combination, config):
    np.random.seed(seed)
    python_random.seed(int(seed))
    skf = StratifiedKFold(n_splits=6,shuffle=True, random_state=seed)

    samples = pd.read_csv(config['labels_path'])
    features_combined = np.zeros((len(samples),2))

    # Cache for features to avoid repeated disk reads
    features_cache = {}

    log = []

    samples_labels = samples[config['label_column_name']].to_numpy()

    for rest_index, test_index in skf.split(features_combined, samples_labels):
       
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

            # Inner loop for leave-one-out validation
            skf_inner = StratifiedKFold(n_splits=5,shuffle=True, random_state=seed)
            inner_features = features[rest_index]
            inner_labels = samples_labels[rest_index]
            inner_auc = []
            for train_index, val_index in skf_inner.split(inner_features, inner_labels):

                X_val = inner_features[val_index]
                y_val = inner_labels[val_index]
                X_train = inner_features[train_index]
                y_train = inner_labels[train_index]
                
                # Optimize SVM configuration for speed
                svc = svm.SVC(
                    kernel=kernel,
                    C=C,
                    random_state=seed,
                    probability=True,
                )

                # Fit the classifier to the data
                svc.fit(X_train, y_train)

                inner_auc.append(roc_auc_score(y_val, svc.predict_proba(X_val)[:, 1]))

            auc_val_mean = np.mean(inner_auc)

            # Train on all training set
            svc = svm.SVC(
                kernel=kernel,
                C=C,
                random_state=seed,
                probability=True,
            )
            svc.fit(features[rest_index], samples[config['label_column_name']].to_numpy()[rest_index])

            X_test = features[test_index]
            y_test = samples[config['label_column_name']].to_numpy()[test_index]
            predictions_test = svc.predict_proba(X_test)
            test_AUC = roc_auc_score(y_test, predictions_test[:, 1])
            log.append((lam1,lam2,rt1_th,rt2_th,auc_val_mean,test_AUC,test_index,C))

            # Clear some memory if cache is too large
            if len(features_cache) > 10:  # Keep only last 10 feature sets
                oldest_key = next(iter(features_cache))
                del features_cache[oldest_key]

    pd.DataFrame(log,columns=['lam1','lam2','rt1_threshold','rt2_threshold','val_AUC','test_AUC','test_id','C']).to_csv(os.path.join(config['eval_path'],f'eval_seed_{seed}.csv'),index=False)

def process_seed_rf(seed, params_combination, config):
    np.random.seed(seed)
    python_random.seed(int(seed))
    skf = StratifiedKFold(n_splits=6,shuffle=True, random_state=seed)

    samples = pd.read_csv(config['labels_path'])
    features_combined = np.zeros((len(samples),2))

    # Cache for features to avoid repeated disk reads
    features_cache = {}

    log = []

    samples_labels = samples[config['label_column_name']].to_numpy()

    for rest_index, test_index in skf.split(features_combined, samples_labels):
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

            # Inner loop for leave-one-out validation
            skf_inner = StratifiedKFold(n_splits=5,shuffle=True, random_state=seed)
            inner_features = features[rest_index]
            inner_labels = samples_labels[rest_index]
            inner_auc = []
            for train_index, val_index in skf_inner.split(inner_features, inner_labels):

                X_val = inner_features[val_index]
                y_val = inner_labels[val_index]
                X_train = inner_features[train_index]
                y_train = inner_labels[train_index]
                
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

                inner_auc.append(roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1]))
            
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
            test_AUC = roc_auc_score(y_test, predictions_test)
            val_AUC = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
            log.append((lam1,lam2,rt1_th,rt2_th,val_AUC,test_AUC,test_index,n_estimators))

            # Clear some memory if cache is too large
            if len(features_cache) > 10:  # Keep only last 10 feature sets
                oldest_key = next(iter(features_cache))
                del features_cache[oldest_key]

            logger.info(f'lam1: {lam1}, lam2: {lam2}, rt1_threshold: {rt1_th}, rt2_threshold: {rt2_th}, n_estimators: {n_estimators}, test_AUC: {test_AUC}')
    pd.DataFrame(log,columns=['lam1','lam2','rt1_threshold','rt2_threshold','val_AUC','test_AUC','test_id', 'n_estimators']).to_csv(os.path.join(config['eval_path'],f'eval_seed_{seed}.csv'),index=False)


def process_seed_NB(seed, params_combination, config):
    np.random.seed(seed)
    python_random.seed(int(seed))
    skf = StratifiedKFold(n_splits=6,shuffle=True, random_state=seed)

    samples = pd.read_csv(config['labels_path'])
    features_combined = np.zeros((len(samples),2))

    # Cache for features to avoid repeated disk reads
    features_cache = {}

    log = []

    samples_labels = samples[config['label_column_name']].to_numpy()

    for rest_index, test_index in skf.split(features_combined, samples_labels):
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

            # Inner loop for leave-one-out validation
            skf_inner = StratifiedKFold(n_splits=5,shuffle=True, random_state=seed)
            inner_features = features[rest_index]
            inner_labels = samples_labels[rest_index]
            inner_auc = []
            for train_index, val_index in skf_inner.split(inner_features, inner_labels):
                X_val = inner_features[val_index]
                y_val = inner_labels[val_index]
                X_train = inner_features[train_index]
                y_train = inner_labels[train_index]
                
                nb = BernoulliNB(
                    alpha=alpha,
                    force_alpha=True,  # Ensures alpha is used even with sparse data
                    binarize=None,  # Data is already binary
                )

                # Fit the classifier to the data
                nb.fit(X_train, y_train)

                # Compute auc
                auc = roc_auc_score(y_val, nb.predict_proba(X_val)[:, 1])

                inner_auc.append(auc)

            auc_val_mean = np.mean(inner_auc)
            
            # Train on all training set
            nb = BernoulliNB(
                alpha=alpha,
                force_alpha=True,
                binarize=None,
            )
            nb.fit(features[rest_index], samples[config['label_column_name']].to_numpy()[rest_index])

            X_test = features[test_index]
            y_test = samples[config['label_column_name']].to_numpy()[test_index]
            predictions_test = nb.predict_proba(X_test)
            test_AUC = roc_auc_score(y_test, predictions_test[:, 1])
            log.append((lam1,lam2,rt1_th,rt2_th,auc_val_mean,test_AUC,test_index,alpha))

            # Clear some memory if cache is too large
            if len(features_cache) > 10:  # Keep only last 10 feature sets
                oldest_key = next(iter(features_cache))
                del features_cache[oldest_key]

            logger.info(f'lam1: {lam1}, lam2: {lam2}, rt1_threshold: {rt1_th}, rt2_threshold: {rt2_th}, alpha: {alpha}, test_AUC: {test_AUC}')
    pd.DataFrame(log,columns=['lam1','lam2','rt1_threshold','rt2_threshold','val_AUC','test_AUC','test_id', 'alpha']).to_csv(os.path.join(config['eval_path'],f'eval_seed_{seed}.csv'),index=False)

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
        val_AUC = []
        test_AUC = []

        result = pd.read_csv(os.path.join(config['eval_path'],f'eval_seed_{seed}.csv'),skipinitialspace=True)

        kf = StratifiedKFold(n_splits=6,shuffle=True, random_state=seed)

        for rest_index, test_index in kf.split(test, labels[config['label_column_name']].to_numpy()):

            test_id = f'{test_index}'

            rs = result[result['test_id'] == test_id]
            rs = rs.sort_values(by=['val_AUC','rt2_threshold'], ascending=[False, True]).reset_index(drop=True)

            row_with_max_val_AUC = rs.iloc[0]

            val_AUC.append(row_with_max_val_AUC['val_AUC'])
            test_AUC.append(row_with_max_val_AUC['test_AUC'])

        avg_val_per_seed.append(np.array(val_AUC).mean()*100)
        avg_test_per_seed.append(np.array(test_AUC).mean()*100)

        std_test_per_seed.append(np.array(test_AUC).std()*100)
    logger.info(f'avg validation AUC: {np.array(avg_val_per_seed).mean()}±{np.array(avg_val_per_seed).std()}')
    logger.info(f'avg test AUC: {np.array(avg_test_per_seed).mean()}±{np.array(avg_test_per_seed).std()}')

    print(avg_test_per_seed)
    print(std_test_per_seed)

def eval(config):
    model = config['model']

    log_path = os.path.join(config['eval_path'], 'eval.log')
    create_folder_if_not_exists(config['eval_path'])
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