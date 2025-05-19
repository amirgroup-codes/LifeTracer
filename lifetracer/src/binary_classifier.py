import json
from loguru import logger
import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from .utils.heatmap_utils import create_folder_if_not_exists
from .utils.plot_utils import plot_2d_features, plot_3d_peaks, plot_3d_peaks_interactive, plot_3d_signatures, plot_3d_signatures_interactive, plot_distribution_of_peaks, plot_pca, plot_top_coefficients, plot_top_features, plot_3d_feature_groups, save_latex_table
import joblib
from statsmodels.sandbox.stats.multicomp import multipletests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from scipy import stats

def binary_class_signatures(coefficients_pvalues, features_info, results_path,samples,X_train, csv_file_name_column, labels):
    signatures_class_0 = []
    signatures_class_1 = []

    for index,coefficient in coefficients_pvalues:
        index = int(index)
        m_z = features_info.iloc[index]['m/z']
        first_time_start = features_info.iloc[index]['RT1_start']
        second_time_start = features_info.iloc[index]['RT2_start']
        first_time_end = features_info.iloc[index]['RT1_end']
        second_time_end = features_info.iloc[index]['RT2_end']
        RT1_center = features_info.iloc[index]['RT1_center']
        RT2_center = features_info.iloc[index]['RT2_center']

        # What samples contain this signature?
        samples_with_this_sign = ''
        for sample in samples.iloc[np.where(X_train[:,index] == 1)[0].reshape(-1)][csv_file_name_column].to_numpy().tolist():
            samples_with_this_sign += sample +', '
        
        samples_with_this_sign = replace_sample_name(samples_with_this_sign, labels)

        values, counts = np.unique(samples.iloc[np.where(X_train[:,index] == 1)[0].reshape(-1)].to_numpy().tolist(), return_counts=True)

        if counts.size == 0: continue
        
        cluster_label = int(values[np.argmax(counts)])
        if coefficient < 0:
            # Abiotic
            signatures_class_0.append([coefficient,m_z,f'[{first_time_start},{first_time_end}]',f'[{second_time_end},{second_time_start}]',RT1_center,RT2_center, samples_with_this_sign, index, cluster_label])
        else:
            # Biotic
            signatures_class_1.append([coefficient,m_z,f'[{first_time_start},{first_time_end}]',f'[{second_time_end},{second_time_start}]',RT1_center,RT2_center, samples_with_this_sign, index, cluster_label])


    create_folder_if_not_exists(os.path.join(results_path))
    signatures_class0 = pd.DataFrame(signatures_class_0,columns=['coefficient','m/z','RT1','RT2','RT1_center','RT2_center','samples','feature_index','class'])
    signatures_class1 = pd.DataFrame(signatures_class_1,columns=['coefficient','m/z','RT1','RT2','RT1_center','RT2_center','samples','feature_index','class'])

    signatures_class0.to_csv(os.path.join(results_path ,f'lr_l2_class0_signatures.csv'))
    signatures_class1.to_csv(os.path.join(results_path, f'lr_l2_class1_signatures.csv'))

    return signatures_class0, signatures_class1

def is_point_inside_rect(rectangle, point):
    x1 = rectangle[0]
    y1 = rectangle[1]
    x2 = rectangle[2]
    y2 = rectangle[3]
    # Check if the point is within the rectangle bounds
    if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
        return True

    return False

def return_peaks_corresponding_to_cluster(peaks_path,m_z,rt1,rt2,feature_id,coefficient,samples, label):
    peaks = pd.read_csv(os.path.join(peaks_path, f'{m_z}.csv'))

    peaks_inside_cluster = []
    for idx, peak in peaks.iterrows():
        if is_point_inside_rect([rt1[0],rt2[0],rt1[1],rt2[1]],[peak['RT1_center'],peak['RT2_center']]):
            peaks_inside_cluster.append((feature_id,peak['RT1_center'],peak['RT2_center'],coefficient,m_z, peak['csv_file_name'], label))
    
    return peaks_inside_cluster

def get_peaks_feature_df(signaturs_combined,peaks_path):
    peaks_features_id = []
    for idx, row in signaturs_combined.iterrows():
        m_z = int(row['m/z'])
        rt1 = eval(row['RT1'])
        rt2 = eval(row['RT2'])
        peaks_features_id += (return_peaks_corresponding_to_cluster(peaks_path, m_z,rt1,rt2,idx,row['coefficient'],row['samples'],row['class']))
    
    peaks_features_df = pd.DataFrame(np.array(peaks_features_id),columns=['feature_id','RT1','RT2','coefficient','m/z','sample','class'])

    peaks_features_df['feature_id'] = peaks_features_df['feature_id'].astype(int)
    peaks_features_df['coefficient'] = peaks_features_df['coefficient'].astype(float)
    peaks_features_df['RT1'] = peaks_features_df['RT1'].astype(float)
    peaks_features_df['RT2'] = peaks_features_df['RT2'].astype(float)
    peaks_features_df['m/z'] = peaks_features_df['m/z'].astype(float)
    num_clusters = peaks_features_df['feature_id'].max()

    return peaks_features_df, num_clusters

def mann_whitney_u_test_mz(peaks_features_df):
    biotic_peaks = peaks_features_df[peaks_features_df['class'] == '1']['m/z'].to_numpy()
    abiotic_peaks = peaks_features_df[peaks_features_df['class'] == '0']['m/z'].to_numpy()

    statistic, p_value = stats.mannwhitneyu(abiotic_peaks, biotic_peaks, alternative='less')

    logger.info(f'Mann Whitney U test for m/z p-value: {p_value}')

    if p_value < 0.05:
        logger.info("Reject null hypothesis-> Abiotic peak distribution for m/z is significantly lower than biotic")
    else:
        logger.info("Fail to reject null hypothesis-> Abiotic peak distribution for m/z is not significantly lower than biotic")

def mann_whitney_u_test_rt1(peaks_features_df):
    biotic_peaks = peaks_features_df[peaks_features_df['class'] == '1']['RT1'].to_numpy()
    abiotic_peaks = peaks_features_df[peaks_features_df['class'] == '0']['RT1'].to_numpy()

    statistic, p_value = stats.mannwhitneyu(abiotic_peaks, biotic_peaks, alternative='less')

    logger.info(f'Mann Whitney U test for RT1 p-value: {p_value}')

    if p_value < 0.05:
        logger.info("Reject null hypothesis-> Abiotic peak distribution for RT1 is significantly lower than biotic")
    else:
        logger.info("Fail to reject null hypothesis-> Abiotic peak distribution for RT1 is not significantly lower than biotic")

def mann_whitney_u_test_rt2(peaks_features_df):
    biotic_peaks = peaks_features_df[peaks_features_df['class'] == '1']['RT2'].to_numpy()
    abiotic_peaks = peaks_features_df[peaks_features_df['class'] == '0']['RT2'].to_numpy()

    statistic, p_value = stats.mannwhitneyu(abiotic_peaks, biotic_peaks, alternative='less')

    logger.info(f'Mann Whitney U test for RT2 p-value: {p_value}')

    if p_value < 0.05:
        logger.info("Reject null hypothesis-> Abiotic peak distribution for RT2 is significantly higher than biotic")
    else:
        logger.info("Fail to reject null hypothesis-> Abiotic peak distribution for RT2 is not significantly higher than biotic")

def kolmogorov_smirnov_test(peaks_features_df, dimension):
    biotic_peaks = peaks_features_df[peaks_features_df['class'] == '1'][dimension].to_numpy()
    abiotic_peaks = peaks_features_df[peaks_features_df['class'] == '0'][dimension].to_numpy()

    statistic, p_value = stats.ks_2samp(abiotic_peaks, biotic_peaks)

    logger.info(f'Kolmogorov-Smirnov test for {dimension} - statistic: {statistic}, p-value: {p_value}')

    if p_value < 0.05:
        logger.info(f"Reject null hypothesis -> The distributions of {dimension} for abiotic and biotic peaks are significantly different")
    else:
        logger.info(f"Fail to reject null hypothesis -> There is no significant difference in the distributions of {dimension} for abiotic and biotic peaks")

def plot_accuracy_drop_zeroing_coefficients(X_train, y_train, lr_model, result_dir):
    """
    Plot the accuracy drop as top feature coefficients are set to zero one by one.
    
    Parameters:
    - X_train: training features
    - y_train: training labels
    - lr_model: trained LogisticRegression model
    - n_features: number of top features to consider (default: 50)
    
    Returns:
    - None (saves the plot)
    """
    # Get the original coefficients
    original_coefficients = lr_model.coef_[0].copy()
    
    # Sort features by absolute coefficient value
    sorted_feature_indices = np.argsort(np.abs(original_coefficients))[::-1]
    
    # Initialize list to store accuracies
    accuracies = []
    
    # Calculate initial accuracy
    accuracies.append(accuracy_score(y_train, lr_model.predict(X_train)))
    
    # Iterate through features, setting their coefficients to zero one by one
    for i in range(1, X_train.shape[1] + 1):
        # Set the coefficient of the i-th most important feature to zero
        lr_model.coef_[0][sorted_feature_indices[i-1]] = 0
        
        # Recalculate accuracy
        accuracies.append(accuracy_score(y_train, lr_model.predict(X_train)))

    
    # Find the number of features that can be removed while maintaining accuracy thresholds
    accuracy_90 = None
    accuracy_80 = None
    
    for i, acc in enumerate(accuracies):
        if acc >= 0.9 and (i == len(accuracies) - 1 or accuracies[i+1] < 0.9):
            accuracy_90 = i
        if acc >= 0.8 and (i == len(accuracies) - 1 or accuracies[i+1] < 0.8):
            accuracy_80 = i
    
    # If accuracy never drops below thresholds, set to total number of features
    if accuracy_90 is None:
        accuracy_90 = len(accuracies) - 1
    if accuracy_80 is None:
        accuracy_80 = len(accuracies) - 1
    logger.info(f'Classification accuracy remained above 90% after removing top {accuracy_90} features')
    logger.info(f'Classification accuracy remained above 80% after removing top {accuracy_80} features')
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(range(X_train.shape[1] + 1), accuracies)
    plt.xlabel('Number of Top Features Zeroed', fontsize=12)
    plt.ylabel('Classification Accuracy', fontsize=12)
    plt.title('Accuracy Drop as Top Feature Coefficients are Set to Zero', fontsize=12)

    # set tick font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(result_dir,'accuracy_drop_zeroing_coefficients_plot.pdf'),format='pdf',bbox_inches='tight', dpi=300)
    plt.close()
    
    # Restore original coefficients
    lr_model.coef_[0] = original_coefficients

# samples: separated by comma
def replace_sample_name(samples, labels):
    samples = samples.split(', ')
    # remove the last empty string
    samples = samples[:-1]
    new_samples = []
    for sample in samples:
        new_samples.append(labels[labels['csv_file_name'] == sample]['sample_name'].iloc[0])
    return ', '.join(new_samples)

def feature_group_finder(df,path, features_info, labels):
    df = df.copy()

    i = 1
    top_feature_group_indices = []
    feature_gp_info = []

    features_groups_combined = []
    while len(df) != 0:
        row = df.iloc[0]
        rt1 = row['RT1_center']
        rt2 = row['RT2_center']
        index = row['feature_index']
        m_z = int(row['m/z'])
        path_to_save = os.path.join(path, f'rank_{i}.csv')

        feature_groups = df[(df['RT1_center'] > rt1 - 50) & (df['RT1_center'] < rt1 + 50) & (df['RT2_center'] > rt2 - 0.8) & (df['RT2_center'] < rt2 + 0.8)]
        if len(feature_groups) != 0:
            fg = pd.DataFrame(feature_groups)

            fg['group_rank'] = i
            fg.to_csv(path_to_save)

            features_groups_combined.append(fg)

            i += 1
            df = df.drop(feature_groups.index)
            
            top_feature_group_indices.append((index, f'({m_z}, {rt1/60:.2f}, {rt2:.2f})'))

            # replace the sample names with the sample names in the labels
            # feature_groups['samples'] = feature_groups['samples'].apply(lambda x: replace_sample_name(x, labels))

            
            feature_gp_info.append(feature_groups.iloc[0])
    
    # Combined the feature groups
    df = pd.concat(features_groups_combined).to_csv(os.path.join(path, 'feature_groups_combined.csv'))
    
    # Assuming feature_gp_info is your data
    df = pd.DataFrame(feature_gp_info, columns=['coefficient', 'm/z', 'RT1', 'RT2', 'RT1_center', 'RT2_center', 'samples', 'feature_index', 'class'])

    # Add a new index column
    df['group_rank'] = range(1, len(df) + 1)
    df = df.set_index(['group_rank',df.index])

    # Save the DataFrame to a CSV
    df.to_csv(os.path.join(path, 'feature_groups_info.csv'))
    pd.DataFrame(top_feature_group_indices, columns=['feature_index', 'm/z, RT1, RT2']).to_csv(os.path.join(path, 'top_feature_groups.csv'))
    return np.array(top_feature_group_indices), df

def binary_classifier(args):
    log_path = os.path.join(args['results_dir'], 'results.log')
    logger.add(log_path, rotation="10 MB")

    seed = args['seed']
    np.random.seed(seed)
    lam1 = args['lambda1']
    lam2 = args['lambda2']
    rt1_th = args['rt1_threshold']
    rt2_th = args['rt2_threshold']
    C = args['C']

    # Load the data
    features = np.load(os.path.join(args['features_path'], f'features_lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_th}_rt2th_{rt2_th}.npy'))
    features_info = pd.read_csv(os.path.join(args['features_path'], f'features_lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_th}_rt2th_{rt2_th}.csv'))

    # Load the labels
    samples = pd.read_csv(args['labels_path'])

    # Unlabel the samples
    unlabeled_samples_index = samples[samples[args['label_column_name']] == -1].index

    # Remove the unlabeled samples indices from the features
    X_train = np.delete(features, unlabeled_samples_index, axis=0)
    y_train = samples.drop(unlabeled_samples_index)['label'].to_numpy()
    train_samples = samples.drop(unlabeled_samples_index)
    X_test = features[unlabeled_samples_index]
    test_samples = samples.iloc[unlabeled_samples_index]

    # Train the model
    lr = LogisticRegression(penalty='l2', solver='liblinear', C=C,random_state=seed)
    lr.fit(X_train, y_train)
    coefficients = lr.coef_[0]

    logger.info(f'Intercept: {lr.intercept_}')

    # Save the trained model
    create_folder_if_not_exists(args['results_dir'])
    joblib.dump(lr, os.path.join(args['results_dir'], f'lr_l2_model_lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_th}_rt2th_{rt2_th}.joblib'))

    # Predicting the labels of the unlabeled samples
    for i in range(len(test_samples)):
        if test_samples.iloc[i]['label'] == -1:
            pred = lr.predict(X_test[i].reshape(1, -1))
            sample_name = test_samples.iloc[i][args['csv_file_name_column']]
            logger.info(f'Predicted label for {sample_name}: {pred}')

    # Get the p-values
    # p_values = calculate_pvalues(args['number_of_bootstraps'], C, seed, X_train, train_samples, args['label_column_name'])
    # logger.info('P-values calculated.')


    # Combine the coefficients
    coefficients = sorted(enumerate(coefficients), key=lambda x: abs(x[1]), reverse=True)
    coefficients = np.array([(index, value) for index, value in coefficients])

    # Save signatures per class
    signatures_class0, signatures_class1 = binary_class_signatures(coefficients, features_info, args['results_dir'],train_samples,X_train, args['csv_file_name_column'],samples)
    logger.info('Signatures saved.')

    # Plotting top 10 highest to lowest coefficients
    top_featurs_path = os.path.join(args['results_dir'] ,'top_features/')
    create_folder_if_not_exists(top_featurs_path)

    signaturs_combined = pd.concat([signatures_class0,signatures_class1]).sort_values(by='coefficient',key=abs,ascending=False).reset_index(drop=True)
    signaturs_combined.index = signaturs_combined.index + 1
    signaturs_combined.to_csv(os.path.join(top_featurs_path,'lr_l2_top_features_combined.csv'))

    create_folder_if_not_exists(os.path.join(args['results_dir'],'feature_groups'))
    top_feature_group_indices, feature_groups_df = feature_group_finder(signaturs_combined, os.path.join(args['results_dir'],'feature_groups'), features_info, samples)
    
    # plot_top_coefficients(top_feature_group_indices,coefficients, top_featurs_path)

    # Define indices to remove determined by expert (indices that are noise)
    indices_to_remove = [12,14,15,17,21,24,26,27,28,29,30,31,32,33,34,35,39,40,45,46,47,48,49,50,51,52,53,54,55,56]  # Example indices to remove
    
    # Create a mask for indices to keep
    mask = np.ones(len(top_feature_group_indices), dtype=bool)
    for idx in indices_to_remove:
        if idx < len(top_feature_group_indices):
            mask[idx] = False
    
    # Filter the top_feature_group_indices using the mask
    top_feature_group_indices = top_feature_group_indices[mask]
    
    logger.info(f'Removed {len(indices_to_remove)} indices from top_feature_group_indices manually')
    plot_top_features(X_train, coefficients, train_samples, top_featurs_path, args['label_column_name'], args['csv_file_name_column'], top_feature_group_indices[:45], type = 'top')

    # Save latex table of top features
    save_latex_table(top_feature_group_indices, top_featurs_path)

    # Plot specific features for figure 1b
    # specific_features = [0,1,800,801,802,803,804, 1806, 4101, 5020, 5021, 6002, 9150,9151]
    # features_info = features_info.iloc[specific_features]
    # specific_features = [(index, f'({index}, {features_info.iloc[i]["m/z"]}, {features_info.iloc[i]["RT1_center"]/60:.2f}, {features_info.iloc[i]["RT2_center"]:.2f})') for i, index in enumerate(specific_features)]
    # plot_top_features(X_train, coefficients, train_samples, top_featurs_path, args['label_column_name'], args['csv_file_name_column'], np.array(specific_features), type = 'specific')

    logger.info('Top 10 signatures plotted in top_coefficients folder.')

    # Plotting PCA
    plot_pca(features, samples[args['label_column_name']].tolist(), samples['sample_name'].tolist() ,coefficients, args['results_dir'], top_n=20)
    logger.info('PCA plot saved.')

    # Plotting 2D plot of peaks and signatures.
    peaks_features_df, num_clusters = get_peaks_feature_df(signaturs_combined,os.path.join(args['peaks_dir_path'], f'peaks_lambda1_{lam1}_lambda2_{lam2}/'))
    plot_2d_features(signaturs_combined, peaks_features_df, num_clusters, args['results_dir'])

    # logger.info('2D plot of peaks and signatures saved.')
    # Plotting 3D plot of peaks (png)
    # plot_3d_peaks(peaks_features_df, samples,args['results_dir'], label = 'biotic',view = 'big')
    # plot_3d_peaks(peaks_features_df, samples,args['results_dir'], label = 'abiotic', view = 'big')

    # peaks_features_df = pd.read_csv('peaks_features_df.csv')
    # samples = pd.read_csv('samples.csv')
    # plot_3d_peaks(peaks_features_df, samples,args['results_dir'], label = 'biotic',view = 'big')
    # plot_3d_peaks(peaks_features_df, samples,args['results_dir'], label = 'abiotic', view = 'big')

    # Plotting 3D plot of signatures (png)
    plot_3d_signatures(signaturs_combined, args['results_dir'],view = 'small')
    logger.info('3D plot of signatures (png) saved.')

    # interactive 3D plot for peaks
    plot_3d_peaks_interactive(peaks_features_df, args['results_dir'])
    logger.info('3D interactive plot of peaks saved.')

    # interactive 3D plot for signatures
    plot_3d_signatures_interactive(signaturs_combined, args['results_dir'])
    logger.info('3D interactive plot of signatures saved.')

    plot_3d_feature_groups(feature_groups_df, args['results_dir'],view = 'small')
    logger.info('3D plot of feature groups saved.')

    # Distribution of peaks across m/z values
    plot_distribution_of_peaks(peaks_features_df, args['results_dir'], x_axis='m/z')
    plot_distribution_of_peaks(peaks_features_df, args['results_dir'], x_axis='RT1')
    plot_distribution_of_peaks(peaks_features_df, args['results_dir'], x_axis='RT2')
    logger.info('Distribution of peaks across m/z values saved.')

    mann_whitney_u_test_mz(peaks_features_df)
    mann_whitney_u_test_rt1(peaks_features_df)
    mann_whitney_u_test_rt2(peaks_features_df)

    # Add K-S tests
    kolmogorov_smirnov_test(peaks_features_df, 'm/z')
    kolmogorov_smirnov_test(peaks_features_df, 'RT1')
    kolmogorov_smirnov_test(peaks_features_df, 'RT2')

    # Plot accuracy drop
    plot_accuracy_drop_zeroing_coefficients(X_train, y_train, lr, args['results_dir'])
    logger.info('Accuracy drop plot saved.')

    # Recalculate accuracy before zeroing out the coefficients of the noisy feature groups
    accuracy = accuracy_score(y_train, lr.predict(X_train))
    logger.info(f'Accuracy before zeroing out the coefficients of the noisy feature groups: {accuracy}')

    # # Given a list of feature group indices, zero out the coefficients of the features in the feature groups and measure the accuracy drop
    # ranking = [13,15,18,22,25,27,36,41,40,55,56,78,79, 83,84,85,86,87,88,89,90,91,92,93, 95,96,97,104,105,106,107,109,137,154,153,156,161,163,164,167,187,195,196,197,198,199,222,232,233,234,235,236,237,239,240,242]

    # # Get the indices of the features in the feature groups

    # feature_indices = []
    # for index in ranking:
    #     index += 1
    #     df = pd.read_csv(os.path.join(args['results_dir'],f'feature_groups/rank_{index}.csv'))
    #     feature_indices.extend(df['feature_index'].tolist())
    

    # # Zero out the coefficients of the features in the feature groups
    # for index in feature_indices:
    #     coefficients[index] = 0

    # # Recalculate accuracy
    # accuracy = accuracy_score(y_train, lr.predict(X_train))
    # logger.info(f'Accuracy after zeroing out the coefficients of the noisy feature groups: {accuracy}')