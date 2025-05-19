import itertools
import os
from loguru import logger
import pandas as pd
import numpy as np

from lifetracer.src.utils.misc import find_center
from .utils.heatmap_utils import create_folder_if_not_exists
from .utils.rt_alignment_utils import find_cluster_features, which_cluster
import concurrent.futures

class RetentionTimesAlignment:
    def __init__(self, samples_name_list : pd.DataFrame,
                  mz_list : list, peaks_dir_path : str, features_out_path: str,  output_dir_TII_aligned : str,
                  rt1_threshold : float, rt2_threshold : float, rt1_timestep: float, rt2_timestep: float) -> None:
        self.samples_name_list = samples_name_list
        self.mz_list = mz_list
        self.peaks_dir_path = peaks_dir_path
        self.features_out_path = features_out_path
        self.rt1_threshold = rt1_threshold
        self.rt2_threshold = rt2_threshold
        self.output_dir_TII_aligned = output_dir_TII_aligned

        self.rt1_timestep = rt1_timestep
        self.rt2_timestep = rt2_timestep

        create_folder_if_not_exists(self.features_out_path)
    
    """
    This function aligns retention times by considering peaks within a certain threshold as the same peak. 
    For example, if the retention time 1 of a peak is 10.0 and the threshold is 0.1, then all peaks with retention time 1 
    between 9.9 and 10.1 will be considered the same peak.

    Output:
    - samples_features_combined: A matrix where each row corresponds to a sample and each column corresponds to a feature. 
    If the value at (i, j) in the matrix is 1, then the j-th peak is present in the i-th sample.
    - features_df_combined: A DataFrame where each row corresponds to a feature and each column corresponds to a property of the feature.
    The columns are: 'm/z', 'RT1_start', 'RT2_start', 'RT1_end', 'RT2_end', 'RT1_center', 'RT2_center'.
    """
    def align_retention_times(self):
        all_features = None
        all_features_info = pd.DataFrame([])

        for m_z in map(int, self.mz_list):
            features, features_info = self.generate_features(m_z, self.samples_name_list)

            if features_info:
                all_features = features if all_features is None else np.concatenate([all_features, features], axis=1)
                features_info = pd.DataFrame(features_info, columns=['m/z', 'RT1_start', 'RT2_start', 'RT1_end', 'RT2_end', 'RT1_center', 'RT2_center'])
                all_features_info = pd.concat([all_features_info, features_info])

        all_features_info.reset_index(drop=True, inplace=True)
        return all_features, all_features_info
    
    def is_contained(self, peak, cluster):
        return (cluster['min_RT1_center'] <= peak['RT1_center'] <= cluster['max_RT1_center']) and (cluster['min_RT2_center'] <= peak['RT2_center'] <= cluster['max_RT2_center'])
    # Returns cluster maps and clusters information(e.g. their starting and end point) for a specific M/Z
    def generate_features(self, m_z, samples_name):
        # Load peaks data from the specified file
        peaks_file_path = os.path.join(self.peaks_dir_path, f'{m_z}.csv')
        peaks = pd.read_csv(peaks_file_path)
        
        peaks = peaks.sort_values(by=['RT1_center', 'RT2_center']).reset_index(drop=True)
        
        # Initialize list to store clusters
        rectangular_clusters = []
        remaining_peaks = peaks.copy()
        
        while not remaining_peaks.empty:
            # Take the first peak as the reference
            ref_peak = remaining_peaks.iloc[0]
            
            # Find all peaks within threshold of the reference peak
            mask = (
                (abs(remaining_peaks['RT1_center'] - ref_peak['RT1_center']) <= self.rt1_threshold) &
                (abs(remaining_peaks['RT2_center'] - ref_peak['RT2_center']) <= self.rt2_threshold)
            )
            
            # Get the cluster peaks and create cluster boundaries
            cluster_peaks = remaining_peaks[mask]
            cluster = {
                'min_RT1_center': cluster_peaks['RT1_center'].min(),
                'max_RT1_center': cluster_peaks['RT1_center'].max(),
                'min_RT2_center': cluster_peaks['RT2_center'].min(),
                'max_RT2_center': cluster_peaks['RT2_center'].max()
            }
            rectangular_clusters.append(cluster)
            
            # Remove the clustered peaks from remaining peaks
            remaining_peaks = remaining_peaks[~mask].reset_index(drop=True)
        
        # Convert list of clusters to DataFrame
        rectangular_clusters = pd.DataFrame(rectangular_clusters)

        if not rectangular_clusters.empty:
            cluster_map = np.zeros((len(samples_name), len(rectangular_clusters))) # Initialize the cluster map

            # Populate the cluster map for each sample
            for sample_id, sample in enumerate(samples_name):
                sample_peaks_df = peaks[peaks['csv_file_name'] == sample]
                for _, peak in sample_peaks_df.iterrows():
                    for cluster_id, cluster in rectangular_clusters.iterrows():
                        if self.is_contained(peak, cluster):
                            cluster_map[sample_id, cluster_id] = 1
            
            # Collect clusters information
            clusters_info = [
                (
                    m_z,
                    clusters['min_RT1_center'], clusters['min_RT2_center'],  # RT1_start, RT2_start
                    clusters['max_RT1_center'], clusters['max_RT2_center'],  # RT1_end, RT2_end
                    find_center((clusters['min_RT1_center'], clusters['min_RT2_center'], 
                                 clusters['max_RT1_center'], clusters['max_RT2_center']), rt1_timestep=self.rt1_timestep, rt2_timestep=self.rt2_timestep)[0],  # RT1_center
                    find_center((clusters['min_RT1_center'], clusters['min_RT2_center'], 
                                 clusters['max_RT1_center'], clusters['max_RT2_center']), rt1_timestep=self.rt1_timestep, rt2_timestep=self.rt2_timestep)[1],  # RT2_center
                )
                for idx, clusters in rectangular_clusters.iterrows()
            ]            

            return cluster_map, clusters_info
        else:
            return [], []

def retention_times_alignment_process(params,peaks_dir_path,samples_name_list,mz_list,features_out_path, output_dir_TII_aligned, rt1_timestep, rt2_timestep):
    for param in params:
        lam1 = param[0]
        lam2 = param[1]
        rt1_threshold,rt2_threshold = param[2],param[3]

        peaks_path = os.path.join(peaks_dir_path, f'peaks_lambda1_{lam1}_lambda2_{lam2}/')
        rta = RetentionTimesAlignment(samples_name_list, mz_list, peaks_path, features_out_path, output_dir_TII_aligned, rt1_threshold, rt2_threshold, rt1_timestep, rt2_timestep)
        features, feature_info_df = rta.align_retention_times()

        np.save(os.path.join(features_out_path, f'features_lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_threshold}_rt2th_{rt2_threshold}.npy'), features)
        feature_info_df.to_csv(os.path.join(features_out_path, f'features_lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_threshold}_rt2th_{rt2_threshold}.csv'))

        logger.info(f'Features for lambda1={lam1}, lambda2={lam2}, rt1_threshold={rt1_threshold}, rt2_threshold={rt2_threshold} are saved.')

def retention_times_alignment(config):
    log_path = os.path.join(config['peaks_dir_path'], 'find_peaks.log')
    logger.add(log_path, rotation="10 MB")

    # Load the m/z list
    m_zs = pd.read_csv(config['mz_list_path'])
    mz_list = m_zs[config['m_z_column_name']].tolist()

    # Samples name
    samples = pd.read_csv(config['labels_path'])
    samples_name = samples[config['csv_file_name_column']].tolist()

    lam1 = config['lambda1']
    lam2 = config['lambda2']

    rt1_threshold = config['rt1_threshold']
    rt2_threshold = config['rt2_threshold']

    params_combination = list(itertools.product(lam1,lam2,rt1_threshold,rt2_threshold))

    splits = [params_combination[i:i + config['number_of_splits']] for i in range(0, len(params_combination), config['number_of_splits'])]

    create_folder_if_not_exists(os.path.join(config['features_path'],'peaks_per_parameter/'))

    rt1_axis = np.load(os.path.join(config['output_dir_TII_aligned'], 'first_time.npy'))
    rt2_axis = np.load(os.path.join(config['output_dir_TII_aligned'], 'second_time.npy'))
    rt1_timestep = round(rt1_axis[1] - rt1_axis[0],3)
    rt2_timestep = round(rt2_axis[0] - rt2_axis[1],3)

    if config['parallel_processing']:
        with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
            executor.map(retention_times_alignment_process, splits, itertools.repeat(config['peaks_dir_path']), 
                itertools.repeat(samples_name), itertools.repeat(mz_list), 
                itertools.repeat(config['features_path']),
                itertools.repeat(config['output_dir_TII_aligned']),
                itertools.repeat(rt1_timestep), itertools.repeat(rt2_timestep))
    else:
        retention_times_alignment_process(params_combination, config['peaks_dir_path'], samples_name, mz_list, config['features_path'], config['output_dir_TII_aligned'], rt1_timestep, rt2_timestep)
    
    logger.info('Retention times alignment process is completed.')