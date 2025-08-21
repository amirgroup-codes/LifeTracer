import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
import concurrent.futures
from loguru import logger
from scipy.spatial.distance import cdist
from .utils.heatmap_utils import create_folder_if_not_exists, load_headmaps_list
from .utils.misc import *
from sklearn.cluster import DBSCAN

def get_bounding_box(cluster_indices, points):
    """
    Given a set (or list) of indices belonging to a cluster and the full list of `points`,
    compute the bounding box in terms of (min_row, min_col, max_row, max_col).
    """
    cluster_points = points[list(cluster_indices)]
    min_row, min_col = np.min(cluster_points, axis=0)
    max_row, max_col = np.max(cluster_points, axis=0)
    return min_row, min_col, max_row, max_col

def bounding_box_dimensions(cluster_indices, points, rt1_timestep, rt2_timestep):
    """
    Return the bounding box "width" in RT1 units and "height" in RT2 units
    given the cluster's indices.
    Note: 
      row -> RT2 index
      col -> RT1 index
      So:
         bounding box width  ~ (max_col - min_col + 1) * rt1_timestep
         bounding box height ~ (max_row - min_row + 1) * rt2_timestep
    """
    min_row, min_col, max_row, max_col = get_bounding_box(cluster_indices, points)
    width = (max_col - min_col + 1) * rt1_timestep
    height = (max_row - min_row + 1) * rt2_timestep
    return width, height

def split_cluster(cluster_indices, points,
                  max_rt1_width, max_rt2_height,
                  rt1_timestep, rt2_timestep):
    """
    Recursively split a cluster if its bounding box exceeds the given max_rt1_width or
    max_rt2_height.
    
    Args:
      cluster_indices: set (or list) of indices belonging to the same cluster.
      points: np.array of shape (N,2), where each row is (row_index, col_index).
      max_rt1_width: maximum allowed width in RT1 units.
      max_rt2_height: maximum allowed height in RT2 units.
      rt1_timestep: RT1 step size between adjacent columns.
      rt2_timestep: RT2 step size between adjacent rows.
    
    Returns:
      A list of "split" clusters. Each element in the list is a set of indices that
      represents a cluster whose bounding box is within the allowed dimension.
    """
    # If empty or single point, just return it
    if not cluster_indices or len(cluster_indices) == 1:
        return [cluster_indices]

    width, height = bounding_box_dimensions(cluster_indices, points, rt1_timestep, rt2_timestep)

    # Base case: the cluster is within allowed bounding box
    if width <= max_rt1_width and height <= max_rt2_height:
        return [cluster_indices]

    # Otherwise, we need to split
    min_row, min_col, max_row, max_col = get_bounding_box(cluster_indices, points)
    
    # Convert cluster_indices to a list for easier handling
    cluster_indices_list = list(cluster_indices)
    cluster_points = points[cluster_indices_list]

    # Decide whether to split along RT1 (columns) or RT2 (rows).
    # Typically, you'd split along the axis that exceeds the limit.
    # If both exceed, choose one (here we prefer splitting by RT1 first).
    split_clusters = []
    if width > max_rt1_width:
        # Split along RT1 (columns)
        midpoint = (min_col + max_col) // 2
        # "Left" subcluster => col <= midpoint
        # "Right" subcluster => col >  midpoint
        left_indices = []
        right_indices = []
        for i, (r, c) in enumerate(cluster_points):
            if c <= midpoint:
                left_indices.append(cluster_indices_list[i])
            else:
                right_indices.append(cluster_indices_list[i])
        
        left_indices_set = set(left_indices)
        right_indices_set = set(right_indices)
        
        # Recursively split further if needed
        if len(left_indices_set) > 0:
            split_clusters.extend(
                split_cluster(left_indices_set, points,
                              max_rt1_width, max_rt2_height,
                              rt1_timestep, rt2_timestep)
            )
        if len(right_indices_set) > 0:
            split_clusters.extend(
                split_cluster(right_indices_set, points,
                              max_rt1_width, max_rt2_height,
                              rt1_timestep, rt2_timestep)
            )
    elif height > max_rt2_height:
        # Split along RT2 (rows)
        midpoint = (min_row + max_row) // 2
        # "Top" subcluster => row <= midpoint
        # "Bottom" subcluster => row >  midpoint
        top_indices = []
        bottom_indices = []
        for i, (r, c) in enumerate(cluster_points):
            if r <= midpoint:
                top_indices.append(cluster_indices_list[i])
            else:
                bottom_indices.append(cluster_indices_list[i])
        
        top_indices_set = set(top_indices)
        bottom_indices_set = set(bottom_indices)
        
        # Recursively split further if needed
        if len(top_indices_set) > 0:
            split_clusters.extend(
                split_cluster(top_indices_set, points,
                              max_rt1_width, max_rt2_height,
                              rt1_timestep, rt2_timestep)
            )
        if len(bottom_indices_set) > 0:
            split_clusters.extend(
                split_cluster(bottom_indices_set, points,
                              max_rt1_width, max_rt2_height,
                              rt1_timestep, rt2_timestep)
            )
    
    return split_clusters

def get_cluster_rectangles(clusters, points):
    """
    Get the smallest rectangle containing each cluster.
    """
    rectangles = []
    for cluster in clusters:
        cluster_points = points[list(cluster)]
        min_row, min_col = np.min(cluster_points, axis=0)
        max_row, max_col = np.max(cluster_points, axis=0)
        rectangles.append((min_row, min_col, max_row, max_col))
    return rectangles

def find_clusters(points, 
                  threshold, 
                  min_points=20,
                  max_rt1_width=None, 
                  max_rt2_height=None,
                  rt1_timestep=3.504, 
                  rt2_timestep=0.008):
    """
    Find clusters of points using DBSCAN, then optionally split them if
    max_rt1_width or max_rt2_height constraints are provided.
    """
    # If we have too many points, you could skip or slice.
    if len(points) > 100000:
        return []

    # Step 1: DBSCAN
    dbscan = DBSCAN(eps=threshold, min_samples=min_points)
    cluster_labels = dbscan.fit_predict(points)

    # Step 2: Organize points into clusters
    raw_clusters = []
    for label in set(cluster_labels):
        if label != -1:  # -1 represents noise in DBSCAN
            cluster = np.where(cluster_labels == label)[0]
            raw_clusters.append(set(cluster))

    # Step 3: If no bounding box constraints given, return raw clusters
    if max_rt1_width is None and max_rt2_height is None:
        return raw_clusters

    # Otherwise, split each cluster as needed
    splitted_clusters = []
    for cluster_indices in raw_clusters:
        # Recursively split this cluster if it exceeds the dimension constraints
        splitted = split_cluster(cluster_indices, points,
                                 max_rt1_width, max_rt2_height,
                                 rt1_timestep, rt2_timestep)
        splitted_clusters.extend(splitted)

    return splitted_clusters


def find_peaks(filtered_binary,threshold = 5, rt1_timestep = 3.504, rt2_timestep = 0.008):
    coordinates = np.column_stack(np.where(filtered_binary))
    
    # Return empty lists if no coordinates found
    if coordinates.size == 0:
        return [], []

    clusters = find_clusters(
        points=coordinates, 
        threshold=threshold,
        min_points=20,
        max_rt1_width=50,
        max_rt2_height=1,
        rt1_timestep=rt1_timestep,
        rt2_timestep=rt2_timestep
    )


    cluster_rectangles = get_cluster_rectangles(clusters, coordinates)

    cluster_centers = []

    for cluster in cluster_rectangles:
        cluster_centers.append(find_center_indices(cluster))

    return cluster_centers,cluster_rectangles

# Calculates the summation of the area of the rectangle
def rectanle_sum_area(arr, rect):
    # Unpack the rectangle coordinates
    start_row, start_col, end_row, end_col = rect
    # Initialize sum
    total = 0
    # Iterate over the rows and columns in the rectangle
    for row in range(start_row, end_row + 1):
        for col in range(start_col, end_col + 1):
            total += arr[row][col]
    return total

def remove_noisy_columns_peaks(df_peaks, ht_df_mean_subtracted, sample):
    df_peaks = df_peaks.copy()
    column_noises = ht_df_mean_subtracted.columns[(ht_df_mean_subtracted != 0).all()].unique()

    distance = 50
    for column_noise in column_noises:

        df_peaks_to_remove = df_peaks[(df_peaks['csv_file_name'] == sample) & (df_peaks['RT1_center'] >= column_noise - distance) & (df_peaks['RT1_center'] <= column_noise + distance)].index

        df_peaks.drop(df_peaks_to_remove, inplace=True)
    return df_peaks

def remove_specified_columns_peaks(df_peaks, ht_df, sample, noise_columns, max_distance_removal_noisy_columns=50, non_zero_ratio_column_threshold=0.1):
    df_peaks = df_peaks.copy()

    for column in noise_columns:
        
        column_closest = ht_df.columns[np.abs(ht_df.columns - column).argmin()]
        column_non_zero_count = np.count_nonzero(ht_df.loc[:,ht_df.columns[(ht_df.columns > column_closest - max_distance_removal_noisy_columns) & (ht_df.columns < column_closest + max_distance_removal_noisy_columns)]].values)

        non_zero_percentage_column = column_non_zero_count/(ht_df[column_closest].values.reshape(-1).shape[0]+1)

        if non_zero_percentage_column > non_zero_ratio_column_threshold:

            df_peaks_to_remove = df_peaks[(df_peaks['csv_file_name'] == sample) & (df_peaks['RT1_center'] >= column_closest - max_distance_removal_noisy_columns) & (df_peaks['RT1_center'] <= column_closest + max_distance_removal_noisy_columns)].index
            
            df_peaks.drop(df_peaks_to_remove, inplace=True)
    
    return df_peaks

def remove_noisy_regions_peaks(df_peaks, ht_df, sample, noisy_region):
    df_peaks = df_peaks.copy()
    rt1_range_start = -np.inf if noisy_region['first_time_start'] == -1 else noisy_region['first_time_start']
    rt2_range_start = -np.inf if noisy_region['second_time_start'] == -1 else noisy_region['second_time_start']
    rt1_range_end = np.inf if noisy_region['first_time_end'] == -1 else noisy_region['first_time_end']
    rt2_range_end = np.inf if noisy_region['second_time_end'] == -1 else noisy_region['second_time_end']
    non_zero_ratio_region_threshold = noisy_region['non_zero_ratio_region_threshold']

    ht_df_values = ht_df.loc[rt2_range_end:rt2_range_start, ht_df.columns[(ht_df.columns > rt1_range_start) & (ht_df.columns < rt1_range_end)]].values.reshape(-1)
    non_zero_count = np.count_nonzero(ht_df_values)
    non_zero_percentage = non_zero_count/(ht_df_values.shape[0]+1)

    if non_zero_percentage > non_zero_ratio_region_threshold:
        df_peaks_to_remove = df_peaks[(df_peaks['csv_file_name'] == sample) & (df_peaks['RT1_center'] >= rt1_range_start) & (df_peaks['RT1_center'] <= rt1_range_end) &
                                       (df_peaks['RT2_center'] >= rt2_range_start) & (df_peaks['RT2_center'] <= rt2_range_end)].index
        
        df_peaks.drop(df_peaks_to_remove, inplace=True)
    
    return df_peaks

def convolution_filter(df_peaks, sample, windows,ht_df_filtered,non_zero_ratio_lambda3_filter):
    df_peaks = df_peaks.copy()
    for rt2_range, rt1_range in windows:

        rt1_range_start, rt1_range_end = rt1_range
        rt2_range_start, rt2_range_end = rt2_range

        selected_segment = ht_df_filtered.loc[rt2_range_end:rt2_range_start, ht_df_filtered.columns[(ht_df_filtered.columns > rt1_range_start) & (ht_df_filtered.columns < rt1_range_end)]].values
        non_zero_count_filtered = np.count_nonzero(selected_segment)
        non_zero_percentage_filtered = non_zero_count_filtered/(selected_segment.reshape(-1).shape[0]+1)

        if non_zero_percentage_filtered > non_zero_ratio_lambda3_filter:
            df_peaks.drop(df_peaks[(df_peaks['csv_file_name'] == sample) & (df_peaks['RT1_center'] >= rt1_range_start) & (df_peaks['RT1_center'] <= rt1_range_end) & 
                                (df_peaks['RT2_center'] >= rt2_range_start) & (df_peaks['RT2_center'] <= rt2_range_end)].index, inplace=True)

    return df_peaks

def extract_peaks_process(save_peaks_path, config, params):
    samples = pd.read_csv(config['labels_path'])
    samples_name = samples[config['csv_file_name_column']].tolist()

    first_time_axis = np.load(os.path.join(config['TII_aligned_dir'], f'first_time.npy'))
    second_time_axis = np.load(os.path.join(config['TII_aligned_dir'], f'second_time.npy'))

    rt1_axis = np.load(os.path.join(config['TII_aligned_dir'], 'first_time.npy'))
    rt2_axis = np.load(os.path.join(config['TII_aligned_dir'], 'second_time.npy'))
    rt1_timestep = round(rt1_axis[1] - rt1_axis[0],3)
    rt2_timestep = round(rt2_axis[0] - rt2_axis[1],3)
    
    for param in params:
        lambda1 = param[0]
        lambda2 = param[1]
        m_z = str(param[2]).replace('.','_')
        heatmaps = load_headmaps_list(config['TII_aligned_dir'],samples_name,m_z)

        df_peaks_list = []
        for idx, sample in enumerate(samples_name):
            
            # Skip the sample if it has no heatmap
            ht_df = heatmaps[idx]
            if len(ht_df) == 0: continue

            heatmap_numpy = ht_df.to_numpy()
            std = np.std(heatmap_numpy.reshape(-1))
            filtered = (heatmap_numpy >= lambda1*std)
            filtered_df = pd.DataFrame(filtered, index=ht_df.index, columns=ht_df.columns)

            # Overall non-zero ratio filtering
            if config['overall_filter']['enable']:
                non_zero_count_filtered = np.count_nonzero(filtered)
                non_zero_percentage_filtered = non_zero_count_filtered/(filtered.size+1)
                if non_zero_percentage_filtered > config['overall_filter']['non_zero_ratio_filter']:
                    df_peaks = pd.DataFrame([], columns=['csv_file_name', 'peak_area','RT1_center', 'RT2_center', 'RT1_start', 'RT2_start', 'RT1_end', 'RT2_end'])
                    continue
            
            cluster_centers,cluster_rectangles = find_peaks(filtered, threshold=config['peak_max_neighbor_distance'], rt1_timestep=rt1_timestep, rt2_timestep=rt2_timestep)

            # Adding actual time values to the peaks instead of indices
            peaks_pairs = []
            # Filter the peaks based on the area
            for id, point in enumerate(cluster_centers):
                intensity = rectanle_sum_area(heatmap_numpy,cluster_rectangles[id])
                if intensity >= lambda2*std:
                    peaks_pairs.append((sample,intensity, first_time_axis[point[1]],second_time_axis[point[0]],
                                            first_time_axis[cluster_rectangles[id][1]],second_time_axis[cluster_rectangles[id][0]],
                                            first_time_axis[cluster_rectangles[id][3]],second_time_axis[cluster_rectangles[id][2]]))

            if len(peaks_pairs) == 0: continue

            df_peaks = pd.DataFrame(peaks_pairs, columns=['csv_file_name', 'peak_area','RT1_center', 'RT2_center', 'RT1_start', 'RT2_start', 'RT1_end', 'RT2_end'])

            if config['strict_noise_filtering']:
                if config['enable_noisy_regions']:
                    # Removing peaks that occur in noisy regions
                    for noisy_region in config['noisy_regions']:
                        df_peaks = remove_noisy_regions_peaks(df_peaks, filtered_df, sample, noisy_region)

                if config['convolution_filter']['enable']:
                    # Convolution filtering
                    windows = generate_windows(ht_df.index.min(), ht_df.index.max(), config['convolution_filter']['rt2_window_size'], 
                                                config['convolution_filter']['rt2_stride'], ht_df.columns.min(), ht_df.columns.max(), 
                                                config['convolution_filter']['rt1_window_size'], 
                                                config['convolution_filter']['rt1_stride'])
                    df_peaks = convolution_filter(df_peaks, sample, windows, filtered_df, config['convolution_filter']['non_zero_ratio_lambda3_filter'])

            df_peaks_list.append(df_peaks)

        logger.info(f'Saving peaks for m/z value: {m_z}')
        if len(df_peaks_list) == 0:
            df_all_peaks = pd.DataFrame([], columns=['csv_file_name', 'peak_area','RT1_center', 'RT2_center', 'RT1_start', 'RT2_start', 'RT1_end', 'RT2_end'])
            df_all_peaks.to_csv(os.path.join(save_peaks_path, f'{m_z}.csv'), index=False)
        else:
            # Concatenate the peaks from all samples
            df_all_peaks = pd.concat(df_peaks_list)
            df_all_peaks.to_csv(os.path.join(save_peaks_path, f'{m_z}.csv'), index=False)
        
def extract_peaks(config):
    log_path = os.path.join(config['peaks_dir_path'], 'find_peaks.log')
    logger.add(log_path, rotation="10 MB")

    lambda1 = config['lambda1']
    lambda2 = config['lambda2']
    parallel_processing = config['parallel_processing']

    m_zs = pd.read_csv(config['mz_list_path'])
    m_z_list = m_zs[config['m_z_column_name']].tolist()

    params_combination = [(lambda1, lambda2, mz) for mz in m_z_list]

    save_peaks_path = os.path.join(config['peaks_dir_path'], f'peaks_lambda1_{lambda1}_lambda2_{lambda2}/')
    create_folder_if_not_exists(save_peaks_path)

    # Split the parameters into multiple splits for parallel processing
    num_splits = config['number_of_splits'] if parallel_processing else 1

    # array split without numpy
    params_splits = [params_combination[i:i + num_splits] for i in range(0, len(params_combination), num_splits)]

    if parallel_processing:
        with concurrent.futures.ProcessPoolExecutor(max_workers=config['number_of_splits']) as executor:
            executor.map(extract_peaks_process, itertools.repeat(save_peaks_path), itertools.repeat(config), params_splits)
    else:
        extract_peaks_process(save_peaks_path, config, params_combination)

    logger.info('Peaks extraction is done.')