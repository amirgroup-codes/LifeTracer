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

# Disable warnings
import warnings
warnings.filterwarnings("ignore")
# Disable Pandas warnings
pd.options.mode.chained_assignment = None

def load_heatmap_data_unaligned(heatmap_dir, m_z, sample):
    first_time = np.load(os.path.join(heatmap_dir, sample, f'{m_z}_first_time.npy'))
    second_time = np.load(os.path.join(heatmap_dir, sample, f'{m_z}_second_time.npy'))
    heatmap_2d = np.load(os.path.join(heatmap_dir, sample, f'{m_z}.npy'))
    ht_df = pd.DataFrame(heatmap_2d, index=second_time, columns=first_time)
    return ht_df


def _collect_time_data(args):
    """Helper function to collect time data for a single sample and m_z combination"""
    sample_name, m_z, heatmap_dir = args
    m_z_str = str(m_z).replace('.','_')
    RT1 = np.load(os.path.join(heatmap_dir, f"{sample_name}",f"{m_z_str}_first_time.npy"))
    RT2 = np.load(os.path.join(heatmap_dir, f"{sample_name}",f"{m_z_str}_second_time.npy"))
    return set(RT1), set(RT2)

def _process_sample_mz(args):
    """Helper function to process a single sample and m_z combination"""
    sample, m_z, config, first_time_all, second_time_all = args
    m_z_str = str(m_z).replace('.','_')
    
    ht_df = load_heatmap_data_unaligned(config['heatmap_dir'], m_z_str, sample)

    # Add columns and index if they are missing with 0 values
    for i in first_time_all:
        if i not in ht_df.columns:
            ht_df[i] = 0
    for i in second_time_all:
        if i not in ht_df.index:
            ht_df.loc[i] = 0

    # Sort the columns and index
    ht_df = ht_df[first_time_all]
    ht_df = ht_df.loc[second_time_all]

    output_path = os.path.join(config['TII_aligned_dir'], sample, f"{m_z_str}.npy")
    np.save(output_path, ht_df.to_numpy())
    return output_path

def align(config):
    log_path = os.path.join(config['TII_aligned_dir'], 'TII_alignment.log')
    logger.add(log_path, rotation="10 MB")

    m_zs = pd.read_csv(config['mz_list_path'])
    samples = pd.read_csv(config['labels_path'])

    print("Aligning TIIs...")

    # Parallelize the first loop to collect all time data
    first_time_all = set()
    second_time_all = set()
    
    # Create all combinations of samples and m_z values for parallel processing
    tasks = []
    for idx, sample in samples.iterrows():
        for m_z in m_zs[config['m_z_column_name']]:
            tasks.append((sample[config['csv_file_name_column']], m_z, config['heatmap_dir']))
    
    # Use ThreadPoolExecutor for I/O bound operations (file loading)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(_collect_time_data, tasks)
        
        for first_times, second_times in results:
            first_time_all.update(first_times)
            second_time_all.update(second_times)

    first_time_all = sorted(list(first_time_all))
    second_time_all = sorted(list(second_time_all), reverse=True)

    # Create output directories for all samples first
    for sample in samples[config['csv_file_name_column']].tolist():
        create_folder_if_not_exists(os.path.join(config['TII_aligned_dir'], sample))
    
    # Parallelize the second loop to process all sample-m_z combinations
    processing_tasks = []
    for sample in samples[config['csv_file_name_column']].tolist():
        for m_z in m_zs[config['m_z_column_name']].tolist():
            processing_tasks.append((sample, m_z, config, first_time_all, second_time_all))
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks and collect results
        results = list(executor.map(_process_sample_mz, processing_tasks))

    
    
    np.save(os.path.join(config['TII_aligned_dir'], "first_time.npy"), np.array(first_time_all))
    np.save(os.path.join(config['TII_aligned_dir'], "second_time.npy"), np.array(second_time_all))


    # RT1_step = first_time[0][1]-first_time[0][0]
    # RT2_step = second_time[0][1]-second_time[0][0]

    # for RT1 in first_time:
    #     if len(RT1) == 0:
    #         continue
    #     RT1 = RT1 - RT1[0]
    #     if not np.allclose(RT1 % RT1_step, 0):
    #         print("RT1 not aligned")
    #         break

    # for RT2 in second_time:
    #     if len(RT2) == 0:
    #         continue
    #     if not np.allclose(RT2 % RT2_step, 0):
    #         print("RT2 not aligned")
    #         print(RT2)
    #         break
    # print("All RTs are aligned")