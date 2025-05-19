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

def load_heatmap_data_unaligned(heatmap_dir, m_z, sample):
    first_time = np.load(os.path.join(heatmap_dir, sample, f'{m_z}_first_time.npy'))
    second_time = np.load(os.path.join(heatmap_dir, sample, f'{m_z}_second_time.npy'))
    heatmap_2d = np.load(os.path.join(heatmap_dir, sample, f'{m_z}.npy'))
    ht_df = pd.DataFrame(heatmap_2d, index=second_time, columns=first_time)
    return ht_df


def align(config):
    log_path = os.path.join(config['output_dir_aligned'], 'find_peaks.log')
    logger.add(log_path, rotation="10 MB")

    m_zs = pd.read_csv(config['mz_list_path'])
    samples = pd.read_csv(config['labels_path'])

    
    first_time_all = set()
    second_time_all = set()
    for idx, sample in samples.iterrows():
        for m_z in m_zs[config['m_z_column_name']]:
            m_z = str(m_z).replace('.','_')
            RT1 = np.load(os.path.join(config['output_dir_heatmap'], f"{sample[config['csv_file_name_column']]}",f"{m_z}_first_time.npy"))
            RT2 = np.load(os.path.join(config['output_dir_heatmap'], f"{sample[config['csv_file_name_column']]}",f"{m_z}_second_time.npy"))
            first_time_all.update(RT1)
            second_time_all.update(RT2)

    first_time_all = sorted(list(first_time_all))
    second_time_all = sorted(list(second_time_all), reverse=True)

    
    for sample in samples[config['csv_file_name_column']].tolist():
        create_folder_if_not_exists(os.path.join(config['output_dir_aligned'], sample))
        for m_z in m_zs[config['m_z_column_name']].tolist():
            m_z = str(m_z).replace('.','_')
            ht_df = load_heatmap_data_unaligned(config['output_dir_heatmap'], (m_z), sample)

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

            np.save(os.path.join(config['output_dir_aligned'], sample, f"{m_z}.npy"), ht_df.to_numpy())

    
    
    np.save(os.path.join(config['output_dir_aligned'], "first_time.npy"), np.array(first_time_all))
    np.save(os.path.join(config['output_dir_aligned'], "second_time.npy"), np.array(second_time_all))


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