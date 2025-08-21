import json
from loguru import logger
import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from .utils.heatmap_utils import heatmap, filter_by_m_z
from .utils.heatmap_utils import create_folder_if_not_exists
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_csv_file(csv_file_name, raw_csv_path, mz_list, threshold, m_z_column_name, first_time_column_name, second_time_column_name, area_column_name, output_dir_heatmap):
    try:
        # Load the raw data
        logger.info(f"Reading csv file: {csv_file_name}....")
        raw_csv_file_path = os.path.join(raw_csv_path, csv_file_name)
        raw_sample = pd.read_csv(raw_csv_file_path)
        logger.info(f"Finished reading csv file: {csv_file_name}")

        # Create a folder for the csv file
        create_folder_if_not_exists(os.path.join(output_dir_heatmap, f'{csv_file_name}/'))

        logger.info(f"Processing file {csv_file_name}....")
        # Extract heatmaps for each m/z of the sample
        for idx, row in mz_list.iterrows():
            m_z = (row[m_z_column_name])

            sample_df = filter_by_m_z(raw_sample, m_z, threshold=threshold, m_z_column_name=m_z_column_name)
            hm = heatmap(sample_df, first_time_column_name=first_time_column_name, second_time_column_name=second_time_column_name, area_column_name=area_column_name)

            # Save the heatmap
            file_path = os.path.join(output_dir_heatmap, f'{csv_file_name}/', str((m_z)).replace('.', '_') + '.npy')
            np.save(file_path, hm.to_numpy())

            # Save the first time array
            first_time = hm.columns.to_numpy()
            file_path = os.path.join(output_dir_heatmap, f'{csv_file_name}/', str((m_z)).replace('.', '_') + '_first_time.npy')
            np.save(file_path, first_time)

            # Save the second time array
            second_time = hm.index.to_numpy()
            file_path = os.path.join(output_dir_heatmap, f'{csv_file_name}/', str((m_z)).replace('.', '_') + '_second_time.npy')
            np.save(file_path, second_time)

        logger.info(f"Finished processing file {csv_file_name}")
    except Exception as e:
        logger.error(f"Error processing file {csv_file_name}: {e}")

def parallel_processing(labels, csv_file_name_column, raw_csv_path, mz_list, threshold, m_z_column_name, first_time_column_name, second_time_column_name, area_column_name, output_dir_heatmap):
    csv_file_names = labels[csv_file_name_column].tolist()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_csv_file, csv_file_name, raw_csv_path, mz_list, threshold, m_z_column_name, first_time_column_name, second_time_column_name, area_column_name, output_dir_heatmap): csv_file_name for csv_file_name in csv_file_names}
        for future in tqdm(as_completed(futures), total=len(futures)):
            csv_file_name = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing file {csv_file_name}: {e}")

# Extract heatmaps for a given m/z list
def heatmap_extraction(args):
    mz_list_path = args['mz_list_path']
    labels_path = args['labels_path']
    area_column_name = args['area_column_name']
    m_z_column_name = args['m_z_column_name']
    first_time_column_name = args['first_time_column_name']
    second_time_column_name = args['second_time_column_name']
    csv_file_name_column = args['csv_file_name_column']

    raw_csv_path = args['extract_heatmaps']['raw_csv_path']
    threshold = args['extract_heatmaps']['m_z_threshold']
    output_dir_heatmap = args['heatmap_dir']
    parallel = args['extract_heatmaps']['parallel_processing']

    log_path = os.path.join(output_dir_heatmap, 'extract_heatmaps.log')
    logger.add(log_path, rotation="10 MB")

    # Load the m/z list
    mz_list = pd.read_csv(mz_list_path)

    # Load the labels
    labels = pd.read_csv(labels_path)

    logger.info(f"Extracting heatmaps for m/z list: {mz_list_path}")

    if parallel:
        parallel_processing(labels, csv_file_name_column, raw_csv_path, mz_list, threshold, m_z_column_name, first_time_column_name, second_time_column_name, area_column_name, output_dir_heatmap)
    else:
        for csv_file_name in tqdm(labels[csv_file_name_column].tolist()):
            # Load the raw data
            logger.info(f"Reading csv file: {csv_file_name}....")
            raw_csv_path = os.path.join(raw_csv_path, csv_file_name)
            raw_sample = pd.read_csv(raw_csv_path)
            logger.info(f"Finished reading csv file: {csv_file_name}")

            # Create a folder for the csv file
            create_folder_if_not_exists(os.path.join(output_dir_heatmap,f'{csv_file_name}/'))

            logger.info(f"Processing file {csv_file_name}....")
            # Extract heatmaps for each m/z of the sample
            for idx, row in mz_list.iterrows():
                m_z = int(row[m_z_column_name])
                sample_df = filter_by_m_z(raw_sample, m_z, threshold=threshold, m_z_column_name=m_z_column_name)
                hm = heatmap(sample_df, first_time_column_name=first_time_column_name, second_time_column_name=second_time_column_name, area_column_name=area_column_name)

                # Save the heatmap
                file_path = os.path.join(output_dir_heatmap,f'{csv_file_name}/', str(int(m_z)) + '.npy')
                np.save(file_path, hm.to_numpy())

                # Saving first time array
                first_time = hm.columns.to_numpy()
                file_path = os.path.join(output_dir_heatmap,f'{csv_file_name}/', str(int(m_z)) + '_first_time.npy')
                np.save(file_path, first_time)

                # Saving the second time array
                second_time = hm.index.to_numpy()
                file_path = os.path.join(output_dir_heatmap,f'{csv_file_name}/', str(int(m_z)) + '_second_time.npy')
                np.save(file_path, second_time)

            logger.info(f"Finished processing file {csv_file_name}")