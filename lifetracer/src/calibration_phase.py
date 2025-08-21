from itertools import product
import json
from lifetracer.src.utils.heatmap_utils import create_folder_if_not_exists
import pandas as pd
import os
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from .find_peaks import extract_peaks

def _process_combination(args):
    lambda1, lambda2, base_config, calibration_dataset, labels, calibration_phase_output_dir, rt1_tol, rt2_tol = args
    
    task_config = copy.deepcopy(base_config)
    task_config['lambda1'] = lambda1
    task_config['lambda2'] = lambda2
    task_config['peaks_dir_path'] = os.path.join(
        calibration_phase_output_dir,
        f'lambda1_{lambda1}_lambda2_{lambda2}'
    )

    # extract_peaks(task_config)
    total_hits = 0
    total_wrong = 0
    total_peaks_number = 0

    for _, row in calibration_dataset.iterrows():
        true_positives = 0
        wrong = 0
        mz = row['Base Mass']
        rt1 = row['RT1']
        rt2 = row['RT2']

        peaks_df = pd.read_csv(
            os.path.join(
                task_config['peaks_dir_path'],
                f'peaks_lambda1_{lambda1}_lambda2_{lambda2}',
                f'{mz}.csv'
            )
        )
        total_peaks_number += len(peaks_df)
        samples = calibration_dataset.columns[5:]
        for sample in samples:
            if row[sample] == 0:
                continue
            sample_filename = labels[labels['sample_name'] == sample].iloc[0]['csv_file_name']
            filtered_peaks_df = peaks_df[
                (peaks_df['RT1_center'] > rt1 - rt1_tol)
                & (peaks_df['RT1_center'] < rt1 + rt1_tol)
                & (peaks_df['RT2_center'] > rt2 - rt2_tol)
                & (peaks_df['RT2_center'] < rt2 + rt2_tol)
            ]
            if sample_filename in filtered_peaks_df['csv_file_name'].values and row[sample] == 1:
                true_positives += 1
            elif (sample_filename in filtered_peaks_df['csv_file_name'].values and row[sample] == 0) or (sample_filename not in filtered_peaks_df['csv_file_name'].values and row[sample] == 1):
                wrong += 1
        total_hits += true_positives
        total_wrong += wrong

    print(f'Lambda1: {lambda1}, Lambda2: {lambda2} Done')

    return {
        'lambda1': lambda1,
        'lambda2': lambda2,
        'accuracy': total_hits / (total_hits + total_wrong),
        'total_peaks_number': total_peaks_number,
    }

def calibration_phase(cali_config):
    lambda1s = cali_config['lambda1s']
    lambda2s = cali_config['lambda2s']

    rt1_tol = cali_config['rt1_tol']
    rt2_tol = cali_config['rt2_tol']

    calibration_dataset = pd.read_csv(cali_config['calibration_dataset_path'])
    labels = pd.read_csv(cali_config['labels_path'])

    calibration_phase_output_dir = cali_config['calibration_phase_output_dir']

    create_folder_if_not_exists(calibration_phase_output_dir)

    accuracy_threshold = cali_config['accuracy_threshold']

    config = {
        "parallel_processing": True,
        "number_of_splits": 100,

        "mz_list_path": cali_config['mz_list_path'],
        "labels_path": cali_config['labels_path'],
        "m_z_column_name": "M/Z",
        "area_column_name": "Area",
        "first_time_column_name": "1st Time (s)",
        "second_time_column_name": "2nd Time (s)",
        "csv_file_name_column": "csv_file_name",
        "label_column_name": "label",

        "output_dir_TII_aligned": cali_config['TII_aligned_dir'],

        "lambda1": None,
        "lambda2": None,
        "peak_max_neighbor_distance": 5,
        "strict_noise_filtering": True,

        "enable_noisy_regions": True,
        "noisy_regions": [
            {
                "first_time_start": 0,
                "second_time_start": 0,
                "first_time_end": -1,
                "second_time_end": 1,
                "non_zero_ratio_region_threshold": 1e-3
            },
            {
                "first_time_start": 8700,
                "second_time_start": 1.1,
                "first_time_end": -1,
                "second_time_end": 1.8,
                "non_zero_ratio_region_threshold": 1e-2
            },
            {
                "first_time_start": 8700,
                "second_time_start": 0,
                "first_time_end": -1,
                "second_time_end": -1,
                "non_zero_ratio_region_threshold": 1e-2
            },
            {
                "first_time_start": 8690,
                "second_time_start": 2.2,
                "first_time_end": 8710,
                "second_time_end": 3,
                "non_zero_ratio_region_threshold": 1e-2
            },
            { # 202 EET
                "first_time_start": 5174-50,
                "second_time_start": 0,
                "first_time_end": 5174 + 50,
                "second_time_end": -1,
                "non_zero_ratio_region_threshold": 1e-1
            },

            { # 202 EET
                "first_time_start": 5300-50,
                "second_time_start": 0,
                "first_time_end": 5300 + 50,
                "second_time_end": 1.8,
                "non_zero_ratio_region_threshold": 1e-2
            },

            {
                "first_time_start": 7700-50,
                "second_time_start": 0,
                "first_time_end": 7700 + 50,
                "second_time_end": -1,
                "non_zero_ratio_region_threshold": 1e-1
            },
            {
                "first_time_start": 8700-50,
                "second_time_start": 0,
                "first_time_end": 8700 + 50,
                "second_time_end": -1,
                "non_zero_ratio_region_threshold": 1e-2
            }
        ],
        
        "convolution_filter": {
            "enable": False,
            "lambda3": 1000000,
            "rt1_window_size": 100,
            "rt2_window_size": 0.5,
            "rt1_stride": 20,
            "rt2_stride": 0.5,
            "non_zero_ratio_lambda3_filter": 0.9
        },

        "overall_filter": {
            "enable": True,
            # "lambda": 10,
            "non_zero_ratio_filter": 0.1
        },
    }

    hit_rates = []
    combinations = list(product(lambda1s, lambda2s))
    args_list = [(l1, l2, config, calibration_dataset, labels, calibration_phase_output_dir, rt1_tol, rt2_tol) 
                 for l1, l2 in combinations]
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_process_combination, args) for args in args_list]
        for future in as_completed(futures):
            hit_rates.append(future.result())

    hit_rates_df = pd.DataFrame(hit_rates)
    hit_rates_df.to_csv(os.path.join(calibration_phase_output_dir, 'hit_rates.csv'), index=False)

    # report the best lambda1 and lambda2 by filtering hit_rates_df by accuracy > 0.9 and then select the biggest lambda1 with accuracy > 0.9 and,
    # and then, among the same lambda1, select the median of lambda2 as the best lambda2
    # Filter hit_rates_df by accuracy > 0.9
    high_accuracy_df = hit_rates_df[hit_rates_df['accuracy'] > accuracy_threshold]

    if len(high_accuracy_df) > 0:
        # Select the biggest lambda1 with accuracy > 0.9
        best_lambda1 = int(high_accuracy_df['lambda1'].max())
        
        # Among the same lambda1, select the median of lambda2 as the best lambda2
        same_lambda1_df = high_accuracy_df[high_accuracy_df['lambda1'] == best_lambda1]
        best_lambda2 = int(same_lambda1_df['lambda2'].median())
        
        print(f'Best lambda1 (filtered): {best_lambda1}, Best lambda2 (median): {best_lambda2}')
    else:
        print(f'No configurations found with accuracy > {accuracy_threshold}. Please lower the accuracy threshold.')
        exit()

    # Find the best RT1thresh and RT2thresh
    rt1_thresholds = []
    rt2_thresholds = []

    for _, row in calibration_dataset.iterrows():
        mz = row['Base Mass']
        rt1 = row['RT1']
        rt2 = row['RT2']
        compound = row['Compound']

        peaks_df = pd.read_csv(
            os.path.join(
                calibration_phase_output_dir,
                f'lambda1_{best_lambda1}_lambda2_{best_lambda2}/peaks_lambda1_{best_lambda1}_lambda2_{best_lambda2}',
                f'{mz}.csv'
            )
        )

        

        filtered_peaks_df = peaks_df[
            (peaks_df['RT1_center'] > rt1 - rt1_tol)
            & (peaks_df['RT1_center'] < rt1 + rt1_tol)
            & (peaks_df['RT2_center'] > rt2 - rt2_tol)
            & (peaks_df['RT2_center'] < rt2 + rt2_tol)
        ]

        rt1_threshold = filtered_peaks_df['RT1_center'].max() - filtered_peaks_df['RT1_center'].min()
        rt2_threshold = filtered_peaks_df['RT2_center'].max() - filtered_peaks_df['RT2_center'].min()

        print(f'Compound: {compound}, RT1: {rt1}, RT2: {rt2}, RT1 Threshold: {rt1_threshold}, RT2 Threshold: {rt2_threshold}')
        rt1_thresholds.append(rt1_threshold.item())
        rt2_thresholds.append(rt2_threshold.item())



    best_rt1_threshold = math.ceil(max(rt1_thresholds))
    best_rt2_threshold = round(max(rt2_thresholds),1)

    best_config = {
        'lambda1': best_lambda1,
        'lambda2': best_lambda2,
        'rt1_threshold': round(best_rt1_threshold),
        'rt2_threshold': round(best_rt2_threshold, 1)
    }
    print(best_config)

    best_config_save_path = cali_config['best_config_save_path']

    create_folder_if_not_exists(best_config_save_path)
    # Save the best config in json
    with open(os.path.join(best_config_save_path, 'best_config.json'), 'w') as f:
        json.dump(best_config, f)