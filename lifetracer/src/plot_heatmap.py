from loguru import logger
import json
import pandas as pd
import os

from .utils.heatmap_utils import create_folder_if_not_exists, load_heatmap_data, plt_heatmap
import argparse

def plot_heatmap(config):
    output_dir_heatmap = config['output_dir_heatmap']
    csv_file_name_column = config['csv_file_name_column']
    labels_path = config['labels_path']
    m_z = config['m_z']
    plot_dir = os.path.join(config['plot_dir'], m_z)
    sample_name = config['sample_name']
    all_samples = config['all_samples']

    create_folder_if_not_exists(plot_dir)

    if all_samples:
        labels = pd.read_csv(labels_path)
        for sample in labels[csv_file_name_column].tolist():
            ht_df = load_heatmap_data(output_dir_heatmap, (str(m_z).replace('.','_')), sample)
            save_path = os.path.join(plot_dir, f'm_z_{m_z}_sample_{sample}_heatmap.pdf')
            plt_heatmap(save_path, ht_df, full_spectrum=True, title=f'Heatmap for {sample} with m/z {m_z}', save=True)
            logger.info(f'Heatmap for {sample} with m/z {m_z} plotted successfully!')
    else:
        ht_df = load_heatmap_data(output_dir_heatmap, (str(m_z).replace('.','_')), sample_name)
        save_path = os.path.join(plot_dir, f'm_z_{m_z}_sample_{sample_name}_heatmap.pdf')
        plt_heatmap(save_path, ht_df, full_spectrum=False, t1_start=400, t1_end=1200, t2_start=350, t2_end=150, title=f'Heatmap for {sample_name} with m/z {m_z}',small=True ,save=True)
        # plt_heatmap(save_path, ht_df, full_spectrum=True, title=f'Heatmap for {sample_name} with m/z {m_z}',small=False ,save=True)
        logger.info(f'Heatmap for {sample_name} with m/z {m_z} plotted successfully!')