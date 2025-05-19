import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Create a heatmap from a dataframe
def heatmap(df : pd.DataFrame, first_time_column_name : str = '1st Time (s)', second_time_column_name : 
            str = '2nd Time (s)', area_column_name : str = 'Area') -> pd.DataFrame:
    # Create heatmap data and then transpose it
    heatmap_data = df.pivot_table(values=area_column_name, index=first_time_column_name, columns=second_time_column_name, aggfunc='sum').T
    heatmap_data.fillna(0, inplace=True)
    # Reverse the order of rows to rotate 90 degrees counter-clockwise
    heatmap_data = heatmap_data.iloc[::-1]
    return heatmap_data

# Filter the dataframe by M/Z
def filter_by_m_z(df : pd.DataFrame,M_Z : float, threshold : float = 0.3, m_z_column_name : str = 'M/Z') -> pd.DataFrame:
    new_df = df[(df[m_z_column_name] > M_Z - threshold) & (df[m_z_column_name] < M_Z + threshold)].copy()
    return new_df

# Create a folder if it does not exist
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder created here: {folder_path}")
        os.makedirs(folder_path)

# Loads the heatmap data from the numpy files
def load_heatmap_data(heatmap_dir, m_z, sample):
    first_time = np.load(os.path.join(heatmap_dir, f'first_time.npy'))
    second_time = np.load(os.path.join(heatmap_dir, f'second_time.npy'))
    heatmap_2d = np.load(os.path.join(heatmap_dir, sample, f'{m_z}.npy'))
    ht_df = pd.DataFrame(heatmap_2d, index=second_time, columns=first_time)
    return ht_df
def load_headmaps_list(path_to_heatmaps, samples, m_z):
    heatmaps = []

    for sample_name in samples:
        ht_df = load_heatmap_data(path_to_heatmaps,m_z,sample_name)
        
        heatmaps.append(ht_df)
    return heatmaps

# Plot the heatmap
def plt_heatmap(path, ht_df, t1_start=0, t1_end=0, t2_start=0, t2_end=0, full_spectrum=False, cluster_rectangles=None, title='', save=False, round_num = True,small = False):
    # if ht_df is empty, return
    if ht_df.empty:
        return
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 14,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})
    # Create a custom colormap with the specified hex colors
    ht_df = ht_df.copy()

    if round_num:
        ht_df.columns = ht_df.columns.astype(int)
        new_columns = {col: round((int(col)),-2) for col in ht_df.columns if str(col).isnumeric()}
        ht_df.rename(columns=new_columns, inplace=True)

        # Format y-axis labels to one decimal place
        ht_df.index = ht_df.index.to_series().round(1)

    custom_colors = ['#000000', '#ff4f27', '#f4139c', '#6270e0', '#ffffff']
    cmap = LinearSegmentedColormap.from_list('custom_colormap', custom_colors)

    # Generate the heatmap with the custom colormap
    
    if small:
        plt.figure(figsize=(5, 3))
    else:
        plt.figure(figsize=(20, 4))
    ax = sns.heatmap(ht_df, cmap=cmap, rasterized=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

    if cluster_rectangles is not None:
        # Plot rectangles
        for (minr, minc, maxr, maxc) in cluster_rectangles:
            plt.plot([minc, maxc], [minr, minr], 'g-')  # Top line
            plt.plot([minc, maxc], [maxr, maxr], 'g-')  # Bottom line
            plt.plot([minc, minc], [minr, maxr], 'g-')  # Left line
            plt.plot([maxc, maxc], [minr, maxr], 'g-')  # Right line

    if not full_spectrum:
        # Zoom in on a specific portion
        ax.set_xlim(t1_start, t1_end)
        ax.set_ylim(t2_start, t2_end)

    # Generate the heatmap with the custom colormap and capture the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    # Format colorbar to scientific notation
    cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.tick_params(axis='y', labelsize=12)  # Set the y-axis label size

    # Adjust the position of the colorbar
    current_position = cbar.ax.get_position()
    new_position = [current_position.x0 - 0.03, current_position.y0, current_position.width, current_position.height]
    cbar.ax.set_position(new_position)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)  # Rotate xticks 90 degrees

    plt.title(title)
    if save:
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()