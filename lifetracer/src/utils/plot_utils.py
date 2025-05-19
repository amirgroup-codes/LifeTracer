import random
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import seaborn as sns
import matplotlib.colors as mcolors
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.cm as cm
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
import matplotlib.font_manager as fm
from loguru import logger
import ast


font_path = '/usr/scratch/dtsui/Helvetica.ttf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
font_name = font_prop.get_name()
plt.rcParams['font.family'] = font_name

colors_abiotic = [
    "#E63946",
    "#F48C06",
    "#FFD60A",
    "#2EC4B6",
    "#00887A",
    "#00A8E8",
    "#264653",
    "#8E44AD",
]
# # Shuffle the colors with seed
# random.seed(102)
# random.shuffle(colors_biotic)
colors_biotic = [
  "#E69F00",  
  "#56B4E9",  
  "#009E73",  
  "#F0E442",  
  "#0072B2",  
  "#D55E00",  
  "#CC79A7",  
  "#000000",  
  "#999999",  
  "#8E44AD"  
]

def get_feature_name(i, signature_info):
    RT1 = eval(signature_info['RT1'])
    RT2 = eval(signature_info['RT2'])
    # average RT1 and RT2
    RT1 = ((RT1[0] + RT1[1]) / 2) / 60
    RT2 = (RT2[0] + RT2[1]) / 2

    return f'({signature_info["m/z"]:.1f}, {RT1:.1f}, {RT2:.1f})'


def plot_top_coefficients(signatures_info, coefficients_pvalues, results_dir, top_n=20):

    # Get the top 'n' feature importances
    top_features = coefficients_pvalues[:top_n]

    # Extract the importance values and feature names
    coeffs = [abs(coeff) for _, coeff in top_features]  # Use absolute for bar length
    feature_ids = [i for i in range(0, top_n)]
    features_names = [get_feature_name(i, signatures_info.iloc[i]) for i in range(0, top_n)]

    # reverse the order of the top features
    coeffs = coeffs[::-1]
    features_names = features_names[::-1]
    top_features = top_features[::-1]

    # Plot the feature importances horizontally
    plt.figure(figsize=(2, 4))  # Adjusted figure size for better display
    bars = plt.barh(feature_ids, coeffs, color=['#ff3333' if coeff < 0 else '#3c5488' for _, coeff in top_features])

    plt.yticks(feature_ids, features_names, fontsize=8)
    plt.ylabel('Feature')
    plt.xlabel('Coefficient')

    # Add border lines to left and bottom
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['top'].set_linewidth(0)
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['right'].set_linewidth(0)

    # Remove the background
    plt.gca().set_facecolor('white')

    # Display the plot
    plt.show()

    # Save the plot to a file
    plt.savefig(os.path.join(results_dir, f'top_{top_n}_coefficient.pdf'), format='pdf', bbox_inches='tight', dpi=300)


def plot_top_features(X_train, coefficients, train_samples, results_dir, label_column_name, csv_file_name_column, top_n_indices = [], type = 'top'):
    
    plt.rcParams.update({'font.size': 7.5})  # Set global font size
    
    X_selected = X_train[:, top_n_indices[:,0].astype(int)].copy()

    class_1_count = np.sum(train_samples[label_column_name] == 1)
    X_selected[class_1_count:,:] = np.where(X_selected[class_1_count:,:] == 1, 2, X_selected[class_1_count:,:])

    plt.figure(figsize=(2.534,12))

    # Define colors for each value
    if type == 'top':
        colors = ["white", "#3c5488", "#e64b35"]
    else:
        colors = ["#999999", "black", "black"]
    cmap = mcolors.ListedColormap(colors)

    if type == 'top':
        # sns.heatmap(X_selected.T, cmap=cmap, 
        #            yticklabels=top_n_indices[:,1], 
        #            xticklabels=train_samples['sample_name'].to_numpy(),
        #            cbar=False, 
        #            linecolor='gray', 
        #            linewidth=0.5,
        #            square=True)
        sns.heatmap(X_selected.T, cmap=cmap, 
                   yticklabels=range(1,len(top_n_indices)+1), 
                   xticklabels=train_samples['sample_name'].to_numpy(),
                   cbar=False, 
                   linecolor='gray', 
                   linewidth=0.5,
                   square=True)
    else:
        sns.heatmap(X_selected.T, cmap=cmap, 
                   xticklabels=train_samples['sample_name'].to_numpy(), 
                   yticklabels=top_n_indices[:,1], 
                   cbar=False, 
                   linecolor='white', 
                   linewidth=0.5,
                   square=True)
    
    # Adjust font sizes for tick labels
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    
    if type == 'top':
        plt.savefig(os.path.join(results_dir,f'top_features.pdf'), format="pdf", bbox_inches='tight', dpi=400)
    else:
        plt.savefig(os.path.join(results_dir,f'specific_features.pdf'), format="pdf", bbox_inches='tight', dpi=400)
    
    plt.close()  # Close the figure to free memory

def plot_all_features(X_train, coefficients, train_samples, results_dir, label_column_name, csv_file_name_column):
    
    sns.set(font_scale=1.2)

    # Define colors for each value
    colors = ["white", "#3c5488", "#e64b35"]
    cmap = mcolors.ListedColormap(colors)

    sns.heatmap(X_selected, cmap=cmap, xticklabels=top_n_indices[:,1], yticklabels=train_samples['sample_name'].to_numpy() ,cbar=False, linecolor='gray', linewidth=0.5,square=True)
    # Save the plot as a PDF file
    # plt.tight_layout()
    plt.savefig(os.path.join(results_dir,f'top_features.pdf'), format="pdf", bbox_inches='tight',dpi=400)

def add_arrow(x_start,y_start,x_end,y_end, text,line_color='#a1caf7'):
    plt.annotate(text,               # text to display
                 xy=(x_end, y_end),                     # point to annotate
                 xytext=(x_start, y_start),       # position of text
                 textcoords='offset points',    # how to interpret xytext
                 ha='right',                    # horizontal alignment
                 va='center',                   # vertical alignment
                 arrowprops=dict(arrowstyle='-', lw=1, color=line_color), # arrow style and color,
                 fontsize=15,
                 color=line_color)  # background box for text
    
def plot_pca(features, labels, samples_name ,coefficients_pvalues, results_dir, top_n = 20):
    X_selected = features[:, coefficients_pvalues[:top_n,0].astype(int)]

    pca = PCA(n_components=2)
    pca.fit(X_selected)
    X_embedded = pca.transform(X_selected)
    plt.figure(figsize=(4, 3))

    ax = plt.gca()  # Get current axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set the color of the x-axis and y-axis
    plt.gca().spines['bottom'].set_edgecolor('black')
    plt.gca().spines['left'].set_edgecolor('black')

    colors = np.array(colors_biotic + colors_abiotic).reshape(-1,1)
    
    hollow_square = get_marker('markers/square-svgrepo-com.svg')
    triangle_marker = get_marker('markers/shape-triangle-svgrepo-com.svg')

    markers = np.array([hollow_square]*10 + [triangle_marker]*8).reshape(-1,1)


    results = pd.DataFrame(np.concatenate([X_embedded,np.array(labels).reshape(len(labels),-1),np.array(samples_name).reshape(len(labels),-1),colors,markers],axis = 1),columns=['PC1','PC2','label','sample_name','color','marker'])


    for i in range(len(results)):
        plt.scatter(results['PC1'][i], results['PC2'][i], color=results['color'][i], label=results['sample_name'][i], s=200, marker=results['marker'][i])


    # for idx, sample_name in enumerate(samples_name):

    #     # Add text labels with arrows
    #     x_end, y_end = X_embedded[idx]
    #     random_directions = [random.choice([-1, 1]), random.choice([-1, 1])]
    #     add_arrow(x_end + random_directions[0] * np.random.randint(15, 80), x_end + random_directions[1] * np.random.randint(15, 80), x_end, y_end, sample_name)
    
    plt.xlabel('PC1', fontsize=15)
    plt.ylabel('PC2', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.gcf().set_facecolor('white')  # gcf() - get current figure
    plt.gca().set_facecolor('white')  # gca() - get current axis
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,'PCA_top20.pdf'),format='pdf',bbox_inches='tight',dpi=400)
    plt.close()

def plot_2d_features(all_signatures, peaks_features_df, num_clusters, result_dir):
    plt.figure(figsize=(15,5))

    # Assign colors based on the 'coefficient' column
    colors = ['#f8951e' if x < 0 else '#35b549' for x in peaks_features_df['coefficient']]

    # Plot the scatter plot with conditional colors
    plt.scatter(peaks_features_df['RT1'], peaks_features_df['RT2'], c=colors, s=10)

    plt.xlabel('RT1 (s)', fontsize=16)  # Increase font size for x-axis label
    plt.ylabel('RT2 (s)', fontsize=16)  # Increase font size for y-axis label

    plt.xticks(fontsize=16)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=16)  # Increase font size for y-axis ticks

    maxc_maxr = []

    i = 1
    colors_per_cluster = ['green']
    for cluster_id in range(1,num_clusters+1):
        rt1 =  eval(all_signatures.iloc[cluster_id-1]['RT1'])
        rt2 =  eval(all_signatures.iloc[cluster_id-1]['RT2'])

        minc = rt1[0] - 20
        maxc = rt1[1] + 20
        minr = rt2[0] - 0.01
        maxr = rt2[1] + 0.01

        #  {'Abiotic':'#f8951e', 'Biotic':'#35b549'}
        color = '#f8951e' if all_signatures.iloc[cluster_id-1]['coefficient'] < 0 else '#35b549'

        plt.plot([minc, maxc], [minr, minr], color)  # Top line
        plt.plot([minc, maxc], [maxr, maxr], color)  # Bottom line
        plt.plot([minc, minc], [minr, maxr], color)  # Left line
        plt.plot([maxc, maxc], [minr, maxr], color)  # Right line

        maxc_maxr.append((maxc,maxr))

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_edgecolor('black')
    ax.spines['bottom'].set_edgecolor('black')

    # Background color
    plt.gcf().set_facecolor('white')
    plt.gca().set_facecolor('white')

    plt.savefig(os.path.join(result_dir,'2d_plots_peaks_signatures.pdf'),format='pdf',bbox_inches='tight', dpi=300)
    plt.close()

def plot_3d_peaks_interactive(peaks_features_df, result_dir):
    peaks_features_df_sorted = peaks_features_df.sort_values(by=['m/z','RT1', 'RT2'], ascending=[True,True, True])
    # Create the initial scatter plot
    fig = px.scatter_3d(peaks_features_df_sorted, x='RT1', y='RT2', z='m/z', color='class', hover_data=['sample'])
    # Update marker size scaling factor if necessary
    fig.update_traces(marker=dict(sizemode='diameter', sizeref=2, sizemin=1,opacity = 0.5))
    # Update layout
    fig.update_layout(scene=dict(
        xaxis_title='1st Time',
        yaxis_title='2nd Time',
        zaxis_title='m/z',
        aspectratio=dict(x=4, y=1, z=1),
    ))
    fig.write_html(os.path.join(result_dir,'3d_plots_peaks_interactive.html'))

def plot_3d_signatures_interactive(all_signatures, result_dir): 
    rt1_center = []
    rt2_center = []
    for idx, signaure in all_signatures.iterrows():
        rt1 = eval(signaure['RT1'])
        rt2 = eval(signaure['RT2'])

        rt1_center.append((rt1[0] + rt1[1]) / 2)
        rt2_center.append((rt2[0] + rt2[1]) / 2)

    all_signatures['RT1_center'] = rt1_center
    all_signatures['RT2_center'] = rt2_center
    all_signatures['coefficient_abs'] = all_signatures['coefficient'].abs()

    all_signatures['class'] = all_signatures['class'].astype(str)

    all_signatures_sorted = all_signatures.sort_values(by=['m/z','RT1_center', 'RT2_center'], ascending=[True,True, True])

    # absolute value of the coefficient
    all_signatures_sorted['coefficient_abs'] = all_signatures_sorted['coefficient'].abs()
    # Create the initial scatter plot
    fig = px.scatter_3d(all_signatures_sorted, x='RT1_center', y='RT2_center', z='m/z', color='class', size='coefficient_abs', hover_data=['samples','RT1','RT2'])

    # Update marker size scaling factor if necessary
    fig.update_traces(marker=dict(sizemode='diameter', sizeref=2.*max(all_signatures_sorted['coefficient_abs'])/(4.**4), sizemin=1,opacity = 0.5))

    # Update layout
    fig.update_layout(scene=dict(
        xaxis_title='1st Time',
        yaxis_title='2nd Time',
        zaxis_title='m/z',
        aspectratio=dict(x=4, y=1, z=1),
    ))

    fig.write_html(os.path.join(result_dir,'3d_plots_signatures_interactive.html'))

def plot_3d_signatures(all_signatures, result_dir, view = 'small'):

    hollow_circle = get_marker('markers/noun-circle-hatch-empty-1625504.svg')


    all_signatures = all_signatures.copy()
    rt1_center = []
    rt2_center = []
    for idx, signaure in all_signatures.iterrows():
        rt1 = eval(signaure['RT1'])
        rt2 = eval(signaure['RT2'])

        rt1_center.append((rt1[0] + rt1[1]) / 2)
        rt2_center.append((rt2[0] + rt2[1]) / 2)

    all_signatures['RT1_center'] = rt1_center
    all_signatures['RT2_center'] = rt2_center
    all_signatures['coefficient_abs'] = all_signatures['coefficient'].abs()

    all_signatures['class'] = all_signatures['class'].astype(str)

    all_signatures = all_signatures.sort_values(by=['m/z','RT1_center', 'RT2_center'], ascending=[True,True, True])

    if view == 'small':
        legend_sizes = [500,
                        250,
                        125,
                        62.5,
                        31.25]
    else:
        legend_sizes = [5000,
                        2500,
                        1800,
                        800,
                        400]

    all_signatures['point_size'] = 0
    for idx, row in all_signatures.iterrows():

        if row['coefficient_abs'] >= 0.02:
            all_signatures.at[idx, 'point_size'] = legend_sizes[0]
        elif row['coefficient_abs'] >= 0.015:
            all_signatures.at[idx, 'point_size'] = legend_sizes[1]
        elif row['coefficient_abs'] >= 0.010:
            all_signatures.at[idx, 'point_size'] = legend_sizes[2]
        elif row['coefficient_abs'] >= 0.005:
            all_signatures.at[idx, 'point_size'] = legend_sizes[3]
        else:
            all_signatures.at[idx, 'point_size'] = legend_sizes[4]

    all_signatures = all_signatures.sort_values(by='point_size', ascending=False)

    # Create a dictionary to map class strings to specific colors
    color_map = {'0': '#e64b35', '1': '#3c5488'}

    if view == 'small':
        fig = plt.figure(figsize=(5.0022, 10))
    else:
        fig = plt.figure(figsize=(20, 20))

    ax = fig.add_subplot(111, projection='3d')

    # Convert class to string if not already and adjust RT1_center for minutes and rounding
    all_signatures['class'] = all_signatures['class'].astype(str)
    all_signatures['RT1_center'] = (all_signatures['RT1_center'] / 60).astype(int)

    # Scatter plot
    sc = ax.scatter(
        all_signatures['RT1_center'],
        all_signatures['RT2_center'],
        all_signatures['m/z'],
        c=[color_map[x] for x in all_signatures['class']],
        s=all_signatures['point_size'],
        alpha=1,
        edgecolors='w',
        linewidths=0.5,
        # marker=hollow_circle
    )

    # Adding labels and title
    ax.set_ylabel('2nd Time (s)', labelpad=1, fontsize=8)
    ax.set_xlabel('1st Time (min)', labelpad=1, fontsize=8)
    ax.set_zlabel('m/z', labelpad=1, fontsize=8)

    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=8)
    # Increase the space between tick labels and the axes
    ax.tick_params(axis='x', which='major', pad=1)
    ax.tick_params(axis='y', which='major', pad=1)
    ax.tick_params(axis='z', which='major', pad=1)

    # Set background color to white and edge color to black
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    if view == 'small':
        ax.set_box_aspect([6, 4, 3])
    else:
        ax.set_box_aspect([3, 4, 2])

    # Adjusting the view and plot appearance
    # Adjusting the view and plot appearance
    if view == 'small':
        ax.view_init(elev=20, azim=-120)
    else:
        ax.view_init(elev=10, azim=-179)
    plt.tight_layout()

    # Create legend handles
    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=np.sqrt(s), label=label)
                  for s, label in zip(legend_sizes, ['>= 0.02',  '0.015 - 0.02', '0.010 - 0.015', '0.005 - 0.010', ' < 0.005'])]

    # Add the legend to the plot
    legend1 = ax.legend(handles=legend_handles, handletextpad=2.5, fontsize='6', title_fontsize='14', handlelength=1, loc='upper right', borderpad=1.2, labelspacing=4, frameon=False, ncol=3)
    ax.add_artist(legend1)

    ax.legend(handles=[Patch(facecolor=color_map['0'], label='Abiotic'), Patch(facecolor=color_map['1'], label='Biotic')],
          fontsize='6', title_fontsize='14', handlelength=2, loc='upper left', borderpad=1.2, labelspacing=1.5, frameon=False)


    ax.set_ylim(0.2, 3.5)
    ax.set_xlim(20, 190)
    ax.set_zlim(30,650)

    # Background color
    plt.gcf().set_facecolor('white')
    plt.gca().set_facecolor('white')

    # Change the color of the gridlines
    ax.xaxis._axinfo['grid'].update(color = '#dcdcdc', linestyle = '-' , linewidth=0.5)
    ax.yaxis._axinfo['grid'].update(color = '#dcdcdc', linestyle = '-', linewidth=0.5)
    ax.zaxis._axinfo['grid'].update(color = '#dcdcdc', linestyle = '-', linewidth=0.5)

    plt.savefig(os.path.join(result_dir, 'Features_3d.pdf'), format='pdf')
    plt.close()



def plot_3d_feature_groups(all_signatures, result_dir, view = 'small'):

    hollow_circle = get_marker('markers/noun-circle-hatch-empty-1625504.svg')


    all_signatures = all_signatures.copy()
    rt1_center = []
    rt2_center = []
    for idx, signaure in all_signatures.iterrows():
        rt1 = eval(signaure['RT1'])
        rt2 = eval(signaure['RT2'])

        rt1_center.append((rt1[0] + rt1[1]) / 2)
        rt2_center.append((rt2[0] + rt2[1]) / 2)

    all_signatures['RT1_center'] = rt1_center
    all_signatures['RT2_center'] = rt2_center
    all_signatures['coefficient_abs'] = all_signatures['coefficient'].abs()

    all_signatures['class'] = all_signatures['class'].astype(str)

    all_signatures = all_signatures.sort_values(by=['m/z','RT1_center', 'RT2_center'], ascending=[True,True, True])

    if view == 'small':
        legend_sizes = [500,
                        250,
                        125,
                        31.25]
    else:
        legend_sizes = [5000,
                        2500,
                        1800,
                        400]

    all_signatures['point_size'] = 0
    for idx, row in all_signatures.iterrows():

        if row['coefficient_abs'] >= 0.03:
            all_signatures.at[idx, 'point_size'] = legend_sizes[0]
        elif row['coefficient_abs'] >= 0.02:
            all_signatures.at[idx, 'point_size'] = legend_sizes[1]
        elif row['coefficient_abs'] >= 0.01:
            all_signatures.at[idx, 'point_size'] = legend_sizes[2]
        else:
            all_signatures.at[idx, 'point_size'] = legend_sizes[3]

    all_signatures = all_signatures.sort_values(by='point_size', ascending=False)

    # Create a dictionary to map class strings to specific colors
    color_map = {'0': '#e64b35', '1': '#3c5488'}

    if view == 'small':
        fig = plt.figure(figsize=(5.0022, 10))
    else:
        fig = plt.figure(figsize=(20, 20))

    ax = fig.add_subplot(111, projection='3d')

    # Convert class to string if not already and adjust RT1_center for minutes and rounding
    all_signatures['class'] = all_signatures['class'].astype(str)
    all_signatures['RT1_center'] = (all_signatures['RT1_center'] / 60).astype(int)

    # Scatter plot
    sc = ax.scatter(
        all_signatures['RT1_center'],
        all_signatures['RT2_center'],
        all_signatures['m/z'],
        c=[color_map[x] for x in all_signatures['class']],
        s=all_signatures['point_size'],
        alpha=1,
        edgecolors='w',
        linewidths=0.5,
        # marker=hollow_circle
    )

    # Adding labels and title
    ax.set_ylabel('2nd Time (s)', labelpad=1, fontsize=8)
    ax.set_xlabel('1st Time (min)', labelpad=1, fontsize=8)
    ax.set_zlabel('m/z', labelpad=1, fontsize=8)

    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=8)
    # Increase the space between tick labels and the axes
    ax.tick_params(axis='x', which='major', pad=1)
    ax.tick_params(axis='y', which='major', pad=1)
    ax.tick_params(axis='z', which='major', pad=1)

    # Set background color to white and edge color to black
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    if view == 'small':
        ax.set_box_aspect([6, 4, 3])
    else:
        ax.set_box_aspect([3, 4, 2])

    # Adjusting the view and plot appearance
    # Adjusting the view and plot appearance
    if view == 'small':
        ax.view_init(elev=20, azim=-120)
    else:
        ax.view_init(elev=10, azim=-179)
    plt.tight_layout()

    # Create legend handles
    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=np.sqrt(s), label=label)
                  for s, label in zip(legend_sizes, ['>= 0.03',  '0.02 - 0.03', '0.01 - 0.02', ' < 0.01'])]

    # Add the legend to the plot
    legend1 = ax.legend(handles=legend_handles, handletextpad=2.5, fontsize='6', title_fontsize='14', handlelength=1, loc='upper right', borderpad=1.2, labelspacing=4, frameon=False, ncol=4)
    ax.add_artist(legend1)

    ax.legend(handles=[Patch(facecolor=color_map['0'], label='Abiotic'), Patch(facecolor=color_map['1'], label='Biotic')],
          fontsize='6', title_fontsize='14', handlelength=2, loc='upper left', borderpad=1.2, labelspacing=1.5, frameon=False)


    ax.set_ylim(0.2, 3.5)
    ax.set_xlim(20, 190)
    ax.set_zlim(30,650)

    # Background color
    plt.gcf().set_facecolor('white')
    plt.gca().set_facecolor('white')

    # Change the color of the gridlines
    ax.xaxis._axinfo['grid'].update(color = '#dcdcdc', linestyle = '-' , linewidth=0.5)
    ax.yaxis._axinfo['grid'].update(color = '#dcdcdc', linestyle = '-', linewidth=0.5)
    ax.zaxis._axinfo['grid'].update(color = '#dcdcdc', linestyle = '-', linewidth=0.5)

    plt.savefig(os.path.join(result_dir, 'Group_Features_3d.pdf'), format='pdf')
    plt.close()


import matplotlib.markers as mmarkers

def create_half_filled_marker(rotation=45):
    # Define the marker path
    marker = mmarkers.MarkerStyle(marker='o')
    path = marker.get_path().transformed(marker.get_transform())

    # Create a wedge to represent the half-filled portion
    wedge = mmarkers.MarkerStyle(marker='|')
    wedge_path = wedge.get_path().transformed(Affine2D().scale(1.5, 1.5).rotate_deg(rotation))

    # Combine the paths
    combined_path = path.intersect_path(wedge_path)
    
    return combined_path

def get_marker(svg_path):
    cwd = os.getcwd()
    custom_path, attributes = svg2paths(svg_path)
    marker = parse_path(attributes[0]['d'])
    marker.vertices -= marker.vertices.mean(axis=0)
    marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
    marker = marker.transformed(mpl.transforms.Affine2D().scale(-1,1))
    return marker


def plot_3d_peaks(peaks_features_df, samples,result_dir, label = 'biotic', view = 'small'):
    peaks_features_df = peaks_features_df.copy()
    if label == 'biotic':
        selected_samples = samples[samples['label']==1]['csv_file_name']
    else:
        selected_samples = samples[samples['label']==0]['csv_file_name']
    peaks_features_df = peaks_features_df[peaks_features_df['sample'].isin(selected_samples)]

    peaks_features_df['point_size'] = 2000
    
    import matplotlib.path as mpath

    verts = [
        (1,-1), #lower right
        (-1,-1), #lower left 
        (-1,1), #upper left 
        (1,1), #upper right
        (1,-1), #lower right
    ]

    codes = [
        Path.MOVETO, #begin the figure in the lower right
        Path.CURVE3, #start a 3 point curve with the control point in lower left
        Path.LINETO, #end curve in the upper left
        Path.CURVE3, #start a new 3 point curve with the upper right as a control point
        Path.LINETO, #end curve in lower right
    ]

    oval = Path(verts,codes)

    hollow_circle = get_marker('markers/noun-circle-hatch-empty-1625504.svg')
    diamond = get_marker('markers/diamond-figure-form-geometry-graphic-line-svgrepo-com.svg')
    hollow_square = get_marker('markers/geometry-shape-square-figure-form-graphic-svgrepo-com.svg')

    
    # markers_biotic = [Hollow_Plus_marker, leaf_marker, pentagon_marker, hollow_circle, diamond, triangle_marker, 'x',star_marker,'+',hollow_square]
    markers_biotic = [hollow_circle,hollow_square,diamond,hollow_circle,
                      hollow_square,diamond,hollow_circle,hollow_square,
                      diamond,hollow_circle]
    
    # Shuffle the markers and colors with seed
    # random.seed(102)
    # random.shuffle(colors_biotic)

    # colors_biotic = ["#000", "#000", "#000", "#000", "#000", "#000", "#000", "#000", "#000", "#000"]

    markers_abiotic = [hollow_circle,hollow_square,diamond,hollow_circle,
                      hollow_square,diamond,hollow_circle,hollow_square]

    if label == 'biotic':
        markers = markers_biotic
        colors = colors_biotic
    else:
        markers = markers_abiotic
        colors = colors_abiotic

    if view == 'small':
        fig = plt.figure(figsize=(5.0022, 10))
    else:
        fig = plt.figure(figsize=(7.08, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Convert class to string if not already and adjust RT1_center for minutes and rounding
    peaks_features_df['class'] = peaks_features_df['class'].astype(str)
    peaks_features_df['RT1'] = (peaks_features_df['RT1'] / 60).astype(int)

    # Scatter plot for each sample with unique marker
    for i, sample in enumerate(selected_samples):
        sample_df = peaks_features_df[(peaks_features_df['sample'] == sample)]
        label = samples[samples['csv_file_name'] == sample]['label'].values[0]
        sample_name = samples[samples['csv_file_name'] == sample].iloc[0]['sample_name']

        ax.scatter(
            sample_df['RT1'],
            sample_df['RT2'],
            sample_df['m/z'],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=10,
            alpha=0.8,
            # edgecolors='w',
            label=sample_name
        )

    # Adding labels and title
    ax.set_ylabel('2nd Time (s)', labelpad=1, fontsize=8)
    ax.set_xlabel('1st Time (min)', labelpad=1, fontsize=8)
    ax.set_zlabel('m/z', labelpad=1, fontsize=8)

    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=8)
    # Increase the space between tick labels and the axes
    ax.tick_params(axis='x', which='major', pad=1)
    ax.tick_params(axis='y', which='major', pad=1)
    ax.tick_params(axis='z', which='major', pad=1)

    if view == 'small':
        ax.set_box_aspect([6, 4, 3])
    else:
        ax.set_box_aspect([6, 4, 3])

    # Adjusting the view and plot appearance
    # Adjusting the view and plot appearance
    if view == 'small':
        ax.view_init(elev=20, azim=-120)
    else:
        ax.view_init(elev=20, azim=-120)
    plt.tight_layout()

    ax.set_ylim(0.2, 3.5)
    ax.set_xlim(35, 190)
    ax.set_zlim(30,650)

    # Background color
    plt.gcf().set_facecolor('white')
    plt.gca().set_facecolor('white')

    # Change the color of the gridlines
    ax.xaxis._axinfo['grid'].update(color = '#dcdcdc', linestyle = '-' , linewidth=0.5)
    ax.yaxis._axinfo['grid'].update(color = '#dcdcdc', linestyle = '-', linewidth=0.5)
    ax.zaxis._axinfo['grid'].update(color = '#dcdcdc', linestyle = '-', linewidth=0.5)

    # Make the edges of the plot black
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # Add legend at the bottom of the plot
    if label == 'biotic':
        ax.legend(handletextpad=1, fontsize='8', title_fontsize='8', handlelength=1, loc='upper right', borderpad=1.2, labelspacing=1, frameon=False, ncols=5)
    else:
        ax.legend(handletextpad=1, fontsize='8', title_fontsize='8', handlelength=1, loc='upper right', borderpad=1.2, labelspacing=1, frameon=False, ncols=4)

    # Adjust layout to make room for the legend
    # Adjust layout to make room for the legend
    # plt.subplots_adjust(bottom=0.25,left=0.25,right=0.75,top=0.75)

    plt.savefig(os.path.join(result_dir, f'Peaks_3d_{label}.pdf'), format='pdf')
    plt.close()

def plot_distribution_of_peaks(peaks_features_df, result_dir, x_axis = 'm/z', separate=True):
    peaks_features_df = peaks_features_df.copy()
    peaks_features_df['RT1'] = (peaks_features_df['RT1'] / 60)
    # Define the color palette
    palette = ['#e64b35','#3c5488']
    # palette = ['#f8951f','#35b548']

    # Define the number of bins and calculate the bin edges
    if x_axis == 'm/z':
        # bin_edges = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
        bin_edges = list(range(0,750,50))
    elif x_axis == 'RT1':
        # bin_edges = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,19
        bin_edges = list(range(30,200,10))

        # print(peaks_features_df['RT1'].max(),peaks_features_df['RT1'].min())
    elif x_axis == 'RT2':
        # 0.2 bars
        bin_edges = [round(i/5, 1) for i in range(0, 38)]

        print(peaks_features_df['RT2'].max(),peaks_features_df['RT2'].min())

    if separate:
        # Plot biotic and abiotic separately
        classes = ['0', '1']  # 0 = abiotic, 1 = biotic
        class_names = ['abiotic', 'biotic']
        
        for i, cls in enumerate(classes):
            # Filter data for this class
            class_df = peaks_features_df[peaks_features_df['class'] == cls]
            
            # Create a figure and axis with specified size
            fig, ax = plt.subplots(figsize=(2.2, 1.3))

            # Remove the top and right borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Create the histogram plot with single class (no hue needed)
            sns.histplot(data=class_df, x=x_axis, color=palette[i], alpha=0.85, bins=bin_edges, linewidth=0.2)

            # Customizing x-axis tick labels to show fewer labels
            tick_labels = [f'{bin_edges[i]}-{bin_edges[i+1]}' for i in range(len(bin_edges)-1)]
            ax.set_xticks([(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])

            # Show only every 2nd or 3rd label
            if x_axis == 'RT2':
                reduced_tick_labels = [tick_labels[i] if i % 3 == 0 else '' for i in range(len(tick_labels))]
            elif x_axis =='RT1':
                reduced_tick_labels = [tick_labels[i] if i % 2 == 0 else '' for i in range(len(tick_labels))]
            else:
                reduced_tick_labels = [tick_labels[i] if i % 2 == 0 else '' for i in range(len(tick_labels))]
            ax.set_xticklabels(reduced_tick_labels, rotation=90, ha='center', fontsize=8)

            ax.tick_params(axis='both', which='major', color='black', length=3, width=0.5, labelsize=8)

            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)

            # Limit the x-axis and y-axis range
            if x_axis == 'm/z':
                ax.set_xlim(0, peaks_features_df['m/z'].max()+51)
                ax.set_xlabel('', fontsize=8)
            elif x_axis == 'RT1':
                ax.set_xlim(peaks_features_df['RT1'].min()-5, peaks_features_df['RT1'].max()+0.5)
                ax.set_xlabel('', fontsize=8)
            else:
                ax.set_xlim(peaks_features_df['RT2'].min()-0.2, peaks_features_df['RT2'].max()+0.2)
                ax.set_xlabel('', fontsize=8)

            ax.set_ylabel('', fontsize=8)
            ax.set_ylim(0, None)

            # Set the background color
            fig.set_facecolor('white')
            ax.set_facecolor('white')

            # Remove grid lines
            ax.grid(False)

            # Adjust spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_edgecolor('black')
            ax.spines['bottom'].set_edgecolor('black')

            # Save the plot
            if x_axis == 'm/z':
                plt.savefig(os.path.join(result_dir, f'distribution_of_peaks_mz_{class_names[i]}.pdf'), format='pdf', bbox_inches='tight', dpi=400)
            elif x_axis == 'RT1':
                plt.savefig(os.path.join(result_dir, f'distribution_of_peaks_RT1_{class_names[i]}.pdf'), format='pdf', bbox_inches='tight', dpi=400)
            else:
                plt.savefig(os.path.join(result_dir, f'distribution_of_peaks_RT2_{class_names[i]}.pdf'), format='pdf', bbox_inches='tight', dpi=400)
            
            plt.close()
    else:
        # Original code for combined plot
        # Create a figure and axis with specified size
        fig, ax = plt.subplots(figsize=(2.2, 1.3))

        # Remove the top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Create the histogram plot
        sns.histplot(data=peaks_features_df, x=x_axis, hue="class", palette=palette, alpha=0.85, bins=bin_edges, linewidth=0.2)

        # Customizing x-axis tick labels to show fewer labels
        tick_labels = [f'{bin_edges[i]}-{bin_edges[i+1]}' for i in range(len(bin_edges)-1)]
        ax.set_xticks([(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])

        # Show only every 2nd label
        if x_axis == 'RT2':
            reduced_tick_labels = [tick_labels[i] if i % 3 == 0 else '' for i in range(len(tick_labels))]
        elif x_axis =='RT1':
            reduced_tick_labels = [tick_labels[i] if i % 2 == 0 else '' for i in range(len(tick_labels))]
        else:
            reduced_tick_labels = [tick_labels[i] if i % 2 == 0 else '' for i in range(len(tick_labels))]
        ax.set_xticklabels(reduced_tick_labels, rotation=90, ha='center', fontsize=8)

        ax.tick_params(axis='both', which='major', color='black', length=3, width=0.5, labelsize=8)

        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        # Limit the x-axis and y-axis range
        if x_axis == 'm/z':
            ax.set_xlim(0, peaks_features_df['m/z'].max()+51)
            ax.set_xlabel('', fontsize=8)
        elif x_axis == 'RT1':
            ax.set_xlim(peaks_features_df['RT1'].min()-5, peaks_features_df['RT1'].max()+0.5)
            ax.set_xlabel('', fontsize=8)
        else:
            ax.set_xlim(peaks_features_df['RT2'].min()-0.2, peaks_features_df['RT2'].max()+0.2)
            ax.set_xlabel('', fontsize=8)

        ax.set_ylabel('', fontsize=8)
        ax.set_ylim(0, None)

        # Set the background color
        fig.set_facecolor('white')
        ax.set_facecolor('white')

        # Remove grid lines
        ax.grid(False)

        # Adjust spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_edgecolor('black')
        ax.spines['bottom'].set_edgecolor('black')

        # Remove legend
        ax.get_legend().remove()

        if x_axis == 'm/z':
            plt.savefig(os.path.join(result_dir, 'distribution_of_peaks_mz.pdf'), format='pdf', bbox_inches='tight', dpi=400)
        elif x_axis == 'RT1':
            plt.savefig(os.path.join(result_dir, 'distribution_of_peaks_RT1.pdf'), format='pdf', bbox_inches='tight', dpi=400)
        else:
            plt.savefig(os.path.join(result_dir, 'distribution_of_peaks_RT2.pdf'), format='pdf', bbox_inches='tight', dpi=400)

        # Show the plot
        plt.close()

def save_latex_table(top_feature_group_indices, result_dir):
    # Create a LaTeX table
    latex_table = ""
    
    # Add header row
    for index, feature_group in enumerate(top_feature_group_indices[:45]):
        mz, rt1, rt2 = ast.literal_eval(feature_group[1])
        latex_table += f"{index+1} & {mz} & {rt1} & {rt2} \\\\ \n"

    # Save the LaTeX table to a file
    with open(os.path.join(result_dir, 'top_features_table.tex'), 'w') as f:
        f.write(latex_table)

