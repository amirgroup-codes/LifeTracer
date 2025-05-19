import numpy as np

def find_center(coordinates, rt1_timestep, rt2_timestep):

    rt1_values = [round(coordinates[0] + round(rt1_timestep * i,3),3) for i in range(int((coordinates[2]-coordinates[0])/rt1_timestep)+1)]
    rt2_values = [round(coordinates[1] + round(rt2_timestep * i,3),3) for i in range(int((coordinates[3]-coordinates[1])/rt2_timestep)+1)]

    # reutrn median value
    return (rt1_values[(len(rt1_values))//2], rt2_values[(len(rt2_values))//2])

def find_center_indices(coordinates):
    return ((coordinates[2] + coordinates[0]) // 2, (coordinates[3] + coordinates[1]) // 2)

def relu(x):
    return np.maximum(0, x)

def generate_windows(index_start, index_end, index_window_size, index_stride, 
                     column_start, column_end, column_window_size, column_stride):
    """
    Generate a list of windows based on the specified index and column dimensions, window sizes, and strides.
    
    Parameters:
    - index_start (float): The starting value of the index.
    - index_end (float): The ending value of the index.
    - index_window_size (float): The window size for the index.
    - index_stride (float): The stride for moving to the next index window start.
    - column_start (float): The starting value of the column.
    - column_end (float): The ending value of the column.
    - column_window_size (float): The window size for the column.
    - column_stride (float): The stride for moving to the next column window start.
    
    Returns:
    - list of tuples: Each tuple contains ((index_start, index_end), (column_start, column_end)).
    """
    # Generate index windows
    index_windows = [(i, min(i + index_window_size, index_end)) for i in np.arange(index_start, index_end, index_stride)]

    # Generate column windows
    column_windows = [(j, min(j + column_window_size, column_end)) for j in np.arange(column_start, column_end, column_stride)]

    # Create a full list of combined windows (cross product)
    windows = [(index_window, column_window) for index_window in index_windows for column_window in column_windows]

    return windows