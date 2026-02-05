import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_everything():
    path = 'build/'
    files = ['std_traj.csv', 'info_traj.csv', 'info_map.csv']
    for f in files:
        if not os.path.exists(path + f):
            print(f"File {f} not found in {path}")
            return

    df_std = pd.read_csv(path + 'std_traj.csv')
    df_info = pd.read_csv(path + 'info_traj.csv')
    
    # Load map
    # info_map.csv is a matrix
    map_data = np.loadtxt(path + 'info_map.csv', delimiter=',')
    # The costs are lambda_info * IG + control_cost. 
    # Since lambda_info=50, and total_info is subtracted, map_data is mostly negative.
    # We want to visualize Information Gain, which is - (cost - control_cost) / lambda_info
    # But for a quick heatmap, just plot the cost values.
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap of Info Gain (Cost)
    # Origin is (0,0), resolution 0.1, 100x100
    im = ax.imshow(map_data, extent=[0, 10, 0, 10], origin='lower', cmap='viridis_r', alpha=0.6)
    fig.colorbar(im, ax=ax, label='Informative Cost (lower is better)')

    # Trajectories
    ax.plot(df_std['x'], df_std['y'], 'w--', label='Standard MPPI', linewidth=2)
    ax.plot(df_info['x'], df_info['y'], 'r-', label='Informative MPPI', linewidth=3)
    
    # Obstacles
    # Wall at x=5, with opening at y=[4.5, 5.5]
    ax.plot([5, 5], [0, 4.5], 'k-', linewidth=4)
    ax.plot([5, 5], [5.5, 10], 'k-', linewidth=4)
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('I-MPPI Exploration Campaign')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    output_path = 'docs/_media/imppi_campaign.png'
    plt.savefig(output_path)
    print(f"Saved campaign plot to {output_path}")

if __name__ == "__main__":
    plot_everything()