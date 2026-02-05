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
    map_data = np.loadtxt(path + 'info_map.csv', delimiter=',')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap of Info Gain (Cost)
    # Origin is (0,0), resolution 0.1, 100x100
    # Map data is (y, x) because of how it was saved in C++
    im = ax.imshow(map_data, extent=[0, 10, 0, 10], origin='lower', cmap='plasma', alpha=0.8)
    fig.colorbar(im, ax=ax, label='Informative Cost (Reward is negative)')

    # Obstacles (Ground Truth)
    # Wall at x=5, with opening at y=[4.5, 5.5]
    ax.vlines(5.0, 0, 4.5, colors='cyan', linestyles='-', linewidth=5, label='Wall')
    ax.vlines(5.0, 5.5, 10.0, colors='cyan', linestyles='-', linewidth=5)
    
    # Trajectories
    ax.plot(df_std['x'], df_std['y'], 'k--', label='Standard MPPI', linewidth=2, alpha=0.7)
    ax.plot(df_info['x'], df_info['y'], 'w-', label='Informative MPPI', linewidth=3)
    
    # Start point
    ax.scatter(df_info['x'].iloc[0], df_info['y'].iloc[0], c='green', s=100, label='Start', zorder=5)
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('I-MPPI: Information Gain Field and Trajectory Comparison')
    ax.legend(loc='upper right', framealpha=1.0)
    ax.grid(True, alpha=0.2)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    output_path = 'docs/_media/imppi_campaign.png'
    plt.savefig(output_path)
    print(f"Saved campaign plot to {output_path}")

if __name__ == "__main__":
    plot_everything()
