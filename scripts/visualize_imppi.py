import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle

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
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap of Cost Field
    im = ax.imshow(map_data, extent=[0, 10, 0, 10], origin='lower', cmap='magma', alpha=0.6)
    fig.colorbar(im, ax=ax, label='Total Cost (Goal + Information)')

    # High Interest Areas (Unknown Blobs)
    # Blob 1: x in [2, 4], y in [7, 9]
    rect1 = Rectangle((2, 7), 2, 2, linewidth=2, edgecolor='yellow', facecolor='yellow', alpha=0.3, label='High Interest (Unknown)')
    ax.add_patch(rect1)
    # Blob 2: x in [6, 8], y in [1, 3]
    rect2 = Rectangle((6, 1), 2, 2, linewidth=2, edgecolor='yellow', facecolor='yellow', alpha=0.3)
    ax.add_patch(rect2)

    # Goal
    goal_pos = (9.0, 5.0)
    ax.scatter(*goal_pos, marker='*', s=300, color='gold', edgecolor='black', label='Goal', zorder=10)

    # Obstacles (Ground Truth)
    # Wall at x=5, with opening at y=[4.0, 6.0]
    ax.vlines(5.0, 0, 4.0, colors='cyan', linestyles='-', linewidth=6, label='Wall')
    ax.vlines(5.0, 6.0, 10.0, colors='cyan', linestyles='-', linewidth=6)
    
    # Trajectories
    ax.plot(df_std['x'], df_std['y'], 'k--', label='Standard MPPI', linewidth=2.5, alpha=0.8)
    ax.plot(df_info['x'], df_info['y'], 'w-', label='Informative MPPI', linewidth=3.5)
    
    # Start point
    ax.scatter(df_info['x'].iloc[0], df_info['y'].iloc[0], c='lime', s=150, marker='o', label='Start', zorder=5, edgecolor='black')
    
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_title('I-MPPI: Goal-Directed Exploration with High-Interest Zones', fontsize=14)
    ax.legend(loc='upper left', framealpha=1.0, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    output_path = 'docs/_media/imppi_campaign.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved refined campaign plot to {output_path}")

if __name__ == "__main__":
    plot_everything()