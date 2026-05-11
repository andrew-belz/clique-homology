import matplotlib.pyplot as plt
import numpy as np

def plot_null_distribution(
    dist: np.ndarray, 
    obs: float, 
    pval: float = None, 
    save_path: str = None,
    log: bool = True
) -> None:
    """Plot the null distribution and the observed distance.
    
    Args:
        dist (np.ndarray): Array of distances from the mean for the null distribution.
        obs (float): The observed distance.
        pval (float, optional): The calculated p-value to display in the title. Defaults to None.
        save_path (str, optional): File path to save the plot. If None, displays it. Defaults to None.
    """
    plt.figure(figsize=(8, 6))
    
    if log:
        # Create log-spaced bins for a log-log histogram, filtering out zeros
        dist_pos = dist[dist > 0]
        if len(dist_pos) > 0 and np.min(dist_pos) < np.max(dist_pos):
            bins = np.logspace(np.log10(np.min(dist_pos)), np.log10(np.max(dist_pos)), 50)
        else:
            bins = 'auto'
            
        plt.hist(dist, bins=bins, alpha=0.7, color='skyblue', edgecolor='black', label='Null Distribution', log=True)
        plt.xscale('log')
    else:
        # Plot the null distribution as a histogram
        plt.hist(dist, bins='auto', alpha=0.7, color='skyblue', edgecolor='black', label='Null Distribution')
    
    # Plot the observed distance as a vertical line
    plt.axvline(obs, color='red', linestyle='dashed', linewidth=2, label=f'Observed (dist={obs:.2f})')
    
    # Build the title
    title = 'Null Distribution of Betti Vector Distances'
    if pval is not None:
        title += f'\n(p-value = {pval:.4f})'
    plt.title(title)
    
    plt.xlabel('Distance from Mean (Standard Deviations)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
