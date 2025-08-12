import numpy as np
import seaborn as sns
import pandas as pd
from pandas.plotting import parallel_coordinates
from scipy.linalg import svd
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt

class ClusterVisualizer:
    def __init__(self, model, criteria_names=None, weights=None):
        """
        W_mean: np.ndarray, shape (DmNo, n_criteria)
        fit_clust: dict, must contain 'theta' (soft assignment, shape: (DmNo, DmC, ...))
        bwm_data: dict, must contain 'DmC' (number of clusters)
        """

        if weights is not None:
            self.W_mean = np.array(weights)
        else:
            self.W_mean = model['W'].mean(axis=2)

        self.DmC = model['wc'].shape[0]
        self.model = model
        self.criteria_labels = criteria_names if criteria_names else [f'C{i+1}' for i in range(self.W_mean.shape[1])]
        self.palette = sns.color_palette("Set2", n_colors=self.DmC)
        self.soft_assignment = model['theta'].mean(axis=2)
        self.hard_labels = np.argmax(self.soft_assignment, axis=1)

    def plot_projection(self, d_plot=2, **kwargs):
        """
        d_plot: int, number of dimensions for PCA plot (default: 2)
        fontsize: int, font size for labels/ticks (from kwargs, default: 12)
        title_fontsize: int, font size for title (from kwargs, default: 14)
        font_bold: bool, whether to use bold font for title (from kwargs, default: False)
        """

        font_size = kwargs.get('font_size', 12)
        title_fontsize = kwargs.get('title_fontsize', 14)
        bold = kwargs.get('font_bold', False)

        # CLR transform
        W_clr = np.log(self.W_mean) - np.mean(np.log(self.W_mean), axis=1, keepdims=True)
        W_clr_centered = W_clr - W_clr.mean(axis=0)
        U, S, Vt = svd(W_clr_centered, full_matrices=False)
        W_pca = U[:, :d_plot] * S[:d_plot]
        colors = plt.cm.tab10.colors

        if d_plot == 2:
            plt.figure(figsize=(7, 5))
            for cluster in range(self.DmC):
                idx = np.where(self.hard_labels == cluster)[0]
                plt.scatter(W_pca[idx, 0], W_pca[idx, 1], 
                            s=200, 
                            c=[colors[cluster]], 
                            edgecolor='k', 
                            label=f'Cluster {cluster+1}', 
                            zorder=2)
                for i in idx:
                    plt.text(W_pca[i, 0], W_pca[i, 1] + 0.03, f'DM {i+1}', fontsize=font_size, ha='center', va='bottom',
                             bbox=dict(facecolor='white', edgecolor='none', pad=0.5), zorder=3)
            plt.title('Projection of Decision Makers\' Weights', fontsize=title_fontsize, weight='bold' if bold else 'normal')
            plt.xlabel('PCA 1', fontsize=font_size)
            plt.ylabel('PCA 2', fontsize=font_size)
            plt.xticks(fontsize=font_size-2)
            plt.yticks(fontsize=font_size-2)
            plt.grid(True, linestyle='--', alpha=0.3, zorder=1)
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=self.DmC, frameon=True)
            plt.tight_layout()
            plt.show()

        elif d_plot == 3:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            for cluster in range(self.DmC):
                idx = np.where(self.hard_labels == cluster)[0]
                ax.scatter(W_pca[idx, 0], W_pca[idx, 1], W_pca[idx, 2],
                           s=200, 
                           c=[colors[cluster]], 
                           edgecolor='k', 
                           label=f'Cluster {cluster+1}', 
                           zorder=2)
                for i in idx:
                    ax.text(W_pca[i, 0], W_pca[i, 1], W_pca[i, 2] + 0.03, f'DM {i+1}', fontsize=font_size, ha='center', va='bottom')
            ax.set_title('Projection of Decision Makers\' Weights', fontsize=title_fontsize, weight='bold' if bold else 'normal')
            ax.set_xlabel('PCA 1', fontsize=font_size)
            ax.set_ylabel('PCA 2', fontsize=font_size)
            ax.set_zlabel('PCA 3', fontsize=font_size)
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=self.DmC, frameon=True)
            plt.tight_layout()
            plt.show()

        else:
            raise ValueError("d_plot must be 2 or 3")

    def plot_parallel_coordinates(self, **kwargs):
        """
        Plot a parallel coordinates plot of mean weights by cluster.

        Parameters (as kwargs):
            font_size: int, font size for labels/ticks (default: 12)
            title_fontsize: int, font size for title (default: 16)
            font_bold: bool, whether to use bold font for title (default: False)
        """
        fontsize = kwargs.get('font_size', 12)
        title_fontsize = kwargs.get('title_fontsize', 16)
        bold = kwargs.get('font_bold', False)

        df = pd.DataFrame(self.W_mean, columns=self.criteria_labels)
        df['Cluster'] = self.hard_labels.astype(str)
        plt.figure(figsize=(12, 6))
        parallel_coordinates(
            df, 'Cluster', color=self.palette, linewidth=2.5, alpha=0.7
        )
        plt.title('Parallel Coordinates Plot of Mean Weights by Cluster', fontsize=title_fontsize, weight='bold' if bold else 'normal')
        plt.xlabel('Criteria', fontsize=fontsize)
        plt.ylabel('Mean Weight', fontsize=fontsize)
        plt.xticks(fontsize=fontsize-1)
        plt.yticks(fontsize=fontsize-1)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(title='Cluster', fontsize=fontsize-1, title_fontsize=fontsize, loc='upper right', frameon=True)
        plt.tight_layout()
        plt.show()

    def plot_radar(self, **kwargs):
        """
        Plot a radar chart of mean weights for each cluster.

        Parameters (as kwargs):
            font_size: int, font size for labels/ticks (default: 12)
            title_fontsize: int, font size for title (default: 15)
            font_bold: bool, whether to use bold font for title (default: False)
        """
        fontsize = kwargs.get('font_size', 12)
        title_fontsize = kwargs.get('title_fontsize', 15)
        bold = kwargs.get('font_bold', False)

        num_clusters = self.DmC
        num_vars = len(self.criteria_labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += [angles[0]]  # Complete the loop

        plt.figure(figsize=(8, 7))
        for cluster in range(num_clusters):
            cluster_idx = np.where(self.hard_labels == cluster)[0]
            cluster_weights = self.W_mean[cluster_idx].mean(axis=0)
            cluster_weights = np.concatenate((cluster_weights, [cluster_weights[0]]))
            plt.polar(angles, cluster_weights, label=f'Cluster {cluster+1}', color=self.palette[cluster], linewidth=2)
            plt.fill(angles, cluster_weights, color=self.palette[cluster], alpha=0.15)

        plt.xticks(angles[:-1], self.criteria_labels, fontsize=fontsize)
        plt.yticks([])
        plt.title('Radar Plot: Mean Weights of All Clusters', fontsize=title_fontsize, weight='bold' if bold else 'normal')
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.tight_layout()
        plt.show()

    def plot_violin(self, **kwargs):
        """
        Plot violin plots showing the distribution of mean weights by cluster for each criterion.

        Parameters (as kwargs):
            font_size: int, font size for labels/ticks (default: 12)
            title_fontsize: int, font size for title (default: 15)
            font_bold: bool, whether to use bold font for title (default: False)
        """
        fontsize = kwargs.get('font_size', 12)
        title_fontsize = kwargs.get('title_fontsize', 15)
        bold = kwargs.get('font_bold', False)

        df_violin = pd.DataFrame(self.W_mean, columns=self.criteria_labels)
        df_violin['Cluster'] = self.hard_labels

        plt.figure(figsize=(14, 6))
        for i, crit in enumerate(self.criteria_labels):
            plt.subplot(1, len(self.criteria_labels), i+1)
            sns.violinplot(
                x='Cluster', y=crit, hue='Cluster', data=df_violin,
                palette=self.palette, inner='box', linewidth=1.2, legend=False
            )
            plt.title(crit, fontsize=fontsize, weight='bold' if bold else 'normal')
            if i == 0:
                plt.ylabel('Mean Weight', fontsize=fontsize)
            else:
                plt.ylabel('')
            plt.xlabel('Cluster', fontsize=fontsize)
            plt.xticks(fontsize=fontsize-2)
            plt.yticks(fontsize=fontsize-3)
            plt.tight_layout()

        plt.suptitle('Violin Plots: Distribution of Mean Weights by Cluster for Each Criterion',
                     fontsize=title_fontsize, weight='bold' if bold else 'normal', y=1.05)
        plt.show()
    
    def plot_cluster_heatmap(self, **kwargs):
        """
        Plot a heatmap of the mean weights for each decision maker, grouped by cluster.

        Parameters (as kwargs):
            font_size: int, font size for labels/ticks (default: 12)
            title_fontsize: int, font size for title (default: 15)
            font_bold: bool, whether to use bold font for title (default: False)
        """
        fontsize = kwargs.get('font_size', 12)
        title_fontsize = kwargs.get('title_fontsize', 15)
        bold = kwargs.get('font_bold', False)

        # Sort by cluster for better visualization
        sorted_idx = np.argsort(self.hard_labels)
        sorted_W = self.W_mean[sorted_idx]
        sorted_labels = self.hard_labels[sorted_idx]

        plt.figure(figsize=(10, 7))
        sns.heatmap(
            sorted_W,
            cmap="YlGnBu",
            cbar_kws={'label': 'Mean Weight'},
            xticklabels=self.criteria_labels,
            yticklabels=[f'DM {i+1} (C{c+1})' for i, c in zip(sorted_idx, sorted_labels)],
            linewidths=0.5,
            linecolor='gray'
        )
        plt.title('Heatmap of Mean Weights by Decision Maker (Clustered)', fontsize=title_fontsize, weight='bold' if bold else 'normal')
        plt.xlabel('Criteria', fontsize=fontsize)
        plt.ylabel('Decision Makers', fontsize=fontsize)
        plt.xticks(fontsize=fontsize-2, rotation=45, ha='right')
        plt.yticks(fontsize=fontsize-3)
        plt.tight_layout()
        plt.show()


    





