import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

class WeightVisualizer:
    def __init__(self, weight_samples, criteria_names):
        """
        weight_samples: numpy array of shape (n_samples, n_criteria)
        criteria_names: list of criteria names (length n_criteria)
        """
        self.weight_samples = weight_samples
        self.criteria_names = criteria_names

    def plot_distributions(self, **kwargs):
        """
        Plots the distributions of the weight samples for each criterion.

        kwargs:
            ncols: number of columns in the subplot grid (default: 3)
            xlim: tuple for x-axis limits (default: (0, 1))
            xticks: list or array for x-axis ticks (default: None)
            font_size: font size for titles and labels (default: 12)
            font_bold: whether to use bold font for titles and labels (default: True)
        """
        import matplotlib.pyplot as plt

        ncols = kwargs.get('ncols', 3)
        xlim = kwargs.get('xlim', (0, 1))
        xticks = kwargs.get('xticks', None)
        font_size = kwargs.get('font_size', 12)
        font_bold = kwargs.get('font_bold', True)

        fontweight = 'bold' if font_bold else 'normal'

        c_no = len(self.criteria_names)
        cols = ncols
        rows = int(np.ceil(c_no / cols))

        fig, axs = plt.subplots(rows, cols, figsize=(
            4 * cols, 3 * rows), sharey=True, sharex=True)
        axs = axs.flatten() if c_no > 1 else [axs]

        for idx in range(rows * cols):
            if idx < c_no:
                axs[idx].hist(self.weight_samples[:, idx], bins=50)
                mean_val = np.mean(self.weight_samples[:, idx])
                axs[idx].axvline(mean_val, color='red',
                                 linestyle='--', linewidth=2)
                axs[idx].set_title(
                    f"{self.criteria_names[idx]} (mean={mean_val:.3f})",
                    fontsize=font_size,
                    fontweight=fontweight
                )
                if xticks is not None:
                    axs[idx].set_xticks(xticks)
                axs[idx].set_xlim(xlim)
                axs[idx].tick_params(axis='both', labelsize=font_size)
                for label in (axs[idx].get_xticklabels() + axs[idx].get_yticklabels()):
                    label.set_fontweight(fontweight)
            else:
                fig.delaxes(axs[idx])

        plt.tight_layout()
        plt.show()

    def ridge_plot(self, **kwargs):
        """
        Plots the posterior weight distributions for each criterion using seaborn FacetGrid.

        kwargs:
            font_size: font size for titles and labels (default: 12)
            font_bold: whether to use bold font for titles and labels (default: True)
            palette: color palette for seaborn (default: "viridis")
        """
        font_size = kwargs.get('font_size', 14)
        font_bold = kwargs.get('font_bold', True)
        palette = kwargs.get('palette', "viridis")

        fontweight = 'bold' if font_bold else 'normal'

        # Convert to long-form DataFrame
        df_long = pd.DataFrame(self.weight_samples, columns=self.criteria_names).melt(
            var_name="Criterion", value_name="Weight"
        )

        # Calculate means for each criterion
        means = {name: self.weight_samples[:, idx].mean()
                 for idx, name in enumerate(self.criteria_names)}

        # Set plot style
        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        # Initialize the FacetGrid
        g = sns.FacetGrid(df_long, row="Criterion", hue="Criterion",
                          aspect=10, height=1.2, palette=palette)
        # Draw densities
        g.map(sns.kdeplot, "Weight",
              bw_adjust=1, clip_on=False, fill=True, alpha=0.8, linewidth=1.5)

        # Add a horizontal line at zero
        g.map(plt.axhline, y=0, lw=1, clip_on=False)

        # Add criteria names and means as text on each facet
        for ax, criterion in zip(g.axes.flat, self.criteria_names):
            mean_val = means[criterion]
            ax.text(
                0.98, 0.7,
                f"{criterion} (mean={mean_val:.3f})",
                transform=ax.transAxes,
                ha='right', va='center', fontsize=font_size, fontweight=fontweight, color='black'
            )
            # Plot red dotted line at mean
            ax.axvline(mean_val, color='red', linestyle='--',
                       linewidth=2, alpha=0.8)

        # Set titles and layout
        g.figure.subplots_adjust(hspace=-0.4)
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

        plt.xlabel("Posterior Weight", fontsize=font_size, fontweight=fontweight)
        plt.tight_layout()
        plt.show()

    def vertical_jakplot(self, **kwargs):
        """
        Plot vertical jakplot-style distributions for each criterion's weights.

        Parameters:
            samples (np.ndarray): shape (n_samples, n_criteria)
            criteria_names (list): list of criterion names
            font_size (int, optional): font size for labels and ticks
            font_bold (bool, optional): whether to use bold font
            ylim (tuple, optional): y-axis limits (min, max)
        """
        font_size = kwargs.get('font_size', 12)
        font_bold = kwargs.get('font_bold', False)
        ylim = kwargs.get('ylim', None)

        label_weight = 'bold' if font_bold else 'normal'

        n_criteria = self.weight_samples.shape[1]
        plt.figure(figsize=(7, 4))
        for idx in range(n_criteria):
            data = self.weight_samples[:, idx]
            # Estimate density using KDE
            kde = stats.gaussian_kde(data)
            y_grid = np.linspace(data.min(), data.max(), 300)
            pdf = kde(y_grid)
            # Normalize pdf for better visualization
            pdf = pdf / pdf.max() * 0.4  # scale width
            plt.plot(idx + pdf, y_grid, color='black', linewidth=3)
            plt.plot([idx], [data.mean()], 'o', color='blue')
        plt.xticks(range(n_criteria), self.criteria_names, fontsize=font_size, fontweight=label_weight)
        plt.xlabel('Criterion', fontsize=font_size, fontweight=label_weight)
        plt.ylabel('Weight', fontsize=font_size, fontweight=label_weight)
        if ylim is not None:
            plt.ylim(ylim)
        plt.tight_layout()
        plt.show()
    