import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import networkx as nx
from graphviz import Digraph
from IPython.display import display

class CredalRanking:
    def __init__(self, weight_samples, **kwargs):
        """
        Initializes the CredalRanking object.

        Parameters
        ----------
        weight_samples : np.ndarray
            A 2D numpy array of shape (num_samples, num_criteria) containing sampled weights for each criterion.
        criteria_name : list of str, optional
            List of criterion names. If not provided, defaults to ['C1', 'C2', ..., 'Cn'] where n = number of criteria.

        Raises
        ------
        AssertionError
            If the length of criteria_name does not match the number of criteria in weight_samples.
        """

        if 'criteria_name' in kwargs:
            self.criteria_names = kwargs['criteria_names'].copy()
        else:
            self.criteria_names = [f'C{i+1}' for i in range(weight_samples.shape[1])]

        assert len(self.criteria_names) == weight_samples.shape[1], "Invalid number of criteria"

        self.weight_samples = weight_samples
        self.avg_weights = np.mean(self.weight_samples, axis=0)
        self.index = np.argsort(-self.avg_weights)
        assert len(self.criteria_names) == self.weight_samples.shape[1], "Invalid number of criteria"
        self.sample_no, self.c_no = self.weight_samples.shape
        # for i in range(self.c_no):
        #     self.criteria_names[i] = self.criteria_names[i] + ' - ' + str(round(self.avg_weights[i],3))
        self.probs = np.empty((self.c_no, self.c_no))
        for i in range(self.c_no):
            for j in range(i, self.c_no):
                self.probs[i,j] = round((self.weight_samples[:,i] >= self.weight_samples[:,j]).sum() / self.sample_no,2)
                self.probs[j,i] = 1 - self.probs[i,j]

        # Edge styles as a class property
        self.edge_styles = [
            {'range': (0.5, 0.6), 'color': 'lightblue', 'style': 'dotted', 'label': '0.5–0.6', 'width': 1},
            {'range': (0.6, 0.7), 'color': 'deepskyblue', 'style': 'dashdot', 'label': '0.6–0.7', 'width': 2},
            {'range': (0.7, 0.8), 'color': 'dodgerblue', 'style': 'dashed', 'label': '0.7–0.8', 'width': 2.5},
            {'range': (0.8, 0.9), 'color': 'blue', 'style': 'solid', 'label': '0.8–0.9', 'width': 2.5},
            {'range': (0.9, 1.01), 'color': 'navy', 'style': 'solid', 'label': '0.9–1.0', 'width': 3}
        ]
    
    def credal_probs(self):
        return pd.DataFrame(self.probs, index=self.criteria_names, columns=self.criteria_names) 
    
    def plot(self, **kwargs):
        """
        Visualizes the credal ranking of criteria using a directed graph.
        ----------
        show_weight : bool, optional
            If True, displays the probability value on each edge. Default is False.
        **kwargs
            Additional keyword arguments for future extensions.
        Notes
        -----
        - Nodes are positioned in a circle for clarity, with labels placed outside the circle.
        - If show_weight is True, edge weights are displayed as labels.
        - Otherwise, Edge styles (color, width, linestyle) are determined by the probability intervals in `self.edge_styles`.
        - A legend is included to explain edge styles unless `show_weight` is True.
        """

        show_probs = kwargs.get('show_probs', False)
        avg_weights = self.avg_weights
        # criteria_names = [name.split(' - ')[0] for name in self.criteria_names]
        node_labels = {name: f"{name}\n({avg_weights[i]:.2f})" for i, name in enumerate(self.criteria_names)}

        # Create directed graph with criteria_names as nodes
        G = nx.DiGraph()
        G.add_nodes_from(self.criteria_names)

        # Add edges based on self.probs
        for i, src in enumerate(self.criteria_names):
            for j, tgt in enumerate(self.criteria_names):
                if i != j:
                    prob = self.probs[i, j]
                    for style in self.edge_styles:
                        lo, hi = style['range']
                        if lo <= prob < hi:
                            G.add_edge(src, tgt, weight=prob, color=style['color'], style=style['style'])
                            break

        # Sort nodes by descending average weight
        sorted_indices = np.argsort(-avg_weights)
        sorted_nodes = [self.criteria_names[i] for i in sorted_indices]

        n = len(sorted_nodes)
        radius = 1.0
        label_offset = 0.15

        # Evenly spaced angles from π/2 (top) to -3π/2 (full circle)
        base_angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)

        # Alternate angles around the top for better label distribution
        angle_order = [0]
        for k in range(1, n):
            if k % 2 == 1:
                angle_order.append((k + 1) // 2)
            else:
                angle_order.append(-(k // 2))
        angles = [base_angles[(i % n)] for i in angle_order]

        # Assign positions
        circle_pos = {}
        label_pos = {}
        for node, angle in zip(sorted_nodes, angles):
            x = np.cos(angle) * radius
            y = np.sin(angle) * radius
            circle_pos[node] = (x, y)
            lx = np.cos(angle) * (radius + label_offset)
            ly = np.sin(angle) * (radius + label_offset)
            label_pos[node] = (lx, ly)

        # Draw nodes
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(G, circle_pos, node_color='white', edgecolors='black', node_size=600)

        # Draw edges by style
        for style in self.edge_styles:
            edges = []
            edge_labels = {}
            for u, v, d in G.edges(data=True):
                prob = d['weight']
                lo, hi = style['range']
                if lo <= prob < hi:
                    edges.append((u, v))
                    if show_probs:
                        edge_labels[(u, v)] = f"{prob:.2f}"

            if edges:
                if show_probs:
                    nx.draw_networkx_edges(
                        G, circle_pos, edgelist=edges,
                        edge_color=style['color'],
                        width=style['width'],
                        arrows=True,
                        arrowsize=15,
                        connectionstyle='arc3,rad=0.05',
                        arrowstyle='-|>',
                    )
                    nx.draw_networkx_edge_labels(
                        G,
                        pos=circle_pos,
                        edge_labels=edge_labels,
                        font_color=style['color'],
                        font_size=9,
                        label_pos=0.5,
                        rotate=False,
                        bbox=dict(alpha=0.9, color='white', edgecolor='none', pad=0.1)
                    )
                else:
                    nx.draw_networkx_edges(
                        G, circle_pos, edgelist=edges,
                        edge_color=style['color'],
                        style=style['style'],
                        width=style['width'],
                        arrows=True,
                        arrowsize=15,
                        connectionstyle='arc3,rad=0.01',
                        arrowstyle='-|>',
                    )

        # Draw labels outside the circles, with average weights
        for node, (lx, ly) in label_pos.items():
            plt.text(lx, ly, node_labels[node], fontsize=13, ha='center', va='center',
                     bbox=dict(facecolor='white', edgecolor='none', pad=0.5))

        # Add legend
        if not show_probs:
            legend_elements = [
                Line2D([0], [0], color=style['color'], linestyle=style['style'], lw=2, label=style['label'])
                for style in self.edge_styles
            ]
            plt.legend(
                handles=legend_elements,
                title="Edge Weights",
                loc="upper center",
                bbox_to_anchor=(0.5, 1.08),
                ncol=len(legend_elements),
                frameon=False
            )
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_ranking(self, **kwargs):
        '''
        Visualizes the credal ranking of criteria using a Graphviz directed graph.
        This method creates a directed graph where each node represents a criterion, labeled with its name and average weight.
        Node appearance (font size, boldness, fill color, etc.) and edge appearance (font size, boldness) can be customized via keyword arguments.
        The width of each node is dynamically estimated based on the label length for better readability.
        Edges are drawn between nodes if the probability of one criterion outranking another exceeds a threshold (default 0.5).
        For probabilities equal to 1, edges are only drawn between adjacent criteria in the sorted order of average weights.
        The resulting graph is displayed inline (e.g., in a Jupyter notebook).
        Keyword Args:
            node_fontsize (int or str): Font size for node labels (default: '14').
            edge_fontsize (int or str): Font size for edge labels (default: '14').
            node_boldfont (bool): Whether to use bold font for node labels (default: False).
            edge_boldfont (bool): Whether to use bold font for edge labels (default: False).
            graph_layout (str): Graphviz rank direction, e.g., 'TB' (top-bottom), 'LR' (left-right) (default: 'TB').
            node_fillcolor (str): Fill color for nodes (default: 'white').
        Explanation:
            - Nodes represent criteria, labeled with their names and average weights.
            - Node width is estimated based on label length for better visualization.
            - Edges are added if the probability of one criterion outranking another is greater than 0.5.
            - For probabilities exactly equal to 1, edges are only drawn between adjacent criteria in the sorted ranking.
            - Node and edge appearance can be customized via keyword arguments.
            - The graph is rendered and displayed using Graphviz.
        '''
        # Handle font size and bold fontface kwargs
        node_fontsize = kwargs.get('node_fontsize', '14')
        edge_fontsize = kwargs.get('edge_fontsize', '14')
        if isinstance(node_fontsize, int):
            node_fontsize = str(node_fontsize)
        if isinstance(edge_fontsize, int):
            edge_fontsize = str(edge_fontsize)

        node_boldfont = kwargs.get('node_boldfont', False)
        edge_boldfont = kwargs.get('edge_boldfont', False)
        node_fontname = 'Helvetica-Bold' if node_boldfont else 'Helvetica'
        edge_fontname = 'Helvetica-Bold' if edge_boldfont else 'Helvetica'

        graph_layout = kwargs.get('graph_layout', 'TB')
        node_fillcolor = kwargs.get('node_fillcolor', 'white')

        def estimate_width(label, char_width=0.13, min_width=1, max_width=3):
            width = max(min_width, min(max_width, len(label) * char_width))
            return width
        
        # You can increase the font size of node labels and edge labels in Graphviz by setting the 'fontsize' attribute.
        dot = Digraph(format='png')
        dot.attr(rankdir=graph_layout, size='8,10')

        # Add nodes with average weights as labels and flexible width
        for i, name in enumerate(self.criteria_names):
            label = f"{name}\n({self.avg_weights[i]:.2f})"
            width = estimate_width(label)
            dot.node(
                name,
                label,
                shape='ellipse',
                style='filled',
                fillcolor=node_fillcolor,
                width=str(width),
                fixedsize='false',
                fontsize=node_fontsize,
                fontname=node_fontname,
            )

        # Add edges for probabilities > min_prob (excluding self-loops)
        for i, src in enumerate(self.criteria_names):
            for j, tgt in enumerate(self.criteria_names):
                if i != j and self.probs[i, j] > 0.5:
                    prob = self.probs[i, j]
                    if prob < 1:
                        dot.edge(src, tgt, label=f"{prob:.2f}")
                    elif prob == 1:
                        sorted_indices = np.argsort(self.avg_weights)
                        idx_src = sorted_indices.tolist().index(i)
                        idx_tgt = sorted_indices.tolist().index(j)
                        if abs(idx_src - idx_tgt) == 1:
                            dot.edge(
                                src, tgt,
                                label=f"{prob:.2f}",
                                font_size=edge_fontsize,
                                fontname=edge_fontname
                            )

        display(dot)

    def generate_ai_report_prompt(self):
        """
        Generates a prompt for an AI model to write a report on the importance of criteria
        based on their mean aggregated weights and the credal ranking probability matrix.

        Returns:
            str: A prompt string for an AI model.
        """
        criteria_names = self.criteria_names
        aggregated_weights = self.avg_weights
        credal_probs = self.probs

        # Prepare criteria and weights table
        table = "Criteria and their mean aggregated weights:\n"
        for name, weight in zip(criteria_names, aggregated_weights):
            table += f"- {name}: {weight:.4f}\n"

        # Prepare credal ranking summary
        ranking_summary = "Credal ranking probabilities (P[i > j]):\n"
        ranking_summary += (
            "Rows: more important; Columns: less important\n"
            "Each entry [i, j] shows the probability that criterion i is more important than criterion j, "
            "based on the uncertainty in the estimated weights.\n"
            "A value close to 1 means high confidence that i is more important than j; "
            "a value close to 0.5 means the ranking between i and j is uncertain.\n"
        )
        header = "      " + "  ".join([f"{n:>5}" for n in criteria_names]) + "\n"
        ranking_summary += header
        for i, name_i in enumerate(criteria_names):
            row = f"{name_i:>5} "
            for j in range(len(criteria_names)):
                row += f"{credal_probs[i, j]:5.2f} "
            ranking_summary += row + "\n"

        # Compose the prompt
        prompt = (
            "You are an expert in multi-criteria decision analysis. "
            "Given the following criteria, their mean aggregated weights, and the credal ranking probability matrix, "
            "write a concise report that:\n"
            "- Explains which criteria are most and least important.\n"
            "- Discusses the extent to which one criterion is more important than another, using the credal probabilities.\n"
            "- Highlights any cases where the ranking is uncertain or close.\n\n"
            "Background:\n"
            "The credal ranking probability matrix quantifies, for each pair of criteria, the probability that one is more important than the other, "
            "based on uncertainty in the estimated weights. Higher probabilities indicate greater confidence in the ranking.\n\n"
            f"{table}\n"
            f"{ranking_summary}\n"
            "Please provide a clear, structured summary suitable for decision makers."
        )
        return prompt