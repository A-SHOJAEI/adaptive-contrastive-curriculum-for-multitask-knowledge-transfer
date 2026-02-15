"""Results analysis and visualization utilities."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


class ResultsAnalyzer:
    """Analyze and visualize multitask learning results."""

    def __init__(self, results_dir: Optional[Union[str, Path]] = None):
        """Initialize results analyzer.

        Args:
            results_dir: Directory to save visualization outputs. If None,
                uses './results' relative to current directory.
        """
        self.results_dir = Path(results_dir) if results_dir else Path("./results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set default style
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def load_results_from_json(self, filepath: Union[str, Path]) -> Dict:
        """Load results from a JSON file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            Dictionary containing the loaded results.
        """
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            results = json.load(f)
        return results

    def load_results_from_csv(
        self, filepath: Union[str, Path], index_col: Optional[int] = 0
    ) -> pd.DataFrame:
        """Load results from a CSV file.

        Args:
            filepath: Path to the CSV file.
            index_col: Column to use as row labels.

        Returns:
            DataFrame containing the loaded results.
        """
        filepath = Path(filepath)
        df = pd.read_csv(filepath, index_col=index_col)
        return df

    def save_results_to_json(
        self, results: Dict, filepath: Union[str, Path], indent: int = 2
    ) -> None:
        """Save results to a JSON file.

        Args:
            results: Dictionary of results to save.
            filepath: Path to save the JSON file.
            indent: JSON indentation level.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        results_serializable = self._make_json_serializable(results)

        with open(filepath, "w") as f:
            json.dump(results_serializable, f, indent=indent)

    def save_results_to_csv(
        self, results: pd.DataFrame, filepath: Union[str, Path]
    ) -> None:
        """Save results DataFrame to a CSV file.

        Args:
            results: DataFrame to save.
            filepath: Path to save the CSV file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(filepath)

    def save_metrics(
        self, metrics: Dict, filename: str = 'metrics.json'
    ) -> None:
        """Save metrics dictionary to JSON file.

        Args:
            metrics: Dictionary of metrics to save.
            filename: Name of the output file.
        """
        filepath = self.results_dir / filename
        self.save_results_to_json(metrics, filepath)

    def create_summary_table(
        self, metrics: Dict, save_csv: bool = False
    ) -> pd.DataFrame:
        """Create a summary table from metrics.

        Args:
            metrics: Dictionary of metrics.
            save_csv: Whether to save as CSV.

        Returns:
            DataFrame containing summary statistics.
        """
        # Extract main metrics
        summary_data = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                summary_data[key] = [value]

        df = pd.DataFrame(summary_data).T
        df.columns = ['Value']
        df.index.name = 'Metric'

        if save_csv:
            self.save_results_to_csv(df, self.results_dir / 'summary_table.csv')

        return df

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        task_name: str = "Task",
        normalize: bool = True,
        figsize: tuple = (10, 8),
        cmap: str = "Blues",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot confusion matrix heatmap.

        Args:
            confusion_matrix: Confusion matrix to plot.
            class_names: Names of classes for axis labels.
            task_name: Name of the task for the title.
            normalize: Whether to normalize the confusion matrix.
            figsize: Figure size (width, height).
            cmap: Colormap to use.
            save_path: Path to save the plot. If None, saves to results_dir.

        Returns:
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Normalize if requested
        cm = confusion_matrix.copy()
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Proportion" if normalize else "Count"},
            ax=ax,
        )

        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(
            f"Confusion Matrix - {task_name}", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.results_dir / f"confusion_matrix_{task_name.lower().replace(' ', '_')}.png"
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_task_performance(
        self,
        per_task_metrics: Dict[str, Dict[str, float]],
        metrics_to_plot: Optional[List[str]] = None,
        figsize: tuple = (12, 6),
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot performance metrics across tasks.

        Args:
            per_task_metrics: Dictionary mapping task names to their metrics.
            metrics_to_plot: List of metric names to plot. If None, plots all.
            figsize: Figure size (width, height).
            save_path: Path to save the plot. If None, saves to results_dir.

        Returns:
            Matplotlib figure object.
        """
        # Prepare data
        df = pd.DataFrame(per_task_metrics).T

        if metrics_to_plot is None:
            metrics_to_plot = list(df.columns)

        # Create subplots
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            tasks = df.index.tolist()
            values = df[metric].values

            # Create bar plot
            bars = ax.bar(range(len(tasks)), values, alpha=0.7, edgecolor="black")

            # Color bars by performance
            colors = plt.cm.RdYlGn(values)
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            ax.set_xlabel("Task", fontsize=10)
            ax.set_ylabel(metric.capitalize(), fontsize=10)
            ax.set_title(f"{metric.capitalize()} by Task", fontsize=11, fontweight="bold")
            ax.set_xticks(range(len(tasks)))
            ax.set_xticklabels(tasks, rotation=45, ha="right")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.results_dir / "task_performance.png"
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_training_curves(
        self,
        train_history: Union[Dict[str, List[float]], List[Dict[str, float]]],
        val_history: Optional[List[Dict[str, float]]] = None,
        metrics: Optional[List[str]] = None,
        figsize: tuple = (12, 6),
        save_path: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """Plot training curves over epochs.

        Args:
            train_history: Training history (dict of lists or list of dicts).
            val_history: Validation history (list of dicts). Optional.
            metrics: List of metrics to plot. If None, plots loss and accuracy.
            figsize: Figure size (width, height).
            save_path: Path to save the plot. If None, saves to results_dir.
            filename: Filename to save. Overrides save_path if provided.

        Returns:
            Matplotlib figure object.
        """
        # Convert list of dicts to dict of lists if needed
        if isinstance(train_history, list):
            training_dict = {}
            for key in train_history[0].keys():
                training_dict[f'train_{key}'] = [epoch[key] for epoch in train_history]
        else:
            training_dict = train_history

        if val_history is not None:
            val_dict = {}
            for key in val_history[0].keys():
                val_dict[f'val_{key}'] = [epoch[key] for epoch in val_history]
            training_dict.update(val_dict)

        if metrics is None:
            # Default to plotting loss and accuracy
            metrics = [k for k in training_dict.keys() if 'loss' in k or 'accuracy' in k]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot loss
        loss_ax = axes[0]
        for metric in metrics:
            if 'loss' in metric and metric in training_dict:
                values = training_dict[metric]
                label = metric.replace('_', ' ').title()
                loss_ax.plot(values, label=label, marker="o", markersize=4, linewidth=2)

        loss_ax.set_xlabel("Epoch", fontsize=12)
        loss_ax.set_ylabel("Loss", fontsize=12)
        loss_ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        loss_ax.legend(loc="best", fontsize=10)
        loss_ax.grid(True, alpha=0.3)

        # Plot accuracy
        acc_ax = axes[1]
        for metric in metrics:
            if 'accuracy' in metric and metric in training_dict:
                values = training_dict[metric]
                label = metric.replace('_', ' ').title()
                acc_ax.plot(values, label=label, marker="o", markersize=4, linewidth=2)

        acc_ax.set_xlabel("Epoch", fontsize=12)
        acc_ax.set_ylabel("Accuracy", fontsize=12)
        acc_ax.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
        acc_ax.legend(loc="best", fontsize=10)
        acc_ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        if filename is not None:
            save_path = self.results_dir / filename
        elif save_path is None:
            save_path = self.results_dir / "training_curves.png"
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_cross_domain_transfer(
        self,
        transfer_scores: Dict[str, float],
        figsize: tuple = (12, 8),
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot cross-domain transfer scores as a heatmap.

        Args:
            transfer_scores: Dictionary mapping "source_to_target" to transfer scores.
            figsize: Figure size (width, height).
            save_path: Path to save the plot. If None, saves to results_dir.

        Returns:
            Matplotlib figure object.
        """
        # Parse transfer scores into matrix
        tasks = set()
        for key in transfer_scores.keys():
            source, target = key.split("_to_")
            tasks.add(source)
            tasks.add(target)

        tasks = sorted(list(tasks))
        n_tasks = len(tasks)

        # Create transfer matrix
        transfer_matrix = np.zeros((n_tasks, n_tasks))
        for i, source in enumerate(tasks):
            for j, target in enumerate(tasks):
                if i != j:
                    key = f"{source}_to_{target}"
                    if key in transfer_scores:
                        transfer_matrix[i, j] = transfer_scores[key]

        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            transfer_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0,
            xticklabels=tasks,
            yticklabels=tasks,
            cbar_kws={"label": "Transfer Score"},
            ax=ax,
        )

        ax.set_xlabel("Target Task", fontsize=12)
        ax.set_ylabel("Source Task", fontsize=12)
        ax.set_title("Cross-Domain Transfer Scores", fontsize=14, fontweight="bold")

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.results_dir / "cross_domain_transfer.png"
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_task_confusion_reduction(
        self,
        confusion_reduction: Dict[tuple, float],
        figsize: tuple = (12, 8),
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot task confusion reduction scores as a heatmap.

        Args:
            confusion_reduction: Dictionary mapping (task1, task2) tuples to scores.
            figsize: Figure size (width, height).
            save_path: Path to save the plot. If None, saves to results_dir.

        Returns:
            Matplotlib figure object.
        """
        # Get unique tasks
        tasks = set()
        for task1, task2 in confusion_reduction.keys():
            tasks.add(task1)
            tasks.add(task2)

        tasks = sorted(list(tasks))
        n_tasks = len(tasks)

        # Create confusion matrix
        confusion_matrix = np.zeros((n_tasks, n_tasks))
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):
                if i < j:
                    key = (task1, task2)
                    if key in confusion_reduction:
                        score = confusion_reduction[key]
                        confusion_matrix[i, j] = score
                        confusion_matrix[j, i] = score  # Symmetric

        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            xticklabels=tasks,
            yticklabels=tasks,
            cbar_kws={"label": "Confusion Reduction Score"},
            ax=ax,
        )

        ax.set_xlabel("Task", fontsize=12)
        ax.set_ylabel("Task", fontsize=12)
        ax.set_title(
            "Task Confusion Reduction (Higher = Less Confusion)",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.results_dir / "task_confusion_reduction.png"
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def generate_summary_report(
        self,
        results: Dict,
        save_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate a text summary report of results.

        Args:
            results: Dictionary containing all evaluation results.
            save_path: Path to save the report. If None, saves to results_dir.

        Returns:
            Summary report as a string.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("MULTITASK LEARNING EVALUATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Overall metrics
        if "average_accuracy" in results:
            lines.append("OVERALL PERFORMANCE")
            lines.append("-" * 80)
            lines.append(f"Average Accuracy:  {results['average_accuracy']:.4f}")
            if "average_precision" in results:
                lines.append(f"Average Precision: {results['average_precision']:.4f}")
            if "average_recall" in results:
                lines.append(f"Average Recall:    {results['average_recall']:.4f}")
            if "average_f1" in results:
                lines.append(f"Average F1 Score:  {results['average_f1']:.4f}")
            lines.append("")

        # Per-task metrics
        if "per_task_metrics" in results:
            lines.append("PER-TASK PERFORMANCE")
            lines.append("-" * 80)
            for task, metrics in results["per_task_metrics"].items():
                lines.append(f"\n{task}:")
                for metric, value in metrics.items():
                    lines.append(f"  {metric.capitalize():12s}: {value:.4f}")
            lines.append("")

        # Task confusion
        if "task_confusion_reduction" in results:
            lines.append("TASK CONFUSION REDUCTION")
            lines.append("-" * 80)
            for task_pair, score in results["task_confusion_reduction"].items():
                if isinstance(task_pair, tuple):
                    task1, task2 = task_pair
                    lines.append(f"{task1} <-> {task2}: {score:.4f}")
            lines.append("")

        # Cross-domain transfer
        if "cross_domain_transfer" in results:
            lines.append("CROSS-DOMAIN TRANSFER SCORES")
            lines.append("-" * 80)
            for transfer, score in results["cross_domain_transfer"].items():
                lines.append(f"{transfer}: {score:+.4f}")
            lines.append("")

        lines.append("=" * 80)

        report = "\n".join(lines)

        # Save report
        if save_path is None:
            save_path = self.results_dir / "summary_report.txt"
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report)

        return report

    def perform_statistical_analysis(
        self,
        results_list: List[Dict[str, Dict[str, float]]],
        metric: str = "accuracy",
        confidence_level: float = 0.95,
    ) -> Dict[str, Union[float, tuple]]:
        """Perform statistical analysis on multiple experimental runs.

        Args:
            results_list: List of per-task metrics from multiple runs.
            metric: Metric to analyze (e.g., 'accuracy', 'f1').
            confidence_level: Confidence level for intervals.

        Returns:
            Dictionary containing statistical measures.
        """
        # Collect metric values across runs
        task_values = {}
        for results in results_list:
            for task, metrics in results.items():
                if task not in task_values:
                    task_values[task] = []
                if metric in metrics:
                    task_values[task].append(metrics[metric])

        # Compute statistics
        statistics = {}
        for task, values in task_values.items():
            values_array = np.array(values)

            statistics[task] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "median": float(np.median(values_array)),
            }

            # Confidence interval
            if len(values_array) > 1:
                confidence_interval = stats.t.interval(
                    confidence_level,
                    len(values_array) - 1,
                    loc=np.mean(values_array),
                    scale=stats.sem(values_array),
                )
                statistics[task]["confidence_interval"] = tuple(
                    float(x) for x in confidence_interval
                )

        return statistics

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, Dict[str, float]]],
        metric: str = "accuracy",
        figsize: tuple = (12, 6),
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Compare performance of different models across tasks.

        Args:
            model_results: Dictionary mapping model names to their per-task metrics.
            metric: Metric to compare.
            figsize: Figure size (width, height).
            save_path: Path to save the plot. If None, saves to results_dir.

        Returns:
            Matplotlib figure object.
        """
        # Prepare data
        tasks = set()
        for model_metrics in model_results.values():
            tasks.update(model_metrics.keys())
        tasks = sorted(list(tasks))

        # Extract values
        model_names = list(model_results.keys())
        data = {model: [] for model in model_names}

        for task in tasks:
            for model in model_names:
                if task in model_results[model] and metric in model_results[model][task]:
                    data[model].append(model_results[model][task][metric])
                else:
                    data[model].append(0.0)

        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(tasks))
        width = 0.8 / len(model_names)

        for idx, (model, values) in enumerate(data.items()):
            offset = (idx - len(model_names) / 2) * width + width / 2
            ax.bar(x + offset, values, width, label=model, alpha=0.8)

        ax.set_xlabel("Task", fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f"Model Comparison - {metric.capitalize()}", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.legend(loc="best")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.results_dir / f"model_comparison_{metric}.png"
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization.

        Args:
            obj: Object to convert.

        Returns:
            JSON-serializable version of the object.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    task_name: str = "Task",
    figsize: tuple = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot learning curve showing performance vs. training set size.

    Args:
        train_sizes: Array of training set sizes.
        train_scores: Array of training scores for each size.
        val_scores: Array of validation scores for each size.
        task_name: Name of the task for the title.
        figsize: Figure size (width, height).
        save_path: Path to save the plot.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot curves
    ax.plot(train_sizes, train_scores, "o-", label="Training Score", linewidth=2)
    ax.plot(train_sizes, val_scores, "o-", label="Validation Score", linewidth=2)

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Learning Curve - {task_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
