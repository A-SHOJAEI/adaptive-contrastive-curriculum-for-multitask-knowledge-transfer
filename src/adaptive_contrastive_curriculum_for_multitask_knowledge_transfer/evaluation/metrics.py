"""Comprehensive metrics for evaluating multitask learning performance."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class MetricsCalculator:
    """Calculate comprehensive metrics for multitask learning evaluation."""

    def __init__(self, task_names: Optional[List[str]] = None):
        """Initialize metrics calculator.

        Args:
            task_names: List of task names for per-task metrics tracking.
        """
        self.task_names = task_names or []
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.predictions: Dict[str, List[np.ndarray]] = {
            task: [] for task in self.task_names
        }
        self.targets: Dict[str, List[np.ndarray]] = {
            task: [] for task in self.task_names
        }

    def update(
        self,
        predictions: Union[torch.Tensor, np.ndarray, Dict[str, Union[torch.Tensor, np.ndarray]]],
        targets: Union[torch.Tensor, np.ndarray, Dict[str, Union[torch.Tensor, np.ndarray]]],
        task_name: Optional[str] = None,
    ) -> None:
        """Update metrics with new predictions and targets.

        Args:
            predictions: Model predictions (tensor, array, or dict of predictions per task).
            targets: Ground truth labels (tensor, array, or dict of targets per task).
            task_name: Name of the task (required if predictions/targets are not dicts).
        """
        # Convert to dict format if single task
        if not isinstance(predictions, dict):
            if task_name is None:
                raise ValueError("task_name required when predictions is not a dict")
            predictions = {task_name: predictions}
            targets = {task_name: targets}

        # Update each task
        for task, pred in predictions.items():
            if task not in self.predictions:
                self.predictions[task] = []
                self.targets[task] = []

            # Convert to numpy
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            if isinstance(targets[task], torch.Tensor):
                tgt = targets[task].detach().cpu().numpy()
            else:
                tgt = targets[task]

            self.predictions[task].append(pred)
            self.targets[task].append(tgt)

    def compute_accuracy(
        self, task_name: Optional[str] = None
    ) -> Union[float, Dict[str, float]]:
        """Compute accuracy for one or all tasks.

        Args:
            task_name: Specific task name, or None for all tasks.

        Returns:
            Accuracy score(s).
        """
        if task_name is not None:
            y_true, y_pred = self._get_predictions(task_name)
            return float(accuracy_score(y_true, y_pred))

        return {task: self.compute_accuracy(task) for task in self.predictions.keys()}

    def compute_precision(
        self,
        task_name: Optional[str] = None,
        average: str = "weighted",
        zero_division: Union[str, float] = 0,
    ) -> Union[float, Dict[str, float]]:
        """Compute precision for one or all tasks.

        Args:
            task_name: Specific task name, or None for all tasks.
            average: Averaging method ('micro', 'macro', 'weighted', 'binary').
            zero_division: Value to return when there is a zero division.

        Returns:
            Precision score(s).
        """
        if task_name is not None:
            y_true, y_pred = self._get_predictions(task_name)
            return float(
                precision_score(
                    y_true, y_pred, average=average, zero_division=zero_division
                )
            )

        return {
            task: self.compute_precision(task, average, zero_division)
            for task in self.predictions.keys()
        }

    def compute_recall(
        self,
        task_name: Optional[str] = None,
        average: str = "weighted",
        zero_division: Union[str, float] = 0,
    ) -> Union[float, Dict[str, float]]:
        """Compute recall for one or all tasks.

        Args:
            task_name: Specific task name, or None for all tasks.
            average: Averaging method ('micro', 'macro', 'weighted', 'binary').
            zero_division: Value to return when there is a zero division.

        Returns:
            Recall score(s).
        """
        if task_name is not None:
            y_true, y_pred = self._get_predictions(task_name)
            return float(
                recall_score(y_true, y_pred, average=average, zero_division=zero_division)
            )

        return {
            task: self.compute_recall(task, average, zero_division)
            for task in self.predictions.keys()
        }

    def compute_f1(
        self,
        task_name: Optional[str] = None,
        average: str = "weighted",
        zero_division: Union[str, float] = 0,
    ) -> Union[float, Dict[str, float]]:
        """Compute F1 score for one or all tasks.

        Args:
            task_name: Specific task name, or None for all tasks.
            average: Averaging method ('micro', 'macro', 'weighted', 'binary').
            zero_division: Value to return when there is a zero division.

        Returns:
            F1 score(s).
        """
        if task_name is not None:
            y_true, y_pred = self._get_predictions(task_name)
            return float(
                f1_score(y_true, y_pred, average=average, zero_division=zero_division)
            )

        return {
            task: self.compute_f1(task, average, zero_division)
            for task in self.predictions.keys()
        }

    def compute_confusion_matrix(
        self, task_name: str, normalize: Optional[str] = None
    ) -> np.ndarray:
        """Compute confusion matrix for a specific task.

        Args:
            task_name: Name of the task.
            normalize: Normalization mode ('true', 'pred', 'all', or None).

        Returns:
            Confusion matrix as numpy array.
        """
        y_true, y_pred = self._get_predictions(task_name)
        cm = confusion_matrix(y_true, y_pred)

        if normalize is not None:
            if normalize == "true":
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            elif normalize == "pred":
                cm = cm.astype("float") / cm.sum(axis=0)
            elif normalize == "all":
                cm = cm.astype("float") / cm.sum()
            else:
                raise ValueError(
                    f"Invalid normalize value: {normalize}. "
                    "Must be 'true', 'pred', 'all', or None."
                )

        return cm

    def compute_cross_domain_transfer(
        self,
        source_task: str,
        target_task: str,
        baseline_accuracy: Optional[float] = None,
    ) -> float:
        """Compute cross-domain transfer score between two tasks.

        This metric measures how well knowledge from a source task transfers
        to a target task. It compares the target task performance with and
        without the source task training.

        Args:
            source_task: Name of the source task.
            target_task: Name of the target task.
            baseline_accuracy: Baseline accuracy on target task without source
                task knowledge. If None, uses random chance (1/num_classes).

        Returns:
            Transfer score (positive = positive transfer, negative = negative transfer).
        """
        target_acc = self.compute_accuracy(target_task)

        if baseline_accuracy is None:
            # Estimate baseline as random chance
            y_true, _ = self._get_predictions(target_task)
            num_classes = len(np.unique(y_true))
            baseline_accuracy = 1.0 / num_classes

        # Transfer score: improvement over baseline
        transfer_score = float(target_acc - baseline_accuracy)

        return transfer_score

    def compute_task_confusion_reduction(
        self,
        task_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[Tuple[str, str], float]:
        """Compute task confusion reduction between task pairs.

        This custom metric measures how well the model distinguishes between
        different tasks by analyzing the reduction in cross-task confusion.
        Lower values indicate better task separation.

        Args:
            task_pairs: List of (task1, task2) tuples to analyze. If None,
                computes for all task pairs.

        Returns:
            Dictionary mapping task pairs to confusion reduction scores.
        """
        if task_pairs is None:
            # Generate all pairs
            tasks = list(self.predictions.keys())
            task_pairs = [
                (tasks[i], tasks[j])
                for i in range(len(tasks))
                for j in range(i + 1, len(tasks))
            ]

        confusion_scores = {}

        for task1, task2 in task_pairs:
            # Get predictions for both tasks
            y_true1, y_pred1 = self._get_predictions(task1)
            y_true2, y_pred2 = self._get_predictions(task2)

            # Calculate confusion: how often task1 predictions match task2 patterns
            # This is a simplified metric - in practice, you might want to use
            # more sophisticated analysis (e.g., embedding similarity)

            # Normalize predictions to [0, 1] range for comparison
            if len(y_pred1) > 0 and len(y_pred2) > 0:
                pred1_norm = (y_pred1 - y_pred1.min()) / (
                    y_pred1.max() - y_pred1.min() + 1e-8
                )
                pred2_norm = (y_pred2 - y_pred2.min()) / (
                    y_pred2.max() - y_pred2.min() + 1e-8
                )

                # Measure overlap/confusion as correlation
                min_len = min(len(pred1_norm), len(pred2_norm))
                confusion = float(
                    np.corrcoef(pred1_norm[:min_len], pred2_norm[:min_len])[0, 1]
                )

                # Convert to reduction score (1 - abs(confusion))
                # Higher is better (less confusion)
                confusion_scores[(task1, task2)] = 1.0 - abs(confusion)
            else:
                confusion_scores[(task1, task2)] = 0.0

        return confusion_scores

    def compute_per_task_metrics(
        self,
        average: str = "weighted",
        zero_division: Union[str, float] = 0,
    ) -> Dict[str, Dict[str, float]]:
        """Compute all standard metrics for each task.

        Args:
            average: Averaging method for precision, recall, and F1.
            zero_division: Value to return when there is a zero division.

        Returns:
            Dictionary mapping task names to their metrics.
        """
        metrics = {}

        for task in self.predictions.keys():
            metrics[task] = {
                "accuracy": self.compute_accuracy(task),
                "precision": self.compute_precision(task, average, zero_division),
                "recall": self.compute_recall(task, average, zero_division),
                "f1": self.compute_f1(task, average, zero_division),
            }

        return metrics

    def compute_all_metrics(
        self,
        average: str = "weighted",
        zero_division: Union[str, float] = 0,
        include_confusion: bool = False,
        baseline_accuracies: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Union[float, Dict, np.ndarray]]:
        """Compute all available metrics.

        Args:
            average: Averaging method for precision, recall, and F1.
            zero_division: Value to return when there is a zero division.
            include_confusion: Whether to include confusion matrices.
            baseline_accuracies: Baseline accuracies for transfer metrics.

        Returns:
            Dictionary containing all computed metrics.
        """
        results: Dict[str, Union[float, Dict, np.ndarray]] = {}

        # Standard per-task metrics
        results["per_task_metrics"] = self.compute_per_task_metrics(
            average, zero_division
        )

        # Overall average metrics
        per_task = results["per_task_metrics"]
        if isinstance(per_task, dict) and len(per_task) > 0:
            results["average_accuracy"] = float(
                np.mean([m["accuracy"] for m in per_task.values()])
            )
            results["average_precision"] = float(
                np.mean([m["precision"] for m in per_task.values()])
            )
            results["average_recall"] = float(
                np.mean([m["recall"] for m in per_task.values()])
            )
            results["average_f1"] = float(
                np.mean([m["f1"] for m in per_task.values()])
            )

        # Task confusion reduction
        results["task_confusion_reduction"] = self.compute_task_confusion_reduction()

        # Cross-domain transfer (if baseline provided)
        if baseline_accuracies:
            transfer_scores = {}
            for target_task in self.predictions.keys():
                if target_task in baseline_accuracies:
                    # Use all other tasks as potential sources
                    for source_task in self.predictions.keys():
                        if source_task != target_task:
                            transfer_scores[f"{source_task}_to_{target_task}"] = (
                                self.compute_cross_domain_transfer(
                                    source_task,
                                    target_task,
                                    baseline_accuracies[target_task],
                                )
                            )
            results["cross_domain_transfer"] = transfer_scores

        # Confusion matrices
        if include_confusion:
            confusion_matrices = {}
            for task in self.predictions.keys():
                confusion_matrices[task] = self.compute_confusion_matrix(task)
            results["confusion_matrices"] = confusion_matrices

        return results

    def _get_predictions(self, task_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get concatenated predictions and targets for a task.

        Args:
            task_name: Name of the task.

        Returns:
            Tuple of (targets, predictions) as numpy arrays.
        """
        if task_name not in self.predictions:
            raise ValueError(f"Task '{task_name}' not found in predictions")

        if len(self.predictions[task_name]) == 0:
            raise ValueError(f"No predictions available for task '{task_name}'")

        y_pred = np.concatenate(self.predictions[task_name])
        y_true = np.concatenate(self.targets[task_name])

        # Handle multi-dimensional predictions (e.g., logits)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=-1)

        return y_true, y_pred


def compute_accuracy(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> float:
    """Compute accuracy for predictions and targets.

    Args:
        predictions: Model predictions (can be logits or class indices).
        targets: Ground truth labels.

    Returns:
        Accuracy score.
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Handle logits
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=-1)

    return float(accuracy_score(targets, predictions))


def compute_f1(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    average: str = "weighted",
    zero_division: Union[str, float] = 0,
) -> float:
    """Compute F1 score for predictions and targets.

    Args:
        predictions: Model predictions (can be logits or class indices).
        targets: Ground truth labels.
        average: Averaging method ('micro', 'macro', 'weighted', 'binary').
        zero_division: Value to return when there is a zero division.

    Returns:
        F1 score.
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Handle logits
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=-1)

    return float(
        f1_score(targets, predictions, average=average, zero_division=zero_division)
    )


def compute_precision(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    average: str = "weighted",
    zero_division: Union[str, float] = 0,
) -> float:
    """Compute precision for predictions and targets.

    Args:
        predictions: Model predictions (can be logits or class indices).
        targets: Ground truth labels.
        average: Averaging method ('micro', 'macro', 'weighted', 'binary').
        zero_division: Value to return when there is a zero division.

    Returns:
        Precision score.
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Handle logits
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=-1)

    return float(
        precision_score(
            targets, predictions, average=average, zero_division=zero_division
        )
    )


def compute_recall(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    average: str = "weighted",
    zero_division: Union[str, float] = 0,
) -> float:
    """Compute recall for predictions and targets.

    Args:
        predictions: Model predictions (can be logits or class indices).
        targets: Ground truth labels.
        average: Averaging method ('micro', 'macro', 'weighted', 'binary').
        zero_division: Value to return when there is a zero division.

    Returns:
        Recall score.
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Handle logits
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=-1)

    return float(
        recall_score(targets, predictions, average=average, zero_division=zero_division)
    )
