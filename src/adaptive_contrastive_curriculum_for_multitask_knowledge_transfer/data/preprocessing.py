"""Data preprocessing utilities."""

import logging
from typing import Any, Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MMLUPreprocessor:
    """Preprocessor for MMLU data."""

    def __init__(self, num_tasks: int):
        """Initialize preprocessor.

        Args:
            num_tasks: Total number of tasks
        """
        self.num_tasks = num_tasks
        self.task_stats: Dict[int, Dict[str, float]] = {}

    def compute_task_statistics(self, data: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        """Compute statistics for each task.

        Args:
            data: List of data samples

        Returns:
            Dictionary mapping task_id to statistics
        """
        task_counts: Dict[int, int] = {}
        task_correct: Dict[int, int] = {}

        for sample in data:
            task_id = sample['task_id']
            task_counts[task_id] = task_counts.get(task_id, 0) + 1

        # Compute statistics
        stats = {}
        for task_id in range(self.num_tasks):
            count = task_counts.get(task_id, 0)
            stats[task_id] = {
                'count': count,
                'proportion': count / len(data) if len(data) > 0 else 0.0,
            }

        self.task_stats = stats
        logger.info(f"Computed statistics for {len(stats)} tasks")
        return stats

    def balance_tasks(
        self, data: List[Dict[str, Any]], strategy: str = 'oversample'
    ) -> List[Dict[str, Any]]:
        """Balance task distribution in dataset.

        Args:
            data: List of data samples
            strategy: Balancing strategy ('oversample' or 'undersample')

        Returns:
            Balanced dataset
        """
        if strategy not in ['oversample', 'undersample']:
            raise ValueError(f"Unknown balancing strategy: {strategy}")

        # Group by task
        task_samples: Dict[int, List[Dict[str, Any]]] = {}
        for sample in data:
            task_id = sample['task_id']
            if task_id not in task_samples:
                task_samples[task_id] = []
            task_samples[task_id].append(sample)

        # Find target count
        counts = [len(samples) for samples in task_samples.values()]
        if strategy == 'oversample':
            target_count = max(counts)
        else:
            target_count = min(counts)

        # Resample each task
        balanced_data = []
        for task_id, samples in task_samples.items():
            if len(samples) < target_count:
                # Oversample
                indices = np.random.choice(len(samples), target_count, replace=True)
                balanced_data.extend([samples[i] for i in indices])
            elif len(samples) > target_count:
                # Undersample
                indices = np.random.choice(len(samples), target_count, replace=False)
                balanced_data.extend([samples[i] for i in indices])
            else:
                balanced_data.extend(samples)

        logger.info(f"Balanced dataset from {len(data)} to {len(balanced_data)} samples")
        return balanced_data

    def create_domain_mapping(self, task_to_category: Dict[str, str]) -> Dict[int, int]:
        """Create mapping from task ID to domain ID.

        Args:
            task_to_category: Mapping from task name to category

        Returns:
            Dictionary mapping task_id to domain_id
        """
        categories = list(set(task_to_category.values()))
        category_to_id = {cat: i for i, cat in enumerate(categories)}

        task_to_domain = {}
        for task_name, category in task_to_category.items():
            # This would need the actual task name to ID mapping
            # For now, return empty dict
            pass

        return {}
