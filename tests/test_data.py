"""Tests for data loading and preprocessing."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.data.preprocessing import (
    MMLUPreprocessor,
)


def test_preprocessor_initialization() -> None:
    """Test MMLUPreprocessor initialization."""
    preprocessor = MMLUPreprocessor(num_tasks=10)
    assert preprocessor.num_tasks == 10
    assert preprocessor.task_stats == {}


def test_compute_task_statistics() -> None:
    """Test task statistics computation."""
    preprocessor = MMLUPreprocessor(num_tasks=3)

    # Create sample data
    data = [
        {'task_id': 0, 'answer': 1},
        {'task_id': 0, 'answer': 2},
        {'task_id': 1, 'answer': 0},
        {'task_id': 2, 'answer': 3},
        {'task_id': 2, 'answer': 1},
    ]

    stats = preprocessor.compute_task_statistics(data)

    assert len(stats) == 3
    assert stats[0]['count'] == 2
    assert stats[1]['count'] == 1
    assert stats[2]['count'] == 2


def test_balance_tasks_oversample() -> None:
    """Test task balancing with oversampling."""
    preprocessor = MMLUPreprocessor(num_tasks=2)

    data = [
        {'task_id': 0, 'answer': 0},
        {'task_id': 0, 'answer': 1},
        {'task_id': 1, 'answer': 2},
    ]

    balanced = preprocessor.balance_tasks(data, strategy='oversample')

    # Should balance to max count (2)
    task_counts = {}
    for sample in balanced:
        tid = sample['task_id']
        task_counts[tid] = task_counts.get(tid, 0) + 1

    assert task_counts[0] == 2
    assert task_counts[1] == 2


def test_balance_tasks_undersample() -> None:
    """Test task balancing with undersampling."""
    preprocessor = MMLUPreprocessor(num_tasks=2)

    data = [
        {'task_id': 0, 'answer': 0},
        {'task_id': 0, 'answer': 1},
        {'task_id': 1, 'answer': 2},
    ]

    balanced = preprocessor.balance_tasks(data, strategy='undersample')

    # Should balance to min count (1)
    task_counts = {}
    for sample in balanced:
        tid = sample['task_id']
        task_counts[tid] = task_counts.get(tid, 0) + 1

    assert task_counts[0] == 1
    assert task_counts[1] == 1


def test_balance_tasks_invalid_strategy() -> None:
    """Test that invalid balancing strategy raises error."""
    preprocessor = MMLUPreprocessor(num_tasks=2)
    data = [{'task_id': 0, 'answer': 0}]

    with pytest.raises(ValueError):
        preprocessor.balance_tasks(data, strategy='invalid')
