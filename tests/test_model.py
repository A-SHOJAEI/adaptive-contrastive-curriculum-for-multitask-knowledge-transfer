"""Tests for model components."""

import pytest
import torch
import numpy as np

from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.models.model import (
    AdaptiveContrastiveMultiTaskModel,
)
from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.models.components import (
    UncertaintyWeightedContrastiveLoss,
    CurriculumScheduler,
    TaskConfusionMatrix,
)


def test_model_initialization() -> None:
    """Test model initialization."""
    model = AdaptiveContrastiveMultiTaskModel(
        base_model='prajjwal1/bert-tiny',  # Use tiny model for testing
        num_tasks=10,
        hidden_dim=128,
        projection_dim=64,
        num_classes=4,
    )

    assert model.num_tasks == 10
    assert model.num_classes == 4
    assert model.projection_dim == 64


def test_model_forward() -> None:
    """Test model forward pass."""
    model = AdaptiveContrastiveMultiTaskModel(
        base_model='prajjwal1/bert-tiny',
        num_tasks=10,
        hidden_dim=128,
        projection_dim=64,
        num_classes=4,
    )

    batch_size = 4
    seq_len = 32

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    task_ids = torch.randint(0, 10, (batch_size,))

    logits, embeddings = model(input_ids, attention_mask, task_ids)

    assert logits.shape == (batch_size, 4)
    assert embeddings.shape == (batch_size, 64)


def test_contrastive_loss() -> None:
    """Test uncertainty-weighted contrastive loss."""
    loss_fn = UncertaintyWeightedContrastiveLoss(
        temperature=0.07,
        lambda_contrastive=0.3,
        lambda_classification=0.7,
        num_tasks=10,
    )

    batch_size = 8
    num_classes = 4
    embedding_dim = 64

    logits = torch.randn(batch_size, num_classes)
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    task_ids = torch.randint(0, 10, (batch_size,))

    loss, loss_dict = loss_fn(logits, embeddings, labels, task_ids)

    assert loss.item() >= 0
    assert 'total' in loss_dict
    assert 'classification' in loss_dict
    assert 'contrastive' in loss_dict


def test_curriculum_scheduler() -> None:
    """Test curriculum scheduler."""
    scheduler = CurriculumScheduler(
        num_tasks=10,
        strategy='uncertainty_gradient',
        temperature=2.0,
        warmup_steps=100,
    )

    # Test warmup phase
    confusion_scores = np.random.rand(10)
    scheduler.update(confusion_scores)

    weights = scheduler.get_task_weights()
    assert weights.shape == (10,)
    assert np.all(weights == 1.0)  # Should be uniform during warmup

    # Test after warmup
    scheduler.step = 200
    scheduler.update(confusion_scores)

    weights = scheduler.get_task_weights()
    assert not np.all(weights == 1.0)  # Should be non-uniform after warmup


def test_task_confusion_matrix() -> None:
    """Test task confusion matrix."""
    confusion_matrix = TaskConfusionMatrix(num_tasks=5)

    task_ids = np.array([0, 0, 1, 1, 2])
    predictions = np.array([0, 1, 2, 1, 2])
    embeddings = np.random.randn(5, 64)

    confusion_matrix.update(task_ids, predictions, embeddings)

    scores = confusion_matrix.get_confusion_scores()
    assert scores.shape == (5,)
    assert np.all(scores >= 0)


def test_model_parameter_count() -> None:
    """Test model parameter counting."""
    model = AdaptiveContrastiveMultiTaskModel(
        base_model='prajjwal1/bert-tiny',
        num_tasks=10,
        hidden_dim=128,
        projection_dim=64,
        num_classes=4,
    )

    param_counts = model.get_num_parameters()

    assert 'total' in param_counts
    assert 'trainable' in param_counts
    assert param_counts['total'] > 0
    assert param_counts['trainable'] > 0
