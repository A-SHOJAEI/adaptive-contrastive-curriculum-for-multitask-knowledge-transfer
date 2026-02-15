"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.models.model import (
    AdaptiveContrastiveMultiTaskModel,
)
from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.models.components import (
    UncertaintyWeightedContrastiveLoss,
    TaskConfusionMatrix,
    CurriculumScheduler,
)


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")  # Use CPU for testing


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "seed": 42,
        "model": {
            "base_model": "bert-base-uncased",
            "num_tasks": 10,
            "hidden_dim": 768,
            "projection_dim": 256,
            "dropout": 0.1,
            "freeze_base": False,
        },
        "data": {
            "max_samples_per_task": 50,
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "max_seq_length": 128,
            "num_workers": 0,
        },
        "curriculum": {
            "enabled": True,
            "warmup_epochs": 1,
            "strategy": "uncertainty_gradient",
            "temperature": 2.0,
            "update_frequency": 10,
            "min_task_weight": 0.1,
            "max_task_weight": 3.0,
        },
        "contrastive": {
            "enabled": True,
            "temperature": 0.07,
            "lambda_contrastive": 0.3,
            "lambda_classification": 0.7,
        },
        "training": {
            "num_epochs": 2,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.00002,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
            "mixed_precision": False,
            "scheduler": {"type": "cosine", "num_cycles": 0.5},
            "early_stopping": {
                "enabled": True,
                "patience": 2,
                "min_delta": 0.001,
                "monitor": "val_accuracy",
            },
            "checkpoint": {
                "save_best": True,
                "save_last": True,
                "monitor": "val_accuracy",
                "mode": "max",
            },
        },
    }


@pytest.fixture
def tokenizer():
    """Load tokenizer for testing."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def sample_data():
    """Sample MMLU-like data for testing."""
    return [
        {
            "question": "What is 2 + 2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
            "subject": "mathematics",
            "task_id": 0,
        },
        {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "answer": 1,
            "subject": "geography",
            "task_id": 1,
        },
        {
            "question": "What is H2O?",
            "choices": ["Oxygen", "Water", "Hydrogen", "Carbon"],
            "answer": 1,
            "subject": "chemistry",
            "task_id": 2,
        },
    ]


@pytest.fixture
def sample_batch(device):
    """Sample batch for testing."""
    batch_size = 4
    seq_len = 128
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)).to(device),
        "attention_mask": torch.ones(batch_size, seq_len).to(device),
        "labels": torch.randint(0, 4, (batch_size,)).to(device),
        "task_ids": torch.randint(0, 10, (batch_size,)).to(device),
    }


@pytest.fixture
def simple_model(sample_config, device):
    """Create a simple model for testing."""
    model = AdaptiveContrastiveMultiTaskModel(
        base_model=sample_config["model"]["base_model"],
        num_tasks=sample_config["model"]["num_tasks"],
        hidden_dim=sample_config["model"]["hidden_dim"],
        projection_dim=sample_config["model"]["projection_dim"],
        num_classes=4,
        dropout=sample_config["model"]["dropout"],
        freeze_base=True,  # Freeze for faster testing
    )
    return model.to(device)


@pytest.fixture
def contrastive_loss(sample_config):
    """Create contrastive loss for testing."""
    return UncertaintyWeightedContrastiveLoss(
        temperature=sample_config["contrastive"]["temperature"],
        lambda_contrastive=sample_config["contrastive"]["lambda_contrastive"],
        lambda_classification=sample_config["contrastive"]["lambda_classification"],
        num_tasks=sample_config["model"]["num_tasks"],
    )


@pytest.fixture
def task_confusion_matrix(sample_config):
    """Create task confusion matrix for testing."""
    return TaskConfusionMatrix(
        num_tasks=sample_config["model"]["num_tasks"], smoothing=0.1
    )


@pytest.fixture
def curriculum_scheduler(sample_config):
    """Create curriculum scheduler for testing."""
    return CurriculumScheduler(
        num_tasks=sample_config["model"]["num_tasks"],
        strategy=sample_config["curriculum"]["strategy"],
        temperature=sample_config["curriculum"]["temperature"],
        warmup_steps=100,
        min_weight=sample_config["curriculum"]["min_task_weight"],
        max_weight=sample_config["curriculum"]["max_task_weight"],
    )


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
