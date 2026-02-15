"""Tests for training components."""

import pytest
import torch
from unittest.mock import Mock

from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.models.model import (
    AdaptiveContrastiveMultiTaskModel,
)
from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.training.trainer import (
    AdaptiveCurriculumTrainer,
)


def test_trainer_initialization() -> None:
    """Test trainer initialization."""
    model = AdaptiveContrastiveMultiTaskModel(
        base_model='prajjwal1/bert-tiny',
        num_tasks=10,
        hidden_dim=128,
        projection_dim=64,
        num_classes=4,
    )

    config = {
        'model': {'num_tasks': 10},
        'training': {
            'num_epochs': 5,
            'learning_rate': 0.0001,
            'weight_decay': 0.01,
            'batch_size': 16,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'mixed_precision': False,
            'scheduler': {'type': 'cosine'},
            'early_stopping': {
                'enabled': True,
                'patience': 3,
                'min_delta': 0.001,
            },
            'checkpoint': {'save_best': True, 'save_last': True},
        },
        'curriculum': {
            'enabled': True,
            'strategy': 'uncertainty_gradient',
            'temperature': 2.0,
            'warmup_epochs': 2,
            'update_frequency': 10,
        },
        'contrastive': {
            'enabled': True,
            'temperature': 0.07,
            'lambda_contrastive': 0.3,
            'lambda_classification': 0.7,
        },
        'logging': {'use_mlflow': False, 'log_interval': 50},
    }

    device = torch.device('cpu')

    # Create mock dataloaders
    mock_train_loader = Mock()
    mock_train_loader.__len__ = Mock(return_value=10)
    mock_val_loader = Mock()

    trainer = AdaptiveCurriculumTrainer(
        model=model,
        train_loader=mock_train_loader,
        val_loader=mock_val_loader,
        config=config,
        device=device,
        output_dir='/tmp/test_trainer'
    )

    assert trainer.num_epochs == 5
    assert trainer.curriculum_enabled is True


def test_trainer_save_checkpoint(tmp_path) -> None:
    """Test checkpoint saving."""
    model = AdaptiveContrastiveMultiTaskModel(
        base_model='prajjwal1/bert-tiny',
        num_tasks=10,
        hidden_dim=128,
        projection_dim=64,
        num_classes=4,
    )

    config = {
        'model': {'num_tasks': 10},
        'training': {
            'num_epochs': 1,
            'learning_rate': 0.0001,
            'weight_decay': 0.01,
            'batch_size': 16,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'mixed_precision': False,
            'scheduler': {'type': 'constant'},
            'early_stopping': {'enabled': False},
            'checkpoint': {'save_best': True, 'save_last': True},
        },
        'curriculum': {'enabled': False},
        'contrastive': {
            'enabled': True,
            'temperature': 0.07,
            'lambda_contrastive': 0.3,
            'lambda_classification': 0.7,
        },
        'logging': {'use_mlflow': False, 'log_interval': 50},
    }

    device = torch.device('cpu')

    # Create mock dataloaders
    mock_train_loader = Mock()
    mock_train_loader.__len__ = Mock(return_value=10)
    mock_val_loader = Mock()

    trainer = AdaptiveCurriculumTrainer(
        model=model,
        train_loader=mock_train_loader,
        val_loader=mock_val_loader,
        config=config,
        device=device,
        output_dir=str(tmp_path)
    )

    # Save checkpoint
    val_metrics = {'loss': 0.5, 'accuracy': 0.75}
    trainer._save_checkpoint(val_metrics, is_best=True, is_last=False)

    # Check file exists
    checkpoint_path = tmp_path / 'best_model.pt'
    assert checkpoint_path.exists()

    # Load and verify
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert 'model_state_dict' in checkpoint
    assert 'optimizer_state_dict' in checkpoint
    assert 'val_metrics' in checkpoint


def test_trainer_load_checkpoint(tmp_path) -> None:
    """Test checkpoint loading."""
    model = AdaptiveContrastiveMultiTaskModel(
        base_model='prajjwal1/bert-tiny',
        num_tasks=10,
        hidden_dim=128,
        projection_dim=64,
        num_classes=4,
    )

    config = {
        'model': {'num_tasks': 10},
        'training': {
            'num_epochs': 1,
            'learning_rate': 0.0001,
            'weight_decay': 0.01,
            'batch_size': 16,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'mixed_precision': False,
            'scheduler': {'type': 'constant'},
            'early_stopping': {'enabled': False},
            'checkpoint': {'save_best': True, 'save_last': True},
        },
        'curriculum': {'enabled': False},
        'contrastive': {
            'enabled': True,
            'temperature': 0.07,
            'lambda_contrastive': 0.3,
            'lambda_classification': 0.7,
        },
        'logging': {'use_mlflow': False, 'log_interval': 50},
    }

    device = torch.device('cpu')

    # Create mock dataloaders
    mock_train_loader = Mock()
    mock_train_loader.__len__ = Mock(return_value=10)
    mock_val_loader = Mock()

    trainer = AdaptiveCurriculumTrainer(
        model=model,
        train_loader=mock_train_loader,
        val_loader=mock_val_loader,
        config=config,
        device=device,
        output_dir=str(tmp_path)
    )

    # Save checkpoint
    val_metrics = {'loss': 0.3, 'accuracy': 0.8}
    trainer.current_epoch = 5
    trainer._save_checkpoint(val_metrics, is_best=True, is_last=False)

    # Create new trainer and load
    new_model = AdaptiveContrastiveMultiTaskModel(
        base_model='prajjwal1/bert-tiny',
        num_tasks=10,
        hidden_dim=128,
        projection_dim=64,
        num_classes=4,
    )
    new_trainer = AdaptiveCurriculumTrainer(
        model=new_model,
        train_loader=mock_train_loader,
        val_loader=mock_val_loader,
        config=config,
        device=device,
        output_dir=str(tmp_path)
    )

    checkpoint_path = tmp_path / 'best_model.pt'
    new_trainer.load_checkpoint(str(checkpoint_path))

    assert new_trainer.current_epoch == 5
