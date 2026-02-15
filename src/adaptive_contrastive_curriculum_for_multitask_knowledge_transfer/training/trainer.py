"""Production-quality trainer for adaptive contrastive curriculum learning.

This module implements a comprehensive trainer with:
- Gradient accumulation and mixed precision training
- Learning rate scheduling (cosine/linear/constant)
- Early stopping with patience
- Checkpoint management (best and last)
- Integration with curriculum learning and contrastive loss
- MLflow and tensorboard logging
- Progress tracking with tqdm
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ConstantLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from ..models.model import AdaptiveContrastiveMultiTaskModel
from ..models.components import (
    UncertaintyWeightedContrastiveLoss,
    CurriculumScheduler,
    TaskConfusionMatrix,
)

logger = logging.getLogger(__name__)


class AdaptiveCurriculumTrainer:
    """Trainer for adaptive contrastive curriculum learning.

    This trainer combines:
    1. Adaptive curriculum learning based on task confusion gradients
    2. Uncertainty-weighted contrastive loss for inter-task learning
    3. Production-ready training features (mixed precision, gradient accumulation, etc.)

    Args:
        model: The multi-task model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
        device: Device to train on (cuda/cpu)
        output_dir: Directory for saving checkpoints and logs

    Example:
        >>> config = load_config('config.yaml')
        >>> model = AdaptiveContrastiveMultiTaskModel(**config['model'])
        >>> trainer = AdaptiveCurriculumTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     config=config,
        ...     output_dir='./results'
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: AdaptiveContrastiveMultiTaskModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        output_dir: str = './results',
    ):
        """Initialize the trainer.

        Args:
            model: Multi-task model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Training device (auto-detected if None)
            output_dir: Output directory for checkpoints and logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model.to(self.device)

        logger.info(f"Training on device: {self.device}")

        # Extract config sections
        self.train_config = config.get('training', {})
        self.curriculum_config = config.get('curriculum', {})
        self.contrastive_config = config.get('contrastive', {})
        self.logging_config = config.get('logging', {})

        # Training hyperparameters
        self.num_epochs = self.train_config.get('num_epochs', 10)
        self.batch_size = self.train_config.get('batch_size', 16)
        self.gradient_accumulation_steps = self.train_config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = self.train_config.get('max_grad_norm', 1.0)
        self.mixed_precision = self.train_config.get('mixed_precision', True)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Initialize loss function
        self.criterion = UncertaintyWeightedContrastiveLoss(
            temperature=self.contrastive_config.get('temperature', 0.07),
            lambda_contrastive=self.contrastive_config.get('lambda_contrastive', 0.3),
            lambda_classification=self.contrastive_config.get('lambda_classification', 0.7),
            num_tasks=config.get('model', {}).get('num_tasks', 57),
        )

        # Initialize curriculum components
        self.num_tasks = config.get('model', {}).get('num_tasks', 57)
        self.curriculum_enabled = self.curriculum_config.get('enabled', True)

        if self.curriculum_enabled:
            self.curriculum_scheduler = CurriculumScheduler(
                num_tasks=self.num_tasks,
                strategy=self.curriculum_config.get('strategy', 'uncertainty_gradient'),
                temperature=self.curriculum_config.get('temperature', 2.0),
                warmup_steps=self.curriculum_config.get('warmup_epochs', 2) * len(train_loader),
                min_weight=self.curriculum_config.get('min_task_weight', 0.1),
                max_weight=self.curriculum_config.get('max_task_weight', 3.0),
            )
            self.task_confusion = TaskConfusionMatrix(num_tasks=self.num_tasks)
            self.curriculum_update_freq = self.curriculum_config.get('update_frequency', 100)
        else:
            self.curriculum_scheduler = None
            self.task_confusion = None
            self.curriculum_update_freq = None

        # Mixed precision scaler (not needed for bfloat16)
        self.scaler = None

        # Early stopping
        self.early_stopping_config = self.train_config.get('early_stopping', {})
        self.early_stopping_enabled = self.early_stopping_config.get('enabled', True)
        self.patience = self.early_stopping_config.get('patience', 3)
        self.min_delta = self.early_stopping_config.get('min_delta', 0.001)
        self.monitor_metric = self.early_stopping_config.get('monitor', 'val_accuracy')
        self.best_metric = -float('inf') if 'accuracy' in self.monitor_metric else float('inf')
        self.patience_counter = 0

        # Checkpointing
        self.checkpoint_config = self.train_config.get('checkpoint', {})
        self.save_best = self.checkpoint_config.get('save_best', True)
        self.save_last = self.checkpoint_config.get('save_last', True)
        self.checkpoint_monitor = self.checkpoint_config.get('monitor', 'val_accuracy')
        self.checkpoint_mode = self.checkpoint_config.get('mode', 'max')

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.train_history = []
        self.val_history = []

        # Task-level metrics tracking
        self.task_accuracies = np.zeros(self.num_tasks)
        self.task_samples = np.zeros(self.num_tasks)

        # MLflow integration
        self.use_mlflow = self.logging_config.get('use_mlflow', False)
        self.mlflow_client = None
        if self.use_mlflow:
            self._setup_mlflow()

        # Log interval
        self.log_interval = self.logging_config.get('log_interval', 50)

        # Log model info
        param_info = self.model.get_num_parameters()
        logger.info(f"Model parameters: {param_info['total']:,} total, {param_info['trainable']:,} trainable")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay.

        Returns:
            Configured optimizer
        """
        learning_rate = self.train_config.get('learning_rate', 2e-5)
        weight_decay = self.train_config.get('weight_decay', 0.01)

        # Separate parameters for weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        logger.info(f"Created AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}")

        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler.

        Returns:
            Configured scheduler or None
        """
        scheduler_config = self.train_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        warmup_ratio = self.train_config.get('warmup_ratio', 0.1)

        total_steps = len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        training_steps = total_steps - warmup_steps

        if scheduler_type == 'constant':
            # Constant LR after warmup
            if warmup_steps > 0:
                warmup_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=training_steps)
                scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_steps]
                )
            else:
                scheduler = ConstantLR(self.optimizer, factor=1.0)

        elif scheduler_type == 'linear':
            # Linear decay after warmup
            if warmup_steps > 0:
                warmup_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                main_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=training_steps
                )
                scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_steps]
                )
            else:
                scheduler = LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=total_steps
                )

        elif scheduler_type == 'cosine':
            # Cosine annealing after warmup
            num_cycles = scheduler_config.get('num_cycles', 0.5)
            if warmup_steps > 0:
                warmup_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                main_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=int(training_steps / num_cycles),
                    eta_min=0
                )
                scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_steps]
                )
            else:
                scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=int(total_steps / num_cycles),
                    eta_min=0
                )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, using constant LR")
            scheduler = None

        if scheduler is not None:
            logger.info(f"Created {scheduler_type} scheduler with {warmup_steps} warmup steps")

        return scheduler

    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        try:
            import mlflow

            experiment_name = self.config.get('experiment_name', 'adaptive_contrastive_curriculum')
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()

            # Log config
            mlflow.log_params({
                'model': self.config.get('model', {}).get('base_model', 'unknown'),
                'num_tasks': self.num_tasks,
                'batch_size': self.batch_size,
                'learning_rate': self.train_config.get('learning_rate', 2e-5),
                'num_epochs': self.num_epochs,
                'curriculum_strategy': self.curriculum_config.get('strategy', 'none'),
                'contrastive_enabled': self.contrastive_config.get('enabled', False),
            })

            self.mlflow_client = mlflow
            logger.info(f"MLflow tracking enabled for experiment: {experiment_name}")

        except ImportError:
            logger.warning("MLflow not installed, logging disabled")
            self.use_mlflow = False
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")
            self.use_mlflow = False

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'classification_loss': 0.0,
            'contrastive_loss': 0.0,
            'uncertainty': 0.0,
            'correct': 0,
            'total': 0,
        }

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}",
            leave=True
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            task_ids = batch['task_ids'].to(self.device)

            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, embeddings = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        task_ids=task_ids,
                        return_embeddings=True
                    )
                    loss, loss_dict = self.criterion(logits, embeddings, labels, task_ids)

                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps

                # Backward pass (bfloat16 doesn't need gradient scaling)
                loss.backward()
            else:
                logits, embeddings = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_ids=task_ids,
                    return_embeddings=True
                )
                loss, loss_dict = self.criterion(logits, embeddings, labels, task_ids)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            # Update curriculum if enabled
            if self.curriculum_enabled and self.global_step % self.curriculum_update_freq == 0:
                # Update task confusion matrix
                predictions = logits.argmax(dim=-1)
                task_ids_np = task_ids.cpu().numpy()
                predictions_np = predictions.detach().cpu().numpy()
                embeddings_np = embeddings.detach().float().cpu().numpy()

                self.task_confusion.update(task_ids_np, predictions_np, embeddings_np)

                # Update curriculum scheduler
                confusion_scores = self.task_confusion.get_confusion_scores()
                self.curriculum_scheduler.update(confusion_scores, self.task_accuracies)

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Update metrics
            epoch_metrics['loss'] += loss_dict['total'] * input_ids.size(0)
            epoch_metrics['classification_loss'] += loss_dict['classification'] * input_ids.size(0)
            epoch_metrics['contrastive_loss'] += loss_dict['contrastive'] * input_ids.size(0)
            epoch_metrics['uncertainty'] += loss_dict['mean_uncertainty'] * input_ids.size(0)

            predictions = logits.argmax(dim=-1)
            correct = (predictions == labels).sum().item()
            epoch_metrics['correct'] += correct
            epoch_metrics['total'] += input_ids.size(0)

            # Update task-level accuracy
            for task_id in task_ids.cpu().numpy():
                self.task_samples[task_id] += 1
            for i, task_id in enumerate(task_ids.cpu().numpy()):
                if predictions[i] == labels[i]:
                    self.task_accuracies[task_id] += 1

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'acc': f"{100.0 * epoch_metrics['correct'] / epoch_metrics['total']:.2f}%",
                'lr': f"{current_lr:.2e}"
            })

            # Logging
            if self.global_step % self.log_interval == 0:
                self._log_step(loss_dict, current_lr)

        # Compute epoch averages
        num_samples = epoch_metrics['total']
        epoch_metrics['loss'] /= num_samples
        epoch_metrics['classification_loss'] /= num_samples
        epoch_metrics['contrastive_loss'] /= num_samples
        epoch_metrics['uncertainty'] /= num_samples
        epoch_metrics['accuracy'] = epoch_metrics['correct'] / num_samples

        # Compute task-level accuracies
        for i in range(self.num_tasks):
            if self.task_samples[i] > 0:
                self.task_accuracies[i] /= self.task_samples[i]

        return epoch_metrics

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_metrics = {
            'loss': 0.0,
            'classification_loss': 0.0,
            'contrastive_loss': 0.0,
            'uncertainty': 0.0,
            'correct': 0,
            'total': 0,
        }

        task_correct = np.zeros(self.num_tasks)
        task_total = np.zeros(self.num_tasks)

        pbar = tqdm(self.val_loader, desc="Validation", leave=False)

        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            task_ids = batch['task_ids'].to(self.device)

            # Forward pass
            if self.mixed_precision:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, embeddings = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        task_ids=task_ids,
                        return_embeddings=True
                    )
                    loss, loss_dict = self.criterion(logits, embeddings, labels, task_ids)
            else:
                logits, embeddings = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_ids=task_ids,
                    return_embeddings=True
                )
                loss, loss_dict = self.criterion(logits, embeddings, labels, task_ids)

            # Update metrics
            val_metrics['loss'] += loss_dict['total'] * input_ids.size(0)
            val_metrics['classification_loss'] += loss_dict['classification'] * input_ids.size(0)
            val_metrics['contrastive_loss'] += loss_dict['contrastive'] * input_ids.size(0)
            val_metrics['uncertainty'] += loss_dict['mean_uncertainty'] * input_ids.size(0)

            predictions = logits.argmax(dim=-1)
            correct = (predictions == labels).sum().item()
            val_metrics['correct'] += correct
            val_metrics['total'] += input_ids.size(0)

            # Task-level metrics
            for i, task_id in enumerate(task_ids.cpu().numpy()):
                task_total[task_id] += 1
                if predictions[i] == labels[i]:
                    task_correct[task_id] += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'acc': f"{100.0 * val_metrics['correct'] / val_metrics['total']:.2f}%"
            })

        # Compute averages
        num_samples = val_metrics['total']
        val_metrics['loss'] /= num_samples
        val_metrics['classification_loss'] /= num_samples
        val_metrics['contrastive_loss'] /= num_samples
        val_metrics['uncertainty'] /= num_samples
        val_metrics['accuracy'] = val_metrics['correct'] / num_samples

        # Task-level accuracy
        task_accuracies = []
        for i in range(self.num_tasks):
            if task_total[i] > 0:
                task_acc = task_correct[i] / task_total[i]
                task_accuracies.append(task_acc)

        if task_accuracies:
            val_metrics['mean_task_accuracy'] = np.mean(task_accuracies)
            val_metrics['min_task_accuracy'] = np.min(task_accuracies)
            val_metrics['max_task_accuracy'] = np.max(task_accuracies)

        return val_metrics

    def _log_step(self, loss_dict: Dict[str, float], learning_rate: float) -> None:
        """Log step metrics.

        Args:
            loss_dict: Dictionary of loss values
            learning_rate: Current learning rate
        """
        if self.use_mlflow and self.mlflow_client is not None:
            try:
                self.mlflow_client.log_metrics({
                    'train/loss': loss_dict['total'],
                    'train/classification_loss': loss_dict['classification'],
                    'train/contrastive_loss': loss_dict['contrastive'],
                    'train/uncertainty': loss_dict['mean_uncertainty'],
                    'train/learning_rate': learning_rate,
                }, step=self.global_step)
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

    def _log_epoch(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Log epoch metrics.

        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # Console logging
        logger.info(
            f"Epoch {self.current_epoch + 1}/{self.num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {100.0 * train_metrics['accuracy']:.2f}%, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {100.0 * val_metrics['accuracy']:.2f}%"
        )

        # MLflow logging
        if self.use_mlflow and self.mlflow_client is not None:
            try:
                metrics = {
                    'epoch/train_loss': train_metrics['loss'],
                    'epoch/train_accuracy': train_metrics['accuracy'],
                    'epoch/val_loss': val_metrics['loss'],
                    'epoch/val_accuracy': val_metrics['accuracy'],
                }

                if 'mean_task_accuracy' in val_metrics:
                    metrics['epoch/val_mean_task_accuracy'] = val_metrics['mean_task_accuracy']
                    metrics['epoch/val_min_task_accuracy'] = val_metrics['min_task_accuracy']
                    metrics['epoch/val_max_task_accuracy'] = val_metrics['max_task_accuracy']

                self.mlflow_client.log_metrics(metrics, step=self.current_epoch)
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria is met.

        Args:
            val_metrics: Validation metrics

        Returns:
            True if training should stop
        """
        if not self.early_stopping_enabled:
            return False

        current_metric = val_metrics.get(self.monitor_metric, val_metrics.get('accuracy', 0.0))

        # Check improvement
        if 'accuracy' in self.monitor_metric or self.checkpoint_mode == 'max':
            improved = current_metric > (self.best_metric + self.min_delta)
        else:
            improved = current_metric < (self.best_metric - self.min_delta)

        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            logger.info(f"Early stopping: {self.patience_counter}/{self.patience}")

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.current_epoch + 1} epochs")
                return True

        return False

    def _save_checkpoint(
        self,
        val_metrics: Dict[str, float],
        is_best: bool = False,
        is_last: bool = False
    ) -> None:
        """Save model checkpoint.

        Args:
            val_metrics: Validation metrics
            is_best: Whether this is the best model
            is_last: Whether this is the last checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'val_metrics': val_metrics,
            'config': self.config,
            'best_metric': self.best_metric,
        }

        if is_best and self.save_best:
            checkpoint_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")

            # Save model config
            config_path = self.output_dir / 'model_config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)

        if is_last and self.save_last:
            checkpoint_path = self.output_dir / 'last_model.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved last model to {checkpoint_path}")

        # Save epoch checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pt'
        torch.save(checkpoint, checkpoint_path)

    def train(self) -> Dict[str, List[float]]:
        """Run the full training loop.

        Returns:
            Dictionary containing training history

        Example:
            >>> history = trainer.train()
            >>> print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
        """
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.num_epochs}")
        logger.info(f"Steps per epoch: {len(self.train_loader)}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Total training steps: {len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps}")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Reset task metrics
            self.task_accuracies = np.zeros(self.num_tasks)
            self.task_samples = np.zeros(self.num_tasks)

            # Train epoch
            train_metrics = self._train_epoch()
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self._validate()
            self.val_history.append(val_metrics)

            # Log metrics
            self._log_epoch(train_metrics, val_metrics)

            # Check for best model
            current_metric = val_metrics.get(self.checkpoint_monitor, val_metrics.get('accuracy', 0.0))
            is_best = False

            if self.checkpoint_mode == 'max':
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
            else:
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    is_best = True

            # Save checkpoint
            self._save_checkpoint(val_metrics, is_best=is_best, is_last=(epoch == self.num_epochs - 1))

            # Check early stopping
            if self._check_early_stopping(val_metrics):
                break

        # Training complete
        logger.info("Training completed!")
        logger.info(f"Best {self.checkpoint_monitor}: {self.best_metric:.4f}")

        # Close MLflow run
        if self.use_mlflow and self.mlflow_client is not None:
            try:
                self.mlflow_client.end_run()
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")

        # Compile history
        history = {
            'train_loss': [m['loss'] for m in self.train_history],
            'train_accuracy': [m['accuracy'] for m in self.train_history],
            'val_loss': [m['loss'] for m in self.val_history],
            'val_accuracy': [m['accuracy'] for m in self.val_history],
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_accuracy': self.best_metric,
        }

        return history

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load a checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint file

        Example:
            >>> trainer.load_checkpoint('results/best_model.pt')
            >>> trainer.train()  # Resume training
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load scaler state
        if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', self.best_metric)

        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Example:
        >>> config = load_config('configs/default.yaml')
        >>> print(config['model']['base_model'])
        'bert-base-uncased'
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config
