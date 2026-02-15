"""Custom loss functions and training components.

This module implements the novel components:
1. Uncertainty-weighted contrastive loss
2. Adaptive curriculum scheduler based on task confusion
3. Task confusion matrix for inter-task relationship modeling
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class UncertaintyWeightedContrastiveLoss(nn.Module):
    """Custom contrastive loss weighted by prediction uncertainty.

    This loss combines:
    1. Standard supervised contrastive learning between tasks
    2. Uncertainty-based weighting to focus on confusing samples
    3. Inter-task contrast to model task relationships explicitly
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_contrastive: float = 0.3,
        lambda_classification: float = 0.7,
        num_tasks: int = 57,
    ):
        """Initialize uncertainty-weighted contrastive loss.

        Args:
            temperature: Temperature for contrastive loss
            lambda_contrastive: Weight for contrastive component
            lambda_classification: Weight for classification component
            num_tasks: Number of tasks
        """
        super().__init__()
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
        self.lambda_classification = lambda_classification
        self.num_tasks = num_tasks
        self.classification_loss = nn.CrossEntropyLoss(reduction='none')

    def compute_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute prediction uncertainty using entropy.

        Args:
            logits: Model logits [batch_size, num_classes]

        Returns:
            Uncertainty scores [batch_size]
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        # Normalize to [0, 1]
        max_entropy = np.log(logits.size(-1))
        return entropy / max_entropy

    def inter_task_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        task_ids: torch.Tensor,
        uncertainties: torch.Tensor,
    ) -> torch.Tensor:
        """Compute inter-task contrastive loss using InfoNCE.

        Uses the standard InfoNCE formulation:
        -log(exp(sim_pos/tau) / sum(exp(sim_neg/tau)))

        Args:
            embeddings: Sample embeddings [batch_size, embedding_dim]
            task_ids: Task IDs [batch_size]
            uncertainties: Uncertainty scores [batch_size]

        Returns:
            Contrastive loss value
        """
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create masks for positive and negative pairs
        task_mask = task_ids.unsqueeze(0) == task_ids.unsqueeze(1)
        task_mask.fill_diagonal_(False)  # Exclude self-similarity

        # Negative pairs are from different tasks
        negative_mask = ~task_mask
        negative_mask.fill_diagonal_(False)

        # Compute InfoNCE loss with uncertainty weighting
        losses = []
        for i in range(batch_size):
            # Positive pairs (same task)
            pos_mask = task_mask[i]
            if pos_mask.sum() == 0:
                continue

            # Negative pairs (different tasks)
            neg_mask = negative_mask[i]
            if neg_mask.sum() == 0:
                continue

            # Weight by uncertainty - focus more on uncertain samples
            weight = uncertainties[i]

            # InfoNCE: -log(exp(sim_pos) / (exp(sim_pos) + sum(exp(sim_neg))))
            pos_sim = similarity_matrix[i][pos_mask]  # Already scaled by temperature
            neg_sim = similarity_matrix[i][neg_mask]

            # For each positive, compute InfoNCE loss
            for pos in pos_sim:
                # Numerator: exp(positive similarity)
                numerator = torch.exp(pos)

                # Denominator: exp(positive) + sum(exp(negatives))
                denominator = numerator + torch.exp(neg_sim).sum()

                # InfoNCE loss: -log(numerator / denominator)
                loss = -torch.log(numerator / (denominator + 1e-8))

                # Weight by uncertainty
                weighted_loss = loss * weight
                losses.append(weighted_loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return torch.stack(losses).mean()

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            logits: Classification logits [batch_size, num_classes]
            embeddings: Contrastive embeddings [batch_size, embedding_dim]
            labels: Ground truth labels [batch_size]
            task_ids: Task IDs [batch_size]

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Classification loss
        classification_loss = self.classification_loss(logits, labels).mean()

        # Compute uncertainty
        uncertainties = self.compute_uncertainty(logits)

        # Contrastive loss
        if self.lambda_contrastive > 0:
            contrastive_loss = self.inter_task_contrastive_loss(
                embeddings, task_ids, uncertainties
            )
        else:
            contrastive_loss = torch.tensor(0.0, device=logits.device)

        # Combined loss
        total_loss = (
            self.lambda_classification * classification_loss +
            self.lambda_contrastive * contrastive_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'classification': classification_loss.item(),
            'contrastive': contrastive_loss.item(),
            'mean_uncertainty': uncertainties.mean().item(),
        }

        return total_loss, loss_dict


class TaskConfusionMatrix:
    """Track inter-task confusion for curriculum learning.

    This tracks which tasks are frequently confused with each other,
    allowing the curriculum to prioritize samples that improve
    task disambiguation.
    """

    def __init__(self, num_tasks: int, smoothing: float = 0.1):
        """Initialize task confusion matrix.

        Args:
            num_tasks: Number of tasks
            smoothing: Smoothing factor for updates
        """
        self.num_tasks = num_tasks
        self.smoothing = smoothing
        self.confusion_matrix = np.zeros((num_tasks, num_tasks))
        self.total_samples = np.zeros(num_tasks)

    def update(
        self,
        task_ids: np.ndarray,
        predictions: np.ndarray,
        embeddings: np.ndarray,
    ) -> None:
        """Update confusion matrix based on predictions.

        Args:
            task_ids: True task IDs [batch_size]
            predictions: Predicted class IDs [batch_size]
            embeddings: Sample embeddings [batch_size, embedding_dim]
        """
        batch_size = len(task_ids)

        # Update total samples per task
        for task_id in task_ids:
            self.total_samples[task_id] += 1

        # Compute pairwise similarities to detect confusion
        if batch_size > 1:
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / (norms + 1e-8)

            # Compute similarity matrix
            similarities = np.matmul(normalized, normalized.T)

            # Update confusion for similar samples from different tasks
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    if task_ids[i] != task_ids[j]:
                        # High similarity between different tasks = confusion
                        confusion_score = max(0, similarities[i, j])
                        self.confusion_matrix[task_ids[i], task_ids[j]] += confusion_score
                        self.confusion_matrix[task_ids[j], task_ids[i]] += confusion_score

    def get_confusion_scores(self) -> np.ndarray:
        """Get confusion scores for each task.

        Returns:
            Array of confusion scores [num_tasks]
        """
        # Sum confusion with all other tasks
        confusion_scores = self.confusion_matrix.sum(axis=1)

        # Normalize by number of samples
        normalized = np.zeros(self.num_tasks)
        for i in range(self.num_tasks):
            if self.total_samples[i] > 0:
                normalized[i] = confusion_scores[i] / self.total_samples[i]

        return normalized

    def get_confusion_matrix(self) -> np.ndarray:
        """Get the full confusion matrix.

        Returns:
            Confusion matrix [num_tasks, num_tasks]
        """
        return self.confusion_matrix.copy()


class CurriculumScheduler:
    """Adaptive curriculum scheduler based on task confusion gradients.

    This scheduler dynamically adjusts task sampling probabilities based on:
    1. Task confusion scores (prioritize confusing tasks)
    2. Current training progress
    3. Temperature-based smoothing
    """

    def __init__(
        self,
        num_tasks: int,
        strategy: str = 'uncertainty_gradient',
        temperature: float = 2.0,
        warmup_steps: int = 1000,
        min_weight: float = 0.1,
        max_weight: float = 3.0,
    ):
        """Initialize curriculum scheduler.

        Args:
            num_tasks: Number of tasks
            strategy: Curriculum strategy ('uniform', 'difficulty', 'uncertainty_gradient')
            temperature: Temperature for sampling distribution
            warmup_steps: Number of warmup steps
            min_weight: Minimum task weight
            max_weight: Maximum task weight
        """
        self.num_tasks = num_tasks
        self.strategy = strategy
        self.temperature = temperature
        self.warmup_steps = warmup_steps
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Initialize uniform weights
        self.task_weights = np.ones(num_tasks)
        self.step = 0

    def update(
        self,
        confusion_scores: np.ndarray,
        task_accuracies: Optional[np.ndarray] = None,
    ) -> None:
        """Update task weights based on confusion and performance.

        Args:
            confusion_scores: Confusion scores for each task [num_tasks]
            task_accuracies: Optional accuracy for each task [num_tasks]
        """
        self.step += 1

        if self.step < self.warmup_steps:
            # Warmup phase: uniform sampling
            self.task_weights = np.ones(self.num_tasks)
            return

        if self.strategy == 'uniform':
            # Uniform sampling
            self.task_weights = np.ones(self.num_tasks)

        elif self.strategy == 'uncertainty_gradient':
            # Prioritize tasks with high confusion (novel approach)
            # Normalize confusion scores
            if confusion_scores.sum() > 0:
                normalized_confusion = confusion_scores / confusion_scores.sum()
            else:
                normalized_confusion = np.ones(self.num_tasks) / self.num_tasks

            # Apply temperature scaling
            weights = np.exp(normalized_confusion / self.temperature)
            weights = weights / weights.sum()

            # Scale to [min_weight, max_weight]
            weights = self.min_weight + (self.max_weight - self.min_weight) * weights

            self.task_weights = weights

        elif self.strategy == 'difficulty':
            # Prioritize difficult tasks (low accuracy)
            if task_accuracies is not None:
                difficulty = 1.0 - task_accuracies
                weights = np.exp(difficulty / self.temperature)
                weights = weights / weights.sum()
                weights = self.min_weight + (self.max_weight - self.min_weight) * weights
                self.task_weights = weights
            else:
                self.task_weights = np.ones(self.num_tasks)

    def get_task_weights(self) -> np.ndarray:
        """Get current task sampling weights.

        Returns:
            Task weights [num_tasks]
        """
        return self.task_weights.copy()

    def sample_task(self) -> int:
        """Sample a task according to current weights.

        Returns:
            Task ID
        """
        probs = self.task_weights / self.task_weights.sum()
        return np.random.choice(self.num_tasks, p=probs)
