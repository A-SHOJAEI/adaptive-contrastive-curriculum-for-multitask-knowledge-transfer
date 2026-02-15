"""Multi-task model with adaptive contrastive learning."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

logger = logging.getLogger(__name__)


class AdaptiveContrastiveMultiTaskModel(nn.Module):
    """Multi-task model combining contrastive learning with adaptive curriculum.

    Architecture:
    1. Shared transformer encoder (DeBERTa)
    2. Task-specific projection heads
    3. Contrastive projection for inter-task learning
    4. Multi-class classification head
    """

    def __init__(
        self,
        base_model: str,
        num_tasks: int,
        hidden_dim: int,
        projection_dim: int,
        num_classes: int = 4,  # MMLU has 4 answer choices
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        """Initialize the model.

        Args:
            base_model: Name of the pre-trained transformer model
            num_tasks: Number of tasks
            hidden_dim: Hidden dimension of the transformer
            projection_dim: Dimension of contrastive projection
            num_classes: Number of answer choices
            dropout: Dropout probability
            freeze_base: Whether to freeze the base model
        """
        super().__init__()

        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim

        # Load pre-trained transformer
        logger.info(f"Loading base model: {base_model}")
        self.encoder = AutoModel.from_pretrained(base_model)

        if freeze_base:
            logger.info("Freezing base model parameters")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Task embeddings
        self.task_embeddings = nn.Embedding(num_tasks, hidden_dim)

        # Contrastive projection head
        self.contrastive_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat encoder + task embedding
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.task_embeddings.weight)

        for module in [self.contrastive_projection, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_ids: torch.Tensor,
        return_embeddings: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            task_ids: Task IDs [batch_size]
            return_embeddings: Whether to return contrastive embeddings

        Returns:
            Tuple of (logits, embeddings) or just logits
        """
        # Encode input
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Use [CLS] token representation
        cls_output = encoder_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]

        # Get task embeddings
        task_emb = self.task_embeddings(task_ids)  # [batch_size, hidden_dim]

        # Concatenate for classification
        combined = torch.cat([cls_output, task_emb], dim=-1)  # [batch_size, hidden_dim * 2]

        # Classification logits
        logits = self.classifier(combined)  # [batch_size, num_classes]

        if return_embeddings:
            # Contrastive embeddings
            embeddings = self.contrastive_projection(cls_output)  # [batch_size, projection_dim]
            return logits, embeddings
        else:
            return logits, None

    def get_task_representations(self) -> torch.Tensor:
        """Get learned task representations.

        Returns:
            Task embeddings [num_tasks, hidden_dim]
        """
        return self.task_embeddings.weight.detach()

    def freeze_encoder(self) -> None:
        """Freeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Froze encoder parameters")

    def unfreeze_encoder(self) -> None:
        """Unfreeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Unfroze encoder parameters")

    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters.

        Returns:
            Dictionary with parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder = sum(p.numel() for p in self.encoder.parameters())

        return {
            'total': total,
            'trainable': trainable,
            'encoder': encoder,
            'task_embeddings': sum(p.numel() for p in self.task_embeddings.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
            'contrastive': sum(p.numel() for p in self.contrastive_projection.parameters()),
        }
