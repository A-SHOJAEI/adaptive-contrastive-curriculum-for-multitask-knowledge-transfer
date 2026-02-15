#!/usr/bin/env python
"""Prediction script for MMLU questions."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer

from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.models.model import (
    AdaptiveContrastiveMultiTaskModel,
)
from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.utils.config import (
    load_config,
    get_device,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (if not in checkpoint)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question text",
    )
    parser.add_argument(
        "--choices",
        type=str,
        nargs='+',
        default=None,
        help="Answer choices (space-separated)",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="Task ID (default: 0)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="JSON file with questions (list of dicts with 'question', 'choices', 'task_id')",
    )
    return parser.parse_args()


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    question: str,
    choices: List[str],
    task_id: int,
    device: torch.device,
    max_length: int = 512,
) -> dict:
    """Make prediction for a single question.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        question: Question text
        choices: List of answer choices
        task_id: Task ID
        device: Device to use
        max_length: Maximum sequence length

    Returns:
        Dictionary with prediction and confidence scores
    """
    model.eval()

    # Format question with choices
    formatted_text = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        formatted_text += f"{chr(65+i)}. {choice}\n"

    # Tokenize
    encoding = tokenizer(
        formatted_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    task_ids = torch.tensor([task_id], dtype=torch.long, device=device)

    # Forward pass
    logits, _ = model(input_ids, attention_mask, task_ids, return_embeddings=False)

    # Get prediction
    probs = torch.softmax(logits, dim=-1)
    predicted_idx = logits.argmax(dim=-1).item()
    confidence = probs[0, predicted_idx].item()

    return {
        'predicted_answer': chr(65 + predicted_idx),
        'predicted_index': predicted_idx,
        'confidence': confidence,
        'probabilities': {
            chr(65 + i): prob.item()
            for i, prob in enumerate(probs[0])
        },
    }


def main() -> None:
    """Main prediction function."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("MMLU Question Answering with Adaptive Contrastive Model")
    logger.info("=" * 80)

    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        logger.info(f"Loaded checkpoint from {args.checkpoint}")

        # Get config
        if args.config:
            config = load_config(args.config)
        elif 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("Config not found in checkpoint and --config not provided")

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    # Get device
    device = get_device()

    # Load tokenizer
    try:
        model_name = config['model']['base_model']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    # Create model
    try:
        logger.info("Creating model...")
        model_config = config['model']

        model = AdaptiveContrastiveMultiTaskModel(
            base_model=model_config['base_model'],
            num_tasks=model_config['num_tasks'],
            hidden_dim=model_config['hidden_dim'],
            projection_dim=model_config['projection_dim'],
            num_classes=4,
            dropout=model_config.get('dropout', 0.1),
            freeze_base=model_config.get('freeze_base', False),
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        sys.exit(1)

    # Make predictions
    try:
        if args.input_file:
            # Batch prediction from file
            with open(args.input_file, 'r') as f:
                questions = json.load(f)

            results = []
            for i, item in enumerate(questions):
                question = item['question']
                choices = item['choices']
                task_id = item.get('task_id', 0)

                result = predict(
                    model, tokenizer, question, choices, task_id, device
                )

                result['question'] = question
                result['choices'] = choices
                result['task_id'] = task_id

                results.append(result)

                logger.info(f"Question {i+1}/{len(questions)}: {result['predicted_answer']} "
                          f"(confidence: {result['confidence']:.4f})")

            # Save results
            output_file = Path(args.input_file).stem + "_predictions.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"Saved predictions to {output_file}")

        else:
            # Single prediction from command line
            if not args.question or not args.choices:
                logger.error("--question and --choices required for single prediction")
                sys.exit(1)

            result = predict(
                model,
                tokenizer,
                args.question,
                args.choices,
                args.task_id,
                device,
            )

            logger.info("=" * 80)
            logger.info("Prediction Results:")
            logger.info(f"  Question: {args.question}")
            logger.info(f"  Choices: {', '.join([f'{chr(65+i)}. {c}' for i, c in enumerate(args.choices)])}")
            logger.info(f"  Predicted Answer: {result['predicted_answer']}")
            logger.info(f"  Confidence: {result['confidence']:.4f}")
            logger.info("\n  Probabilities:")
            for choice, prob in result['probabilities'].items():
                logger.info(f"    {choice}: {prob:.4f}")
            logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.exception("Detailed error:")
        sys.exit(1)


if __name__ == "__main__":
    main()
