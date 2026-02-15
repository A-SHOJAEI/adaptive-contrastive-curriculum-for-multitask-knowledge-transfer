#!/usr/bin/env python
"""Evaluation script for MMLU benchmark."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.data.loader import MMLUDataLoader
from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.models.model import (
    AdaptiveContrastiveMultiTaskModel,
)
from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.evaluation.analysis import (
    ResultsAnalyzer,
)
from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.evaluation.metrics import (
    MetricsCalculator,
)
from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.utils.config import (
    load_config,
    set_seed,
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
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU")
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
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on dataset.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to use

    Returns:
        Dictionary with predictions, labels, and task IDs
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_task_ids = []
    all_logits = []

    for batch in tqdm(data_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        task_ids = batch['task_ids'].to(device)

        # Forward pass
        logits, _ = model(input_ids, attention_mask, task_ids, return_embeddings=False)

        predictions = logits.argmax(dim=-1)

        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_task_ids.append(task_ids.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    task_ids = np.concatenate(all_task_ids)
    logits = np.concatenate(all_logits)

    return {
        'predictions': predictions,
        'labels': labels,
        'task_ids': task_ids,
        'logits': logits,
    }


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    task_ids: np.ndarray,
    task_to_category: dict,
) -> dict:
    """Compute comprehensive metrics.

    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        task_ids: Task IDs
        task_to_category: Mapping from task ID to category

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = {}

    # Overall metrics
    metrics['accuracy'] = float(accuracy_score(labels, predictions))
    metrics['f1_macro'] = float(f1_score(labels, predictions, average='macro', zero_division=0))
    metrics['f1_micro'] = float(f1_score(labels, predictions, average='micro', zero_division=0))
    metrics['precision'] = float(precision_score(labels, predictions, average='macro', zero_division=0))
    metrics['recall'] = float(recall_score(labels, predictions, average='macro', zero_division=0))

    # Per-task metrics
    task_accuracies = {}
    task_f1_scores = {}

    num_tasks = len(np.unique(task_ids))

    for task_id in range(num_tasks):
        task_mask = task_ids == task_id
        if task_mask.sum() == 0:
            continue

        task_preds = predictions[task_mask]
        task_labels = labels[task_mask]

        task_accuracies[int(task_id)] = float(accuracy_score(task_labels, task_preds))
        task_f1_scores[int(task_id)] = float(
            f1_score(task_labels, task_preds, average='macro', zero_division=0)
        )

    metrics['task_accuracies'] = task_accuracies
    metrics['task_f1_scores'] = task_f1_scores
    metrics['mean_task_accuracy'] = float(np.mean(list(task_accuracies.values())))
    metrics['std_task_accuracy'] = float(np.std(list(task_accuracies.values())))

    # Per-category metrics
    category_accuracies = {}

    for task_id, category in task_to_category.items():
        if category not in category_accuracies:
            category_accuracies[category] = []

        task_mask = task_ids == task_id
        if task_mask.sum() == 0:
            continue

        task_preds = predictions[task_mask]
        task_labels = labels[task_mask]
        category_accuracies[category].append(accuracy_score(task_labels, task_preds))

    for category, accs in category_accuracies.items():
        if len(accs) > 0:
            metrics[f'{category}_accuracy'] = float(np.mean(accs))

    # Cross-domain transfer (variance-based measure)
    if len(task_accuracies) > 0:
        accs = list(task_accuracies.values())
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        metrics['cross_domain_transfer'] = float(mean_acc * (1.0 - min(std_acc, 1.0)))

    # Task confusion reduction
    unique_tasks = np.unique(task_ids)
    if len(unique_tasks) >= 2:
        task_diversity = []

        for task_id in unique_tasks:
            task_mask = task_ids == task_id
            task_preds = predictions[task_mask]

            if len(task_preds) > 0:
                unique, counts = np.unique(task_preds, return_counts=True)
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                max_entropy = np.log(len(unique) + 1e-8)
                if max_entropy > 0:
                    task_diversity.append(entropy / max_entropy)

        if len(task_diversity) > 0:
            metrics['task_confusion_reduction'] = float(1.0 - np.mean(task_diversity))

    return metrics


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Evaluating Adaptive Contrastive Curriculum Model on MMLU")
    logger.info("=" * 80)

    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        logger.info(f"Loaded checkpoint from {args.checkpoint}")

        # Get config from checkpoint or file
        if args.config:
            config = load_config(args.config)
        elif 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("Config not found in checkpoint and --config not provided")

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    # Set seed
    set_seed(config.get('seed', 42))

    # Get device
    device = get_device()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    try:
        model_name = config['model']['base_model']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    # Load data
    try:
        logger.info("Loading MMLU dataset...")
        data_config = config['data']
        data_loader = MMLUDataLoader(data_config, tokenizer)

        train_data, val_data, test_data = data_loader.load_data()

        # Select split
        if args.split == 'train':
            eval_data = train_data
        elif args.split == 'val':
            eval_data = val_data
        else:
            eval_data = test_data

        logger.info(f"Evaluating on {args.split} split with {len(eval_data)} samples")

        # Create dataloader
        batch_size = config.get('evaluation', {}).get('batch_size', 32)
        _, _, eval_loader = data_loader.create_dataloaders(
            train_data[:1], val_data[:1], eval_data, batch_size, num_workers=4
        )

        # Get task to category mapping
        task_to_category = {}
        for task_id, task_name in data_loader.id_to_task.items():
            task_to_category[task_id] = data_loader.get_task_category(task_name)

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.exception("Detailed error:")
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
        logger.exception("Detailed error:")
        sys.exit(1)

    # Evaluate
    try:
        logger.info("Running evaluation...")
        results = evaluate_model(model, eval_loader, device)

        # Compute metrics
        metrics = compute_metrics(
            results['predictions'],
            results['labels'],
            results['task_ids'],
            task_to_category,
        )

        logger.info("Evaluation complete!")
        logger.info("=" * 80)
        logger.info("Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (Macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  Mean Task Accuracy: {metrics['mean_task_accuracy']:.4f}")
        logger.info(f"  Std Task Accuracy: {metrics['std_task_accuracy']:.4f}")

        if 'cross_domain_transfer' in metrics:
            logger.info(f"  Cross-Domain Transfer: {metrics['cross_domain_transfer']:.4f}")
        if 'task_confusion_reduction' in metrics:
            logger.info(f"  Task Confusion Reduction: {metrics['task_confusion_reduction']:.4f}")

        logger.info("=" * 80)

        # Save results
        analyzer = ResultsAnalyzer(results_dir=str(output_dir))
        analyzer.save_metrics(metrics, filename=f'{args.split}_metrics.json')

        # Create summary table
        summary_df = analyzer.create_summary_table(metrics, save_csv=True)
        print("\nSummary Table:")
        print(summary_df.to_string(index=False))

        # Plot task performance
        if 'task_accuracies' in metrics:
            analyzer.plot_task_performance(
                metrics['task_accuracies'],
                filename=f'{args.split}_task_performance.png',
            )

        logger.info(f"Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Detailed error:")
        sys.exit(1)


if __name__ == "__main__":
    main()
