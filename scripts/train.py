#!/usr/bin/env python
"""Training script for adaptive contrastive curriculum learning on MMLU."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer

from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.data.loader import MMLUDataLoader
from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.models.model import (
    AdaptiveContrastiveMultiTaskModel,
)
from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.training.trainer import (
    AdaptiveCurriculumTrainer,
)
from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.utils.config import (
    load_config,
    set_seed,
    get_device,
    save_config,
)
from adaptive_contrastive_curriculum_for_multitask_knowledge_transfer.evaluation.analysis import (
    ResultsAnalyzer,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log'),
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train adaptive contrastive curriculum model on MMLU"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Adaptive Contrastive Curriculum Learning for MMLU")
    logger.info("=" * 80)

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)

    # Get device
    device = get_device()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config to results dir
    save_config(config, str(results_dir / "config.yaml"))

    # Load tokenizer
    try:
        model_name = config['model']['base_model']
        logger.info(f"Loading tokenizer: {model_name}")
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
        logger.info(
            f"Loaded {len(train_data)} train, {len(val_data)} val, "
            f"{len(test_data)} test samples"
        )

        # Create dataloaders
        batch_size = config['training']['batch_size']
        num_workers = data_config.get('num_workers', 4)

        train_loader, val_loader, test_loader = data_loader.create_dataloaders(
            train_data, val_data, test_data, batch_size, num_workers
        )

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
            num_classes=4,  # MMLU has 4 answer choices
            dropout=model_config.get('dropout', 0.1),
            freeze_base=model_config.get('freeze_base', False),
        )

        # Log model info
        param_counts = model.get_num_parameters()
        logger.info(f"Model parameters: {param_counts['total']:,}")
        logger.info(f"Trainable parameters: {param_counts['trainable']:,}")

    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        logger.exception("Detailed error:")
        sys.exit(1)

    # Create trainer
    try:
        logger.info("Creating trainer...")
        trainer = AdaptiveCurriculumTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            output_dir=str(output_dir),
        )

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        logger.exception("Detailed error:")
        sys.exit(1)

    # Initialize MLflow (optional)
    if config.get('logging', {}).get('use_mlflow', False):
        try:
            import mlflow

            mlflow.set_experiment(config.get('experiment_name', 'adaptive_contrastive_mmlu'))
            mlflow.start_run()
            mlflow.log_params({
                'base_model': model_config['base_model'],
                'batch_size': batch_size,
                'learning_rate': config['training']['learning_rate'],
                'num_epochs': config['training']['num_epochs'],
                'curriculum_enabled': config.get('curriculum', {}).get('enabled', True),
                'contrastive_enabled': config.get('contrastive', {}).get('enabled', True),
            })
            logger.info("MLflow tracking enabled")
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")

    # Train model
    try:
        logger.info("Starting training...")
        history = trainer.train()

        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {history['best_val_accuracy']:.4f}")

        # Log to MLflow
        if config.get('logging', {}).get('use_mlflow', False):
            try:
                mlflow.log_metric("best_val_accuracy", history['best_val_accuracy'])
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Detailed error:")
        if config.get('logging', {}).get('use_mlflow', False):
            try:
                mlflow.end_run(status='FAILED')
            except:
                pass
        sys.exit(1)

    # Save training history
    try:
        analyzer = ResultsAnalyzer(results_dir=str(results_dir))

        # Save metrics
        final_metrics = {
            'best_val_accuracy': history['best_val_accuracy'],
            'final_train_loss': history['train_history'][-1]['loss'],
            'final_val_loss': history['val_history'][-1]['loss'],
        }
        analyzer.save_metrics(final_metrics, filename='training_metrics.json')

        # Plot training curves
        analyzer.plot_training_curves(
            history['train_history'],
            history['val_history'],
            filename='training_curves.png',
        )

        logger.info(f"Saved results to {results_dir}")

    except Exception as e:
        logger.warning(f"Failed to save analysis: {e}")

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best model saved to: {output_dir / 'best_model.pt'}")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
