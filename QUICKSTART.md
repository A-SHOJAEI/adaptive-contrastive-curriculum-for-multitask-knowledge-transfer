# Quick Start Guide

Get started with Adaptive Contrastive Curriculum Learning in 3 steps.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 16GB RAM minimum

## Installation

```bash
# Clone or navigate to project directory
cd adaptive-contrastive-curriculum-for-multitask-knowledge-transfer

# Install dependencies
pip install -r requirements.txt

# Verify installation
bash verify_project.sh
```

## Training

### Quick Demo (2 epochs, 20 samples/task)

```bash
python3 scripts/train.py --config configs/demo.yaml
```

Expected time: ~10 minutes on GPU, ~1 hour on CPU

### Full Training (10 epochs, 500 samples/task)

```bash
python3 scripts/train.py --config configs/default.yaml
```

Expected time: ~6 hours on GPU

### Baseline (No curriculum/contrastive)

```bash
python3 scripts/train.py --config configs/ablation.yaml
```

## Evaluation

```bash
python3 scripts/evaluate.py \
    --checkpoint models/best_model.pt \
    --split test
```

## Single Prediction

```bash
python3 scripts/predict.py \
    --checkpoint models/best_model.pt \
    --question "What is the powerhouse of the cell?" \
    --choices "Nucleus" "Mitochondria" "Ribosome" "Golgi" \
    --task_id 0
```

## Configuration

Edit `configs/default.yaml` to adjust:

```yaml
curriculum:
  strategy: "uncertainty_gradient"  # uniform, difficulty, uncertainty_gradient
  temperature: 2.0                  # Higher = smoother task distribution

contrastive:
  lambda_contrastive: 0.3          # Weight for contrastive loss
  temperature: 0.07                 # Lower = harder negatives

training:
  learning_rate: 0.00002
  batch_size: 16
  num_epochs: 10
```

## Output Files

After training:

```
models/
  ├── best_model.pt          # Best validation checkpoint
  └── last_model.pt          # Most recent checkpoint

results/
  ├── config.yaml            # Training configuration
  ├── training_metrics.json  # Final metrics
  └── training_curves.png    # Loss/accuracy plots
```

## Troubleshooting

**Out of memory?**
- Reduce `batch_size` in config
- Enable `gradient_accumulation_steps: 4`
- Use `freeze_base: true` to freeze encoder

**Slow data loading?**
- Increase `num_workers` in config
- Reduce `max_samples_per_task`

**MLflow errors?**
- Set `use_mlflow: false` in config

## Next Steps

1. **Run ablation study**: Compare `default.yaml` vs `ablation.yaml`
2. **Analyze confusion matrix**: Check `results/task_confusion_matrix.png`
3. **Visualize curriculum**: Plot task weights over time
4. **Experiment with hyperparameters**: Adjust temperatures, loss weights

## Documentation

- `README.md`: Project overview
- `RESULTS.md`: Methodology and theory
- `IMPROVEMENTS.md`: Recent fixes and changes
- `UPGRADE_SUMMARY.md`: Quality assessment

## Support

Check existing issues in the codebase or create a new one with:
- Python version
- GPU/CPU
- Error message
- Config file used
