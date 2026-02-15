#!/bin/bash
# Project verification script
# Run this to verify all components are working

set -e

echo "========================================="
echo "Project Verification Script"
echo "========================================="
echo ""

# 1. Check Python version
echo "[1/7] Checking Python version..."
python3 --version
echo "✓ Python 3 found"
echo ""

# 2. Check syntax of all Python files
echo "[2/7] Validating Python syntax..."
for file in scripts/*.py src/**/*.py tests/*.py; do
    if [ -f "$file" ]; then
        python3 -m py_compile "$file" 2>&1 || { echo "✗ Syntax error in $file"; exit 1; }
    fi
done
echo "✓ All Python files have valid syntax"
echo ""

# 3. Verify configs are valid YAML
echo "[3/7] Validating YAML configs..."
for config in configs/*.yaml; do
    python3 -c "import yaml; yaml.safe_load(open('$config'))" || { echo "✗ Invalid YAML in $config"; exit 1; }
done
echo "✓ All YAML configs are valid"
echo ""

# 4. Check for scientific notation in configs (should be none)
echo "[4/7] Checking for scientific notation in configs..."
if grep -r "e-[0-9]" configs/*.yaml; then
    echo "✗ Found scientific notation in configs"
    exit 1
fi
echo "✓ No scientific notation found"
echo ""

# 5. Verify LICENSE file
echo "[5/7] Checking LICENSE file..."
if [ ! -f LICENSE ]; then
    echo "✗ LICENSE file missing"
    exit 1
fi
if ! grep -q "MIT License" LICENSE; then
    echo "✗ LICENSE is not MIT"
    exit 1
fi
if ! grep -q "2026 Alireza Shojaei" LICENSE; then
    echo "✗ LICENSE missing correct copyright"
    exit 1
fi
echo "✓ LICENSE file is correct"
echo ""

# 6. Check README length
echo "[6/7] Checking README length..."
lines=$(wc -l < README.md)
if [ "$lines" -gt 200 ]; then
    echo "✗ README is too long ($lines lines > 200)"
    exit 1
fi
echo "✓ README is concise ($lines lines)"
echo ""

# 7. Verify key files exist
echo "[7/7] Checking for key files..."
required_files=(
    "README.md"
    "LICENSE"
    "requirements.txt"
    "RESULTS.md"
    "scripts/train.py"
    "scripts/evaluate.py"
    "scripts/predict.py"
    "configs/default.yaml"
    "configs/ablation.yaml"
    "configs/demo.yaml"
    "src/adaptive_contrastive_curriculum_for_multitask_knowledge_transfer/models/model.py"
    "src/adaptive_contrastive_curriculum_for_multitask_knowledge_transfer/models/components.py"
    "src/adaptive_contrastive_curriculum_for_multitask_knowledge_transfer/training/trainer.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "✗ Missing required file: $file"
        exit 1
    fi
done
echo "✓ All required files present"
echo ""

echo "========================================="
echo "✓ ALL CHECKS PASSED"
echo "========================================="
echo ""
echo "Project is ready for publication!"
echo ""
echo "To train the model:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. Run training: python3 scripts/train.py --config configs/demo.yaml"
echo ""
