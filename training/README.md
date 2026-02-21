# Floorplan Segmentation Training

## Quick Start

### 1. Setup

```bash
cd training
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python generate_synthetic.py --count 2000 --output-dir data/prepared/synthetic/train
python generate_synthetic.py --count 500 --output-dir data/prepared/synthetic/val --seed 999
```

### 3. (Optional) Prepare CubiCasa5K

```bash
git clone https://github.com/CubiCasa/CubiCasa5k data/cubicasa5k
python prepare_cubicasa.py --data-dir data/cubicasa5k --output-dir data/prepared/cubicasa
```

### 4. Train

```bash
# Synthetic only:
python train.py \
  --data-dirs data/prepared/synthetic/train \
  --val-dirs data/prepared/synthetic/val \
  --output-dir checkpoints/segformer-floorplan

# Synthetic + CubiCasa5K:
python train.py \
  --data-dirs data/prepared/cubicasa/train data/prepared/synthetic/train \
  --val-dirs data/prepared/cubicasa/val data/prepared/synthetic/val \
  --output-dir checkpoints/segformer-floorplan
```

### 5. Test locally

```bash
python predict.py \
  --model checkpoints/segformer-floorplan/best \
  --image ../test-data/U3laAgYJ.jpg \
  --output result.json \
  --save-mask
```

### 6. Deploy to Replicate

```bash
# Copy best model to ./model/ for Cog
cp -r checkpoints/segformer-floorplan/best model

# Push to Replicate
cog push r8.im/your-username/floorplan-segformer
```

## Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | background | Everything else |
| 1 | wall | Walls |
| 2 | door | Doors |
| 3 | window | Windows |
| 4 | balcony | Balcony area |
| 5 | balcony_window | Glass partition |
| 6 | bedroom | Bedroom |
| 7 | living_room | Living room |
| 8 | kitchen | Kitchen |
| 9 | bathroom | Bathroom/WC |

## Architecture

- **Model**: SegFormer-B2 (nvidia/segformer-b2-finetuned-ade-512-512)
- **Input**: 512x512 (resize with aspect ratio preservation + padding)
- **Loss**: Combined CrossEntropy + Tversky (alpha=0.3, beta=0.7)
- **Output**: 10-class pixel mask -> vectorized polygons in 0-1000 coords

## File Overview

| File | Purpose |
|------|---------|
| `config.py` | Class definitions, hyperparameters, model config |
| `prepare_cubicasa.py` | CubiCasa5K dataset preparation |
| `floorplan_generator.py` | Procedural apartment layout generation |
| `generate_synthetic.py` | Synthetic image + mask renderer |
| `dataset.py` | PyTorch Dataset with augmentation |
| `loss.py` | Tversky + Combined loss functions |
| `train.py` | Training script with early stopping |
| `vectorize.py` | Mask -> polygon vectorization |
| `predict.py` | Full inference pipeline |
| `cog.yaml` | Replicate deployment config |
| `cog_predict.py` | Replicate predictor |
| `test_vectorize.py` | Vectorization unit tests |
