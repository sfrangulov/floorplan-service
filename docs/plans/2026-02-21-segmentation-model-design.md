# Floorplan Recognition: Semantic Segmentation Model Design

**Date**: 2026-02-21
**Status**: Approved

## Problem

Current Gemini-based approach has three critical issues:
1. **Coordinate precision** - LLM "hallucinates" coordinates (generates numbers autoregressively, not from pixel data)
2. **Classification accuracy** - Confuses doors/windows, misses elements depending on prompt
3. **Robustness** - Works on some floorplans, produces garbage on others

## Approach: Semantic Segmentation + Vectorization

Replace the LLM-based pipeline with a dedicated segmentation model that classifies every pixel, then vectorize the masks into polygons.

### Why This Over Alternatives

| Approach | Accuracy | Speed | Robustness | Complexity |
|----------|----------|-------|------------|------------|
| **Semantic Segmentation** (chosen) | Pixel-perfect | <1s | Deterministic | Medium |
| LLM prompting (current) | Low | 3-10s | Non-deterministic | Low |
| YOLO-Seg | Good | 20-30ms | Good | Low |
| VLM fine-tune (Qwen2.5-VL) | Medium | 500-2000ms | Medium | High |

Semantic segmentation wins on quality because each pixel is classified independently - no coordinate hallucination, deterministic output.

## Architecture

### Model: SegFormer (MitUNet-style)

- **Encoder**: SegFormer-B2 (Mix-Transformer) - captures global context (wall continuity across the floor plan)
- **Decoder**: U-Net decoder with spatial + channel attention - recovers fine details (thin walls 2-5px)
- **Loss**: Tversky loss (alpha=0.3, beta=0.7) - emphasizes recall for thin structures
- **Input**: 512x512 (resize with aspect ratio preservation + padding)
- **Parameters**: ~25-30M
- **Base**: `nvidia/segformer-b2-finetuned-ade-512-512` from HuggingFace

### Segmentation Classes (10)

| ID | Class | Description |
|----|-------|-------------|
| 0 | background | Everything else |
| 1 | wall | Exterior and interior walls |
| 2 | door | Doors (including swing arcs) |
| 3 | window | Windows |
| 4 | balcony | Balcony area |
| 5 | balcony_window | Glass partition (balcony-apartment) |
| 6 | bedroom | Bedroom area |
| 7 | living_room | Living room area |
| 8 | kitchen | Kitchen area |
| 9 | bathroom | Bathroom / WC area |

## Data Strategy

### Stage 1: Pre-training on CubiCasa5K
- 5,000 real floorplans with SVG annotations
- Map CubiCasa's 80+ categories to our 10 classes
- Gives the model fundamental understanding of floorplan elements

### Stage 2: Synthetic dataset (2,000-3,000 images)
- Procedural generation of apartment layouts:
  - 1-4 bedrooms, kitchen, bathroom, living room
  - Variable wall thickness, door/window styles
  - Different visual styles (line weight, hatching, text)
  - Scan artifacts, slight rotation, JPEG compression noise
- Automatic ground truth: render each class as a separate layer -> pixel-perfect masks
- Output: COCO segmentation format (images + per-class polygon annotations)

### Stage 3: Fine-tune on real data
- Start with 4 test images + manual annotation
- Continuously improve as real data accumulates

### Existing Datasets to Leverage
- **CubiCasa5K**: 5,000 real CAD-derived floorplans
- **RPLAN**: 80,000+ residential floorplans
- **SYNBUILD-3D**: 6.2M 3D models with floorplan renders

## Pipeline: Image -> JSON

```
Input image (JPEG/PNG/WebP)
  |
  v
[1] Preprocessing (Sharp)
  - Resize to 512x512 (preserve aspect ratio + padding)
  - Record scale factors for coordinate mapping
  |
  v
[2] Semantic Segmentation (SegFormer)
  - Output: 512x512 mask with 10 class channels
  - Per-pixel classification with softmax
  |
  v
[3] Per-class binary mask extraction
  - Threshold softmax output per class
  |
  v
[4] Contour extraction (OpenCV findContours, CHAIN_APPROX_SIMPLE)
  - Extract contour polygons for each class
  |
  v
[5] Polygon simplification (Douglas-Peucker, epsilon=2-3px)
  - Reduce point count while preserving shape
  |
  v
[6] Scale to original image dimensions
  - Undo padding and resize
  |
  v
[7] Normalize to 0-1000 coordinate space
  - X and Y normalized independently
  - 1 decimal place precision
  |
  v
[8] Assemble JSON v3 format
  - {version, image_width_meters, wall[], door[], window[],
     balcony, balcony_window, bedroom[], living_room[], kitchen[], bathroom[]}
```

### Special handling:
- **Walls**: preserve both edges (wall thickness visible in contours)
- **Rooms**: flood-fill from mask -> one polygon per room instance
- **Doors/Windows**: bounding rectangles (5 points: 4 corners + closing)

## Training

### Environment
- **Hardware**: Google Colab Pro (A100, 40GB VRAM) or RunPod
- **Framework**: PyTorch + HuggingFace Transformers
- **Estimated time**: ~4-6h pre-training on CubiCasa5K, ~2h fine-tuning on synthetic

### Hyperparameters
- Optimizer: AdamW (lr=6e-5, weight_decay=0.01)
- Scheduler: Polynomial decay
- Batch size: 8-16 (depends on GPU memory)
- Epochs: 50-100 with early stopping (patience=10)
- Data augmentation: random crop, flip, rotation, color jitter, elastic deformation

## Deployment

### Recommended: Replicate Custom Model
- Package model in Cog container
- Deploy on Replicate
- Call from Node.js via `replicate` npm package (already in dependencies)
- Cold start: ~10-15s, warm inference: ~1-2s
- Cost: ~$0.001-0.005 per inference

### Alternative: Python microservice
- FastAPI service alongside Fastify
- Docker compose: node + python containers
- Inference: ~200-500ms on GPU
- More control, lower latency, higher ops overhead

## Success Criteria

1. **Wall detection**: IoU > 0.85 against reference polygons
2. **Door/Window detection**: Count accuracy > 90%, IoU > 0.70
3. **Room classification**: > 85% correct room type assignment
4. **Robustness**: Consistent results across 4 test images (no garbage output)
5. **Speed**: < 2 seconds total (segmentation + vectorization)
6. **Determinism**: Same input -> same output (no LLM randomness)

## Implementation Phases

1. **Synthetic data generator** - Node.js tool to generate training images + masks
2. **CubiCasa5K data preparation** - Download, convert annotations, map classes
3. **Model training pipeline** - Colab notebook for SegFormer fine-tuning
4. **Post-processing module** - Mask -> polygon vectorization in Python
5. **Replicate deployment** - Cog container + deployment
6. **Integration** - Replace Gemini call in `src/gemini.js` with Replicate call
7. **Benchmarking** - Compare A9 (segmentation) vs A1-A8 approaches
