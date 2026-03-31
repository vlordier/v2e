# Efficiency Improvements for 3D-Aware Event Prediction

## Current State

| Model | Params | event_bpb | Training Time |
|-------|--------|-----------|---------------|
| **3D-Aware (Full)** | 1.19M | 0.000141 | 15 min |
| **Naive (RGB only)** | 0.40M | 0.000190 | 15 min |

**Goal:** Maintain accuracy while reducing model size and inference latency.

---

## 1. Event Hallucination Prevention ✅

### Problem
Models can "hallucinate" events by predicting too many events, artificially improving MSE but wasting bandwidth.

### Solution: Event Rate Regularization
```python
# Penalize when predicted event rate >20% higher than ground truth
pred_event_rate = pred_events.abs().mean()
gt_event_rate = gt_events.abs().mean()
rate_ratio = pred_event_rate / (gt_event_rate + 1e-6)
rate_penalty = F.relu(rate_ratio - 1.2) ** 2

loss = event_loss + rate_weight * rate_penalty
```

**Benefits:**
- Prevents bandwidth waste
- More realistic event rates
- Better compression in practice

---

## 2. Knowledge Distillation 🎓

### Teacher-Student Setup

| Model | Params | Architecture |
|-------|--------|--------------|
| **Teacher** | 1.19M | 3D-aware, multi-scale FiLM, depth head |
| **Student** | ~0.30M | Compact RGB+IMU, single-scale fusion |

### Distillation Loss
```python
# Teacher provides soft targets
with torch.no_grad():
    teacher_events = teacher(images, imu_seq)

# Student learns from teacher + ground truth
distill_loss = MSE(student_events, teacher_events)
gt_loss = MSE(student_events, gt_events)

loss = (1 - α) * gt_loss + α * distill_loss
```

**Expected Results:**
- Student achieves 90-95% of teacher accuracy
- 4x smaller model
- 3-4x faster inference

### Usage
```bash
# Train teacher first
uv run python train.py

# Distill to student
uv run python train_distill.py
```

---

## 3. Structured Pruning ✂️

### Approach
Remove redundant channels/filters based on importance scores.

### Implementation Strategy
```python
# 1. Train model to convergence
# 2. Compute L1 norm of each filter
# 3. Remove bottom X% of filters
# 4. Fine-tune pruned model

# Example: 50% channel pruning
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        # Compute importance
        importance = module.weight.abs().sum(dim=(1,2,3))
        # Get threshold for bottom 50%
        threshold = importance.kthvalue(importance.numel() // 2).values
        # Create mask
        mask = (importance > threshold).float()
        # Apply mask
        module.weight.data *= mask.view(-1, 1, 1, 1)
```

**Expected Compression:**
- 50% pruning → 2x smaller, minimal accuracy loss
- 75% pruning → 4x smaller, ~5-10% accuracy loss

---

## 4. Quantization 🔢

### Post-Training Quantization (PTQ)
```python
# Convert to INT8
model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)
```

**Benefits:**
- 4x smaller model size
- 2-3x faster inference (on supported hardware)
- Minimal accuracy loss (<2%)

### Quantization-Aware Training (QAT)
```python
# Prepare for QAT
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model)

# Train with quantization simulation
# ... training loop ...

# Convert to quantized model
model_quantized = torch.quantization.convert(model_prepared)
```

**Benefits:**
- Better accuracy than PTQ
- Same size/speed benefits

---

## 5. Architecture Optimizations 🏗️

### A. Reduce FiLM Scales
Current: Multi-scale FiLM at 3 encoder layers
Optimized: Single-scale FiLM at bottleneck only

**Savings:** -6K params, -10% FLOPs

### B. Smaller Depth Head
Current: MLP (32→64→32→1)
Optimized: MLP (32→16→1)

**Savings:** -2K params, negligible accuracy loss

### C. Grouped Convolutions
Replace standard convolutions with grouped convolutions:
```python
# Standard
nn.Conv2d(32, 32, 3, padding=1)

# Grouped (4 groups)
nn.Conv2d(32, 32, 3, padding=1, groups=4)
```

**Savings:** 4x fewer params in conv layers

---

## 6. Expected Results Summary

| Optimization | Params | event_bpb | Speed | Memory |
|--------------|--------|-----------|-------|--------|
| **Baseline (3D-Aware)** | 1.19M | 0.000141 | 1.0x | 1.0x |
| **+ Rate Regularization** | 1.19M | 0.000145 | 1.0x | 1.0x |
| **+ Knowledge Distillation** | 0.30M | 0.000155 | 3.5x | 4.0x |
| **+ 50% Pruning** | 0.60M | 0.000150 | 1.8x | 2.0x |
| **+ INT8 Quantization** | 0.30M | 0.000160 | 5.0x | 4.0x |
| **All Combined** | 0.15M | 0.000180 | 7.0x | 8.0x |

**Trade-off:** 28% accuracy loss for 7x speedup and 8x memory reduction.

---

## 7. Recommended Pipeline

### For Research (Maximum Accuracy)
1. Train full 3D-aware model
2. Apply rate regularization
3. Fine-tune with distillation (self-distillation)

### For Deployment (Balanced)
1. Train full 3D-aware teacher
2. Distill to compact student
3. Apply 50% structured pruning
4. Fine-tune pruned model

### For Edge Devices (Maximum Efficiency)
1. Train full 3D-aware teacher
2. Distill to tiny student
3. Apply quantization-aware training
4. Convert to INT8

---

## 8. Implementation Priority

### High Priority (Easy wins)
1. ✅ Event rate regularization (already implemented)
2. Knowledge distillation (script ready)
3. Post-training quantization (1 line of code)

### Medium Priority (Moderate effort)
4. Structured pruning (requires careful tuning)
5. Architecture optimization (design choices)

### Low Priority (Complex)
6. Quantization-aware training (requires retraining)
7. Neural architecture search (expensive)

---

## 9. Next Steps

1. **Run distillation experiment**
   ```bash
   uv run python train_distill.py
   ```

2. **Measure compression ratio**
   - Compare student vs teacher event_bpb
   - Measure inference latency
   - Check memory footprint

3. **Iterate on student architecture**
   - Try different channel widths
   - Experiment with fusion strategies
   - Test depth-wise separable convolutions

---

*For questions or issues, see COMPARISON_RESULTS.md*
