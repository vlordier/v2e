# 📐 Dimensionality Analysis: It's Not Just "3D"!

## 🎯 **What We're Actually Building**

### **Current Name:** "3D-Aware Event Prediction" ❌
### **Accurate Name:** "4D Spatiotemporal + 6D Motion Event Prediction" ✅

---

## 📊 **Input Dimensions**

| Dimension | Type | Data Source | Shape |
|-----------|------|-------------|-------|
| **2D Spatial** | Visual | RGB images | (H=260, W=346) |
| **1D Temporal** | Time | Event window | (T=33ms) |
| **3D Motion** | Inertial | IMU accelerometer | (ax, ay, az) |
| **3D Rotation** | Inertial | IMU gyroscope | (wx, wy, wz) |
| **1D Depth** | Geometric | Scene depth estimate | (scalar per frame) |

**Total Input: 10 dimensions!**

---

## 🔄 **Model Architecture**

```
Input:
  - RGB frames:      (B, 1, H, W)      # 2D spatial
  - IMU sequence:    (B, T=50, 6)      # 3D motion + 3D rotation over time
  - Time window:     33ms              # 1D temporal

Processing:
  - RGB encoder:     Conv2D layers     # 2D spatial features
  - IMU encoder:     LSTM              # Temporal + 6D motion features
  - Fusion:          FiLM modulation   # Spatiotemporal + motion fusion
  - Depth head:      MLP               # 1D depth from features

Output:
  - Events:          (B, 2, H, W)      # 2D spatial event maps
  - Depth:           (B, 1)            # 1D scene depth
```

---

## 🏷️ **Naming Options**

### **Option 1: "4D Spatiotemporal Event Prediction"**
- ✅ Accurate (2D spatial + 1D temporal + 1D depth)
- ❌ Doesn't capture 6D IMU motion

### **Option 2: "Motion-Aware Event Prediction"**
- ✅ Captures IMU integration
- ❌ Doesn't capture depth or spatiotemporal

### **Option 3: "Spatiotemporal + 6D Motion Event Prediction"**
- ✅ Most accurate
- ❌ Too long

### **Option 4: "4D+6D Event Prediction"**
- ✅ Concise and accurate
- ⚠️ Might be confusing

### **Option 5: "Multimodal Spatiotemporal Event Prediction"**
- ✅ Captures RGB + IMU fusion
- ✅ Captures spatiotemporal nature
- ✅ Concise

---

## 🎯 **Recommended: "Multimodal Spatiotemporal Event Prediction"**

**Why:**
1. ✅ **Multimodal** - RGB + IMU fusion
2. ✅ **Spatiotemporal** - 2D space + 1D time
3. ✅ **Event Prediction** - Clear task definition
4. ✅ **Concise** - Easy to say/write
5. ✅ **Accurate** - Describes what we do

---

## 📝 **Files to Update**

| File | Current | Proposed |
|------|---------|----------|
| `train.py` | "Multimodal Spatiotemporal model" | "Multimodal spatiotemporal model" |
| `COMPARISON_RESULTS.md` | "3D-Aware Event Prediction" | "Multimodal Spatiotemporal Event Prediction" |
| `V6_RESULTS.md` | "Multimodal Spatiotemporal architecture" | "Multimodal spatiotemporal architecture" |
| All docs | "Multimodal Spatiotemporal" | "Multimodal spatiotemporal" |

---

## 🔬 **Scientific Accuracy**

### **What "3D" Usually Means**
- 3D geometry (x, y, z coordinates)
- 3D point clouds
- 3D voxel grids

### **What We're Doing**
- 2D images (H, W)
- 1D time (T)
- 6D motion (ax, ay, az, wx, wy, wz)
- 1D depth (scalar)

**We're NOT doing:**
- ❌ 3D reconstruction
- ❌ Point cloud generation
- ❌ Voxel prediction

**We ARE doing:**
- ✅ 2D event prediction from 2D images
- ✅ Using 6D IMU motion as additional input
- ✅ Estimating 1D scene depth as auxiliary task

---

## 🎓 **Conclusion**

**"Multimodal Spatiotemporal" is misleading.** We should use **"Multimodal Spatiotemporal Event Prediction"** because:

1. It's more accurate (we use multiple modalities: RGB + IMU)
2. It captures the spatiotemporal nature (2D space + 1D time)
3. It doesn't overclaim (we're not doing 3D reconstruction)
4. It's scientifically precise

---

*Generated: March 31, 2026*  
*Recommendation: Update all documentation to use "Multimodal Spatiotemporal"*
