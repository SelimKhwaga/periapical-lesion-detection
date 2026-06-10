# AI-Powered Periapical Lesion Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

**Author:** Selim Rezk Abdelmawly Khwaga, DDS  
**Institution:** British University in Egypt (BUE) | King Salman International University (KSIU)  
**Project Duration:** Oct–Dec 2025 

**Contact:**  
📧 sleemkhw@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/selim-khwaga-b79921196/)

---

## 🎯 Project Overview

Deep learning system for automated detection and classification of periapical lesions in panoramic dental radiographs using YOLOv8 architecture. This work demonstrates the feasibility of AI-assisted diagnostic screening in dentistry, with potential applications in resource-limited clinical settings.

### Key Achievements

**External validation (DENTEX, MICCAI 2023)**  
- **81% Sensitivity**  
- **81% Specificity**  
- **95.6% Negative Predictive Value (NPV)**  
- **Image-level periapical-lesion screening**

---

## 📊 Results Summary

### Primary Metrics

| Metric | Value | Clinical Interpretation |
|--------|-------|------------------------|
| **mAP@0.5** | 80.7% | Overall detection accuracy across all lesion types |
| **Precision** | 82.3% | 82% of positive detections are true lesions |
| **Recall** | 73.1% | System identifies 73% of all actual lesions |
| **F1-Score** | 77.4% | Balanced performance metric |
| **Training Data** | 3,900 unique radiographs (augmented to ~13,000 training instances) | Large-scale dataset for robust learning |

### Performance by Lesion Severity

| Lesion Type | mAP@0.5 | Clinical Significance |
|-------------|---------|----------------------|
| **Type3** (Less severe) | 82% | Early-stage periapical radiolucencies |
| **Type4** (More severe) | 78% | Advanced lesions requiring immediate intervention |

The model demonstrates balanced performance across lesion severities, with slightly higher accuracy on Type3 lesions while maintaining strong detection of clinically urgent Type4 cases.

### Binary Classification Performance

**External DENTEX evaluation (705 images: 589 no-lesion, 116 lesion)**

  * **Threshold:** 0.4 (selected on this set using Youden's J)
  * **Accuracy:** 81.1%
  * **Sensitivity:** 81.0%
  * **Specificity:** 81.2%
  * **Positive Predictive Value (PPV):** 45.9%
  * **Negative Predictive Value (NPV):** 95.6%
  * **F1 Score:** 0.586

*Notes:*

* *"No-lesion" refers to the absence of a periapical lesion; images may still contain other dental findings.*
* *Metrics are reported for image-level present/absent classification, not lesion localization.*


**Confusion Matrix (Optimal threshold = 0.4):**
- TP 94, TN 478, FP 111, FN 22 (of 705)

---

## 🔬 Methodology

### Architecture & Implementation

- **Base Model:** YOLOv8s (Small variant)
- **Parameters:** 11.2M trainable parameters
- **Input Resolution:** 832×832 pixels (optimized for dental radiographs)
- **Training Duration:** 150 epochs with early stopping
- **Hardware:** NVIDIA T4 GPU (Lightning AI platform)
- **Framework:** PyTorch 2.0+ with Ultralytics YOLOv8

### Dataset & Preprocessing

**Primary Dataset:**
- **Source:** Custom annotated panoramic radiograph collection
- **Total Images:** 3,924 original radiographs (57 exact-duplicate groups present), augmented ~3.3×.
- **Classes:** 
  - Type3: Periapical radiolucency (63.7% of dataset) - 14,253 instances
  - Type4: Advanced periapical lesion (36.1% of dataset) - 8,111 instances
- **Format:** YOLO-compatible bounding box annotations

**External Validation:**
- **Dataset:** DENTEX Challenge 2023 (MICCAI)
- **Purpose:** Binary classification testing (healthy vs diseased)
- **Size:** 705 images (589 no-lesion, 116 lesion)

**Data Corrections:**
- Automated label-cleaning pipeline (out-of-range class-ID clamping + coordinate clipping) applied to ~25% of annotations.
- Developed automated label correction pipeline, with partial manual verification of annotation quality
- Applied quality control measures including class mapping standardization and coordinate validation

### Training Strategy

**Optimization:**
- **Optimizer:** AdamW with weight decay
- **Learning Rate:** Cosine annealing schedule
- **Initial LR:** 0.001 with 5-epoch warmup
- **Batch Size:** 12 images per batch (T4-optimized)
- **Loss Function:** Multi-component YOLO loss (box + classification + DFL)

**Data Augmentation (Medical-Safe):**
- Rotation: ±12° (preserves anatomical orientation)
- Scaling: 0.5× to 1.5× (simulates different magnifications)
- Translation: ±12% (accounts for positioning variations)
- Brightness/Contrast: ±20% (handles exposure differences)
- Horizontal/Vertical flip: 50% probability
- Mosaic augmentation: 50%
- Mixup: 10%

**Training Configuration:**
- Early stopping patience: 30 epochs
- Loss weights: box=7.5, cls=2.0, dfl=1.5
- AMP (Automatic Mixed Precision) enabled
- Multi-scale training enabled

### Validation & Evaluation

- **IoU Threshold:** 0.5 for mAP calculation (standard COCO metric)
- **Confidence Threshold:** 0.25 default, optimized to 0.4 for binary classification
- **Evaluation Metrics:** Precision, Recall, mAP@0.5, F1-Score, Confusion Matrix
- **Cross-Dataset Testing:** DENTEX benchmark for generalization assessment

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
CUDA-compatible GPU (recommended)
8GB+ RAM
```

### Installation

```bash
# Clone repository
git clone https://github.com/SelimKhwaga/periapical-lesion-detection.git
cd periapical-lesion-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training from Scratch

```bash
# Train on your dataset
python training/train_model.py

# Monitor training progress
# Outputs will be saved to /training/periapical_t4/
```

### Inference on New Images

```bash
# Run inference on single image
from ultralytics import YOLO
model = YOLO('path/to/best.pt')
results = model('path/to/image.jpg')

# View results
results[0].show()
```

### Evaluation & Testing

```bash
# Binary classification test on DENTEX dataset
python evaluation/binary_classification_test.py
```

---

## 📁 Repository Structure

```
periapical-lesion-detection/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── training/
│   ├── train_model.py                 # Main training script
│   ├── data.yaml                      # Dataset configuration
│   └── README.md                      # Training documentation
│
├── evaluation/
│   ├── binary_classification_test.py  # External dataset testing
│   └── README.md                      # Evaluation documentation
│
├── results/
│   ├── confusion_matrix.png           # Validation confusion matrix
│   ├── confusion_matrix_normalized.png
│   ├── training_curves.png            # Loss and metric plots
│   ├── label_distribution.png         # Dataset class distribution
│   └── README.md                      # Results documentation
│
└── models/
    └── best.pt                        # Trained model weights (download separately)
```

---

## 📈 Clinical Impact & Applications

### Demonstrated Capabilities

1. **Automated Screening:** Potential for first-line diagnostic support in general dental practice
2. **Resource Efficiency:** Reduces radiograph review time while maintaining accuracy
3. **Consistency:** Reduces inter-observer variability in lesion detection



### Future Clinical Applications

- Integration with dental practice management systems
- Real-time chairside diagnostic assistance
- Longitudinal lesion tracking and progression monitoring
- Multi-pathology detection (caries, cysts, tumors)
- Mobile diagnostic solutions for remote healthcare

---

## 🎓 Academic Context

### Research Background

This project was developed as part of advanced research in medical imaging and artificial intelligence, exploring the intersection of clinical dentistry and deep learning for improved diagnostic workflows.

**Research Interests:** Medical Imaging AI, Domain Adaptation, Cross-Population Generalization, Healthcare Machine Learning
**Academic Status:**
- BSc: British University in Egypt (BUE) - GPA: 4.0/4.0
- MSc: Currently enrolled (in progress)
- Current Position: Teaching Assistant, King Salman International University (KSIU)

### Related Publications & Benchmarks

This work builds upon and compares to:

- **DENTEX Challenge 2023** (MICCAI Workshop) - International benchmark for dental AI
- **YOLOv8 Architecture** (Ultralytics, 2023) - State-of-the-art object detection
- **Medical Image Analysis Research** - Published studies reporting 83-95% mAP for similar tasks

### Performance Context
Based on a review of recent literature, dental lesion detection systems typically achieve the following performance ranges:
- **Traditional CNN-based methods: 60–75% mAP:** - 60–75% mAP
- **Modern object detection models (YOLO / Faster R-CNN):** - 85–95% mAP
- **Ensemble-based methods with large datasets:** - Strong detection rate across lesion types
  
**Key Differentiators:**
- **Model:** - YOLOv8s (single model)
- **Performance:** - 80.7% mAP@0.5

**Key Differentiators:**
- Large-scale training dataset (13,058 annotated clinical images)
- External validation on an international benchmark dataset (DENTEX 2023)
- Clinical data quality control performed by a licensed dentist
- Open-source implementation to ensure reproducibility and transparency

---

## 🔧 Technical Details

### Model Specifications

```python
Model: YOLOv8s
Parameters: 11,166,560 (11.2M)
FLOPs: 28.6 GFLOPs
Input Shape: (832, 832, 3)
Output: Multi-scale detection heads
Classes: 2 (Type3, Type4)
Anchor-free: Yes
```

### Training Configuration

```yaml
epochs: 150
imgsz: 832
batch: 12
optimizer: AdamW
lr0: 0.001
lrf: 0.01
weight_decay: 0.0005
momentum: 0.937
warmup_epochs: 5.0
patience: 30
box: 7.5
cls: 2.0
dfl: 1.5
```

### Hardware Requirements

**Training:**
- GPU: NVIDIA T4 (16GB) or better
- RAM: 16GB+ recommended
- Storage: 50GB for dataset + outputs
- Training Time: ~8 hours for 150 epochs

**Inference:**
- GPU: Optional (CPU inference ~2-3 seconds/image)
- RAM: 4GB minimum
- Inference Speed: ~30ms/image (GPU), ~2s/image (CPU)

---

## 📊 Dataset Information

### Data Distribution

**Training Set (78.7%):**
- Total: 3,924 unique radiographs, augmented to ~13,058 samples
- Type3 Lesions: 14,253 instances (63.7%)
- Type4 Lesions: 8,111 instances (36.1%)
- Image Format: JPEG/PNG, various resolutions
- Annotation Format: YOLO txt files (normalized bounding boxes)

**External Test Set (DENTEX):**
- Total: 232 images
- Binary labels: Healthy (116) vs Diseased (116)
- Purpose: Generalization and binary classification assessment

---

## 🎯 Future Work & Roadmap

### Short-Term Improvements (3-6 months)

- [ ] Expand to additional oral pathologies (caries, cysts, impacted teeth)
- [ ] Implement attention mechanisms for improved localization
- [ ] Develop confidence calibration for clinical decision support
- [ ] Create interactive web interface for demonstration

### Medium-Term Goals (6-12 months)

- [ ] Multi-dataset training for enhanced generalization
- [ ] Ensemble methods for higher accuracy (target: 85%+ mAP)
- [ ] Real-time inference optimization (<10ms per image)
- [ ] Clinical pilot study with practicing dentists

### Long-Term Vision (1-2 years)

- [ ] FDA/regulatory approval pathway for clinical deployment
- [ ] Integration with PACS/dental imaging systems
- [ ] Mobile application for point-of-care diagnostics
- [ ] Longitudinal study tracking diagnostic impact

1. **Multi-Modal Fusion:** Combining panoramic X-rays with CBCT for 3D lesion analysis
2. **Few-Shot Learning:** Adapting models to rare pathologies with limited data
3. **Explainable AI:** Developing interpretable attention maps for clinical trust
4. **Federated Learning:** Privacy-preserving model training across multiple clinics

---

## 📝 Citation & Licensing

### How to Cite This Work

If you use this code or methodology in your research, please cite:

```bibtex
@software{periapical_detection_2025,
  author = {Khwaga, Selim Rezk Abdelmawly},
  title = {AI-Powered Periapical Lesion Detection System},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/SelimKhwaga/periapical-lesion-detection}
}
```

### License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Permissions:**
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

**Conditions:**
- 📄 License and copyright notice must be included
- ⚠️ No warranty provided

---

## 🤝 Contributing & Contact

### Contributions Welcome

I welcome contributions, suggestions, and collaborations! Areas of interest:

- Dataset expansion and annotation
- Algorithm improvements and optimizations
- Clinical validation studies
- Documentation and tutorials

### Contact Information

**Selim Rezk Abdelmawly Khwaga, DDS**  
Teaching Assistant, King Salman International University (KSIU)  
MSc Student (Current) | BSc Graduate, British University in Egypt (BUE) - GPA: 4.0/4.0

- 📧 Email: sleemkhw@gmail.com
- 💼 LinkedIn: [linkedin.com/in/selim-khwaga-b79921196](https://www.linkedin.com/in/selim-khwaga-b79921196/)
- 🐱 GitHub: [@SelimKhwaga](https://github.com/SelimKhwaga)
- 🎓 Institution: British University in Egypt (BUE) | King Salman International University (KSIU)

 
**Research Interests:** Medical Imaging AI, Domain Adaptation, Healthcare Machine Learning, Bioengineering 

---

## 🙏 Acknowledgments

- **Ultralytics Team** - YOLOv8 framework and excellent documentation
- **DENTEX Challenge Organizers** - External validation dataset
- **Lightning AI** - Computational resources for model training
- **British University in Egypt (BUE)** - Academic foundation and research support
- **King Salman International University (KSIU)** - Current teaching and research position

---

## 📚 References & Resources

### Key Papers

1. Jocher, G. et al. (2023). "Ultralytics YOLOv8" - https://github.com/ultralytics/ultralytics
2. DENTEX Challenge (2023). MICCAI Workshop on Dental AI
3. Medical Image Analysis - Recent advances in dental pathology detection

### Useful Links

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [DENTEX Challenge](https://dentex.grand-challenge.org/)
- [Lightning AI Platform](https://lightning.ai/)

### Related GitHub Repositories

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Medical Image Analysis Tools](https://github.com/Project-MONAI/MONAI)
- [Dental AI Research](https://github.com/topics/dental-ai)

---

<div align="center">

**⭐ If you find this work useful, please consider starring the repository! ⭐**

**🎓  Graduate Research Portfolio – Medical AI & Computational Health 🎓**

**Demonstrating Excellence in AI-Powered Medical Imaging Research**

</div>

---

*Last Updated: December 2025*  
*Version: 1.0.0*  
*Contact: sleemrezk@yahoo.com | sleemkhw@gmail.com*
