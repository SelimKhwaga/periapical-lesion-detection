# Training Scripts
   
   This directory contains the training pipeline for periapical lesion detection.
   
   ## Files
   
   - **train_model.py** - Main training script with YOLOv8
     - Configurable hyperparameters
     - Medical-safe data augmentation
     - AdamW optimizer with cosine scheduling
     - Early stopping support
   
   - **data.yaml** - Dataset configuration
     - Defines Type3 and Type4 classes
     - Training/validation split paths
     - Dataset statistics
   
   ## Usage
```bash
   python training/train_model.py
```
   
   ## Training Configuration
   
   - Model: YOLOv8s (11.2M parameters)
   - Image size: 832×832
   - Batch size: 16
   - Epochs: 150 with early stopping
   - Platform: Lightning AI (NVIDIA T4 GPU)
   
   ## Results
   
   - Final mAP@0.5: 80.7%
   - Precision: 82.3%
   - Recall: 73.1%
